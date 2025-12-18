import os
import re
import json
import time
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from SendNewsletter import send_html_file

# --- NEW IMPORT ---
from tavily import TavilyClient 

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# -------------------------------------
# CONFIG
# -------------------------------------

GEMINI_MODEL = "gemini-3-pro-preview" # Recommended stable model
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("Missing env: GOOGLE_API_KEY")
if not TAVILY_API_KEY:
    raise ValueError("Missing env: TAVILY_API_KEY")

SEARCH_CONFIG = {
    "max_iterations": 10,
    "min_valid_sources": 15,
    "max_results_per_search": 5, # Tavily gives high quality, so we need fewer per batch
}

OUTPUT_CONFIG = {
    "output_dir": "Output",
    "html_filename_template": "monthly_digest_{timestamp}.html",
}

# -------------------------------------
# AI AGENT PROMPTS
# -------------------------------------

# 1. RESEARCHER: Finds the data
RESEARCHER_PROMPT = """
You are an expert Business Research Analyst.

Your task is to generate short, effective search queries to find 
RECENT and RELEVANT information on the user’s topic.

RULES:
1. Keep queries short: 2–6 keywords.
2. Prioritize recency: focus on developments, trends, releases, reports, signals.
3. Include geographic modifiers ONLY when useful ("Nordics", "Sweden", "Finland", etc.).
4. NEVER invent facts. Only generate search queries.
5. After each iteration, refine the query based on editor feedback.

COMMAND FORMAT:
SEARCH: your query
"""

# 2. EDITOR: Filters the data
EDITOR_PROMPT = """
You are the quality assurance officer.

Your task is to evaluate search results and decide which are VALID signals.

A result is VALID if:
1. It is PUBLISHED within the last 30 days.
2. It is RELEVANT to the topic current topic.
3. The content has implications for the Nordics, more specifically Sweden.
4. Headlines OR descriptions may indicate relevance. Partial signals are allowed.
5. Reject only if clearly irrelevant (e.g., food recipes, sports, politics, unrelated tech etc.).

Your output MUST be valid JSON of this form:
{
  "valid_ids": [0, 3, 4],
  "feedback": "Short explanation for researcher."
}
"""

# 3. WRITER: Synthesizes the text
WRITER_PROMPT = """
You are a Senior Consultant at Differ.

Write a "Monthly Business Digest" based on the provided Validated Facts.

TONE & STYLE:
- Insightful, forward-looking, and concise.
- Use "We" tone (Differ's voice).
- Structure:
  1. Executive Summary (The "So What?")
  2. Key Strategic Themes (Group findings by trend, e.g., "AI Adoption", "Sustainability")
  3. Implications (What this means for leaders and businesses)

CRITICAL:
- Do NOT output HTML here, just plain text with Markdown.
- CITE sources inline using [Source Name](URL).
"""

# 4. DESIGNER: Formats the HTML (The "Differ.se" Look)
HTML_FORMATTER_PROMPT = """
You are the Lead Web Designer at Differ.

Your task is to take the provided Report Text and Sources and convert them into a 
stunning, responsive HTML document that matches the design of https://www.differ.se/

DESIGN GUIDELINES (Differ Aesthetic):
1.  **Typography**: Use 'Inter', 'Helvetica Neue', or system-ui. Clean, crisp, high readability.
2.  **Colors**: 
    - Background: White (#FFFFFF) or very light grey (#F9F9F9).
    - Text: Dark Grey/Black (#1A1A1A).
    - Accents: Use a deep distinct Teal/Green (e.g., #004D40) or Differ's signature muted green/grey for headers.
3.  **Layout**:
    - **Header**: Minimalist. Text logo "Differ" on the left.
    - **Container**: Max-width 900px, centered, ample whitespace (padding).
    - **Headings**: Big, bold, uppercase or heavy weight (font-weight: 700+).
    - **Cards**: Group key themes into subtle "cards" with thin borders or soft shadows.
4.  **Footer**: 
    - Simple grey background.
    - Text: "Regeringsgatan 67, 111 56 Stockholm | info@differ.se".
5.  **Sources**: A clean, numbered list at the bottom, styled visibly but unobtrusively. The sources should also be clickable, both in the reference list at the end but also throughout the text.

INPUT: A raw report text + list of sources.
OUTPUT: Valid, standalone HTML5 code. Do not use Markdown fences.
"""

# -------------------------------------
# DATA CLASSES
# -------------------------------------

@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str
    query: str
    published_date: Optional[str] = None

@dataclass
class MonthlyDigest:
    title: str
    period: str
    generated_at: str
    content: str
    sources: List[SearchResult]
    metadata: Dict[str, Any]

# -------------------------------------
# PERFORMANCE TIMER
# -------------------------------------

class PerfTimer:
    def __init__(self, name: str):
        self.name = name
    def __enter__(self):
        self.start = time.perf_counter()
        print(f"⏱️ START: {self.name}")
    def __exit__(self, exc_type, exc, tb):
        elapsed = time.perf_counter() - self.start
        print(f"🏁 DONE: {self.name} ({elapsed:.2f}s)")

# -------------------------------------
# GENERATOR CLASS
# -------------------------------------

class Generator:

    def __init__(self):
        genai.configure(api_key=GOOGLE_API_KEY)
        
        # Initialize Tavily Client
        self.tavily = TavilyClient(api_key=TAVILY_API_KEY)

        self.researcher_log = []
        self.validated_results: List[SearchResult] = []
        self.seen_urls = set()
        self.iteration = 0

        self.safety = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

    def get_model(self, system_prompt: str, json_mode=False):
        cfg = {"response_mime_type": "application/json"} if json_mode else {}
        return genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=system_prompt,
            safety_settings=self.safety,
            generation_config=cfg
        )

    # ------------------------------------------------------
    # TAVILY SEARCH (Replaces DDGS/NewsAPI)
    # ------------------------------------------------------

    def perform_search(self, query: str) -> List[SearchResult]:
        """
        Uses Tavily API to find high-quality, relevant results.
        Tavily handles the scraping and parsing for us.
        """
        with PerfTimer(f"Tavily Search: {query}"):
            try:
                # Tavily arguments explained:
                # topic="news": Optimizes for recent events/articles
                # days=30: STRICTLY filters for last 30 days (Server-side)
                # search_depth="advanced": actually visits the page to get better content
                response = self.tavily.search(
                    query=query,
                    search_depth="advanced", 
                    topic="news", 
                    days=30,
                    max_results=SEARCH_CONFIG["max_results_per_search"]
                )
                
                tavily_results = response.get("results", [])
                results = []

                for item in tavily_results:
                    # 1. Deduplication
                    url = item.get("url", "")
                    if url in self.seen_urls:
                        continue
                    self.seen_urls.add(url)

                    # 2. Map to your SearchResult Class
                    # Note: Tavily returns 'content' which is a rich snippet/summary
                    results.append(SearchResult(
                        title=item.get("title", "No Title"),
                        url=url,
                        snippet=item.get("content", "")[:600], # Grab first 600 chars of content
                        source="Tavily",
                        query=query,
                        published_date=item.get("published_date", "Recent")
                    ))
                
                print(f"📰 Found {len(results)} new unique results.")
                return results

            except Exception as e:
                print(f"❌ Tavily error: {e}")
                return []

    # ------------------------------------------------------
    # ITERATIVE RESEARCH + EDITOR LOOP (Your Logic)
    # ------------------------------------------------------

    def run_research_cycle(self, topic: str, period: str):

        researcher = self.get_model(RESEARCHER_PROMPT)
        editor = self.get_model(EDITOR_PROMPT, json_mode=True)

        # Kickstart conversation
        self.researcher_log.append({"role": "user", "content": f"Topic: {topic}. Begin with one SEARCH query."})
        response = researcher.generate_content(self.to_history(self.researcher_log))
        self.researcher_log.append({"role": "model", "content": response.text})

        while len(self.validated_results) < SEARCH_CONFIG["min_valid_sources"]:
            self.iteration += 1

            if self.iteration > SEARCH_CONFIG["max_iterations"]:
                print("⛔ Max iterations reached.")
                break

            # Parse SEARCH command
            match = re.search(r"SEARCH:\s*(.+)", response.text)
            
            # --- ROBUSTNESS CHECK ---
            if not match:
                if "DONE" in response.text:
                    print("Researcher signal DONE.")
                    break
                print("🔄 Researcher failed to give query. Asking again.")
                self.researcher_log.append({"role": "user", "content": "You must provide a SEARCH: [query] or state DONE."})
                response = researcher.generate_content(self.to_history(self.researcher_log))
                self.researcher_log.append({"role": "model", "content": response.text})
                continue
            # ------------------------

            query = match.group(1).strip()
            print(f"\n🔍 ITERATION {self.iteration} — Query: {query}")

            # CALL TAVILY
            raw_results = self.perform_search(query)

            if not raw_results:
                feedback = "No new results found. Try a different angle."
                self.researcher_log.append({"role": "user", "content": feedback})
                response = researcher.generate_content(self.to_history(self.researcher_log))
                self.researcher_log.append({"role": "model", "content": response.text})
                continue

            # EDITOR EVALUATION
            editor_input = {
                "topic": topic,
                "results": [
                    {
                        "id": i,
                        "title": r.title,
                        "snippet": r.snippet,
                        "date": r.published_date
                    }
                    for i, r in enumerate(raw_results)
                ]
            }

            editor_json = json.dumps(editor_input, ensure_ascii=False)
            editor_response = editor.generate_content(editor_json)

            try:
                parsed = json.loads(editor_response.text)
            except:
                print("❌ Editor JSON parsing failed. Skipping.")
                continue

            valid_ids = parsed.get("valid_ids", [])
            feedback = parsed.get("feedback", "")

            # Store Validated Results
            for i in valid_ids:
                if 0 <= i < len(raw_results):
                    self.validated_results.append(raw_results[i])

            print(f"✔ Valid this round: {len(valid_ids)} | Total valid: {len(self.validated_results)}")

            # Send feedback to researcher
            self.researcher_log.append({"role": "user", "content": f"Editor Feedback: {feedback}. Total valid sources: {len(self.validated_results)}"})
            response = researcher.generate_content(self.to_history(self.researcher_log))
            self.researcher_log.append({"role": "model", "content": response.text})

    # ------------------------------------------------------
    # WRITER (Your Logic)
    # ------------------------------------------------------

    def call_writer(self, topic: str, period: str) -> str:
        writer = self.get_model(WRITER_PROMPT)

        facts = ""
        for r in self.validated_results:
            facts += f"Title: {r.title}\nURL: {r.url}\nSnippet: {r.snippet}\nDate: {r.published_date}\n\n"

        task = f"Topic: {topic}\nPeriod: {period}\n\nVALIDATED FACTS:\n{facts}"

        with PerfTimer("Writer"):
            response = writer.generate_content(task)

        return response.text

    # ------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------

    def to_history(self, messages):
        history = []
        for m in messages:
            # Simple role mapping for Gemini
            role = "user" if m["role"] == "user" else "model"
            history.append({"role": role, "parts": [m["content"]]})
        return history

# --- STEP 4: DESIGNER (HTML GENERATION) ---
    def save_report(self, report: MonthlyDigest):
        print("\n🎨 DESIGNER: Formatting HTML (Differ Style)...")
        
        # 1. Prepare Data for Designer Agent
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        sources_list_text = "\n".join([f"{i+1}. {s.title} - {s.url} ({s.published_date})" for i, s in enumerate(report.sources)])
        
        designer_prompt = f"""
        REPORT TITLE: {report.title}
        DATE: {report.generated_at}
        
        === CONTENT ===
        {report.content}
        
        === SOURCES LIST ===
        {sources_list_text}
        """

        # 2. Generate HTML
        designer = self.get_model(HTML_FORMATTER_PROMPT)
        response = designer.generate_content(designer_prompt)
        
        # Cleanup
        html_code = response.text.replace("```html", "").replace("```", "").strip()
        
        # 3. Save
        out_dir = Path(__file__).parent / OUTPUT_CONFIG["output_dir"]
        out_dir.mkdir(exist_ok=True)
        filename = OUTPUT_CONFIG["html_filename_template"].format(timestamp=timestamp)
        path = out_dir / filename
        
        path.write_text(html_code, encoding="utf-8")
        print(f"🎉 SUCCESS! Report saved to: {path}")
        return str(path)

# -------------------------------------
# MAIN
# -------------------------------------

def main():
    topic = "MarTech trends in the Nordics"
    gen = Generator()
    
    # 1. Research Loop
    gen.run_research_cycle(topic, period="This Month")
    
    if not gen.validated_results:
        print("❌ No valid sources found.")
        return

    # 2. Write Report
    report_content = gen.call_writer(topic, "This Month")
    
    # 3. Save
    digest = MonthlyDigest(
        title=f"Monthly Digest: {topic}",
        period="This Month",
        generated_at=datetime.datetime.now().isoformat(),
        content=report_content,
        sources=gen.validated_results,
        metadata={"count": len(gen.validated_results)}
    )
    file_path = gen.save_report(digest)
    if input(f"Send {file_path}? (y/n): ").lower() == 'y':
                send_html_file(file_path, digest.title)

if __name__ == "__main__":
    main()
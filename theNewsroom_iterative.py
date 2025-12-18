import datetime
from ddgs import DDGS
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import json
from typing import List, Dict, Any, Optional
import re
from dataclasses import dataclass, asdict
from urllib.parse import urlparse
from pathlib import Path
import os
from url_normalize import url_normalize
from SendNewsletter import send_html_file
import time

# --- CONFIGURATION ---
GEMINI_MODEL = "gemini-3-pro-preview"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

SEARCH_CONFIG = {
    "min_sources": 10,            # Stop searching if we have this many VALIDATED sources
    "max_results_per_search": 25,
    "target_word_count": 800,
}

OUTPUT_CONFIG = {
    "output_dir": "Output",
    "html_filename_template": "monthly_digest_{timestamp}.html",
    "txt_filename_template": "monthly_digest_{timestamp}.txt"
}

# --- PROMPTS ---

# RESEARCHER: Focused purely on finding raw data
RESEARCHER_PROMPT = """
You are an expert Market Research Analyst.
Your goal is to use the search tool to find information on the user's topic.

GUIDELINES:
1. GEOGRAPHY: Prioritize Sweden/Nordics.
2. TIMELINESS: Focus on the last 30 days.
3. STRATEGY: Look for broad business shifts, not just press releases.

COMMANDS:
- To search: `SEARCH: [your search query]`
- If you have enough information or hit the search limit, reply: `DONE`

Do not summarize findings yet. Just find the data.
"""

# EDITOR: Acts as a strict filter using JSON output
EDITOR_PROMPT = """
You are the Curation Lead at Differ Strategy. 
Your job is to look at raw search snippets and identify which ones represent "Signal" for a management consultant.

YOUR EVALUATION CRITERIA:
1. THE HARD LIMIT (TIMELINESS): If a result is clearly older than 30 days, it must be REJECTED [ID excluded].
2. INTEREST FACTOR: Is this "interesting news"? (e.g., a company launching a new strategy, a major partnership, a market shift, or a tech breakthrough). 
3. RELEVANCE: Does it relate to the user's topic? Be inclusive—if it's a "maybe," keep it.

OUTPUT FORMAT:
You must return a JSON object with a list of 'valid_ids' and a brief 'feedback' string for the researcher.
Example: 
{
  "valid_ids": [0, 2],
  "feedback": "Kept the news about the Volvo merger and the MarTech shift. ID 1 was a 2023 article, so I dropped it."
}
"""

# WRITER: Synthesizes the validated list
WRITER_PROMPT = """
You are a Senior Strategy Consultant.
You will receive a list of VALIDATED FACTS (Title, URL, Snippet).

TASK:
Write a "Monthly Business Digest" based ONLY on these facts.

STRUCTURE:
1. HEADER: Professional Title.
2. EXECUTIVE SUMMARY: 3-4 sentences.
3. KEY DEVELOPMENTS: Group facts into themes.
   - CRITICAL: You MUST use the exact URLs provided. Hyperlink them like this: <a href="URL">Source</a>.
4. CONSULTANT'S TAKE: Strategic conclusion.

TONE: Professional, pragmatic, growth-oriented.
"""

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

class PerfTimer:
    def __init__(self, name):
        self.name = name
        self.start_time = None
    def __enter__(self):
        self.start_time = time.perf_counter()
        print(f"⏱️ [START] {self.name}...")
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.start_time
        print(f"🏁 [DONE]  {self.name} finished in {elapsed:.2f} seconds.")

class Generator:
    def __init__(self):
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        genai.configure(api_key=GOOGLE_API_KEY)
        
        self.researcher_log: List[Dict[str, str]] = []
        self.validated_results: List[SearchResult] = [] # Stores only approved results
        self.seen_urls: set[str] = set()
        self.performed_searches: List[str] = []
        self.allowed_domains: set[str] = self.load_domains()

    def get_agent(self, role: str, json_mode=False) -> genai.GenerativeModel:
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        generation_config = {"response_mime_type": "application/json"} if json_mode else {}
        
        return genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=role,
            safety_settings=safety_settings,
            generation_config=generation_config
        )

    def load_domains(self) -> set[str]:
        script_dir = Path(__file__).parent
        file_path = script_dir / "domainFilter.json"
        domains = set()
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for category, items in data.items():
                if isinstance(items, list):
                    for obj in items:
                        d = (obj.get("domain") or "").strip().lower()
                        if d: domains.add(d)
        except FileNotFoundError:
            pass
        return domains

    def perform_search(self, query: str) -> List[SearchResult]:
        with PerfTimer(f"DDGS Search: '{query}'"):
            try:
                # Adding 'date' parameter to DDGS to help with recency filter at source
                with DDGS() as ddgs:
                    results = []
                    # timelimit='m' gets results from past month
                    ddgs_gen = ddgs.text(query, region='wt-wt', safesearch='off', timelimit='m', max_results=SEARCH_CONFIG["max_results_per_search"])
                    for r in ddgs_gen:
                        if r.get("href"):
                            results.append(SearchResult(
                                title=r.get("title", ""),
                                url=r.get("href", ""),
                                snippet=r.get("body", ""),
                                source="DDGS",
                                query=query
                            ))
                    print(f"🔬 [RESEARCHER] Found {len(results)} sources.")     
                    return results
            except Exception as e:
                print(f"❌ Search error: {e}")
                return []

    def validate_domain(self, url: str) -> bool:
        if not self.allowed_domains: return True
        try:
            host = urlparse(url).netloc.lower()
            if host.startswith("www."): host = host[4:]
            for d in self.allowed_domains:
                if host == d or host.endswith("." + d): return True
            return False
        except: return True

    # --- THE NEW CORE LOGIC: RESEARCH + VALIDATE LOOP ---
    def run_research_cycle(self, topic: str, period: str):
        print(f"\n🚀 --- STARTING ITERATIVE RESEARCH & VALIDATION ---")
        
        researcher_agent = self.get_agent(RESEARCHER_PROMPT)
        # Note: Using json_mode ensures we get a clean dictionary back
        editor_agent = self.get_agent(EDITOR_PROMPT, json_mode=True) 
        
        initial_prompt = f"Topic: {topic}. Period: {period}. Start your first search."
        self.researcher_log.append({"role": "user", "content": initial_prompt})
        
        response = researcher_agent.generate_content(self.gemini_crunch(self.researcher_log))
        self.researcher_log.append({"role": "model", "content": response.text})

        while True:

            queries = re.findall(r'(?i)SEARCH:\s*([^\n]+)', response.text)
            if not queries:
                print("❌ [R] Agent did not request search, forcing new request.")
                self.researcher_log.append({"role": "user", "content": "Please provide a new SEARCH query now. Format: SEARCH: [query]"})
                response = researcher_agent.generate_content(self.gemini_crunch(self.researcher_log))
                self.researcher_log.append({"role": "model", "content": response.text})
                continue

            current_query = queries[0].strip()
            self.performed_searches.append(current_query)
            
            # Get raw results
            raw_batch = self.perform_search(current_query)
            
            # Initial Python-side filtering (Dedup and Domain)
            clean_batch = []
            for r in raw_batch:
                norm_url = url_normalize(r.url)
                if norm_url not in self.seen_urls and self.validate_domain(norm_url):
                    self.seen_urls.add(norm_url)
                    r.url = norm_url
                    clean_batch.append(r)
            
            if clean_batch:
                # Prepare the payload for the Editor
                editor_input = f"Topic: {topic}\nResults to evaluate:\n"
                for idx, item in enumerate(clean_batch):
                    editor_input += f"ID {idx}: {item.title} | {item.snippet[:400]}\n"

                print(f"📑 [EDITOR] Reviewing {len(clean_batch)} items for interest and recency...")
                
                try:
                    # Get JSON response from Editor
                    edit_res = editor_agent.generate_content(editor_input)
                    res_data = json.loads(edit_res.text)
                    
                    valid_ids = res_data.get("valid_ids", [])
                    editor_feedback = res_data.get("feedback", "No specific feedback.")

                    new_validated = []
                    for v_id in valid_ids:
                        if 0 <= v_id < len(clean_batch):
                            new_validated.append(clean_batch[v_id])
                    
                    self.validated_results.extend(new_validated)

                    if len(self.validated_results) >= SEARCH_CONFIG["min_sources"]:
                        break
                    
                    # We feed the Editor's logic back to the Researcher!
                    feedback_to_researcher = f"Feedback: {editor_feedback}. Total valid results now: {len(self.validated_results)}."
                    print(f"📑 [EDITOR] {feedback_to_researcher}")
                    
                except Exception as e:
                    feedback_to_researcher = "Editor busy or error parsing. Move to next search."
                    print(feedback_to_researcher)
            else:
                feedback_to_researcher = "No new unique/allowed domains found for this query."
                print(feedback_to_researcher)

            # Continue the loop
            self.researcher_log.append({"role": "user", "content": feedback_to_researcher})
            response = researcher_agent.generate_content(self.gemini_crunch(self.researcher_log))
            self.researcher_log.append({"role": "model", "content": response.text})

    def call_writer(self, topic: str, period: str) -> str:
        """Takes the CLEAN list and generates the report."""
        print(f"\n🚀 --- STARTING WRITER ---")
        writer_agent = self.get_agent(WRITER_PROMPT)
        
        # Format the validated results into a clean string for the writer
        facts_text = ""
        for i, r in enumerate(self.validated_results, 1):
            facts_text += f"{i}. Title: {r.title}\n   URL: {r.url}\n   Snippet: {r.snippet}\n\n"
            
        task = f"Topic: {topic}\nPeriod: {period}\n\nVALIDATED FACTS:\n{facts_text}"
        
        with PerfTimer("Gemini Writer"):
            response = writer_agent.generate_content(task)
            
        return response.text

    def generate_report(self, topic: str):
        current_date = datetime.date.today()
        period = f"{current_date.strftime('%B %Y')}"

        # 1. Loop: Researcher finds <-> Editor validates
        self.run_research_cycle(topic, period)
        
        if not self.validated_results:
            print("❌ No valid results found after all searches.")
            return None

        # 2. Writer compiles
        digest_content = self.call_writer(topic, period)
        
        return MonthlyDigest(
            title=f"Report on {topic}",
            period=period,
            generated_at=datetime.datetime.now().strftime("%Y-%m-%d"),
            content=digest_content,
            sources=self.validated_results, # We return the Python objects, not text
            metadata={"source_count": len(self.validated_results)}
        )

    def gemini_crunch(self, messages: List[Dict]) -> List[Dict]:
        """Converts internal list to Gemini history format."""
        history = []
        for msg in messages:
            role = "model" if msg["role"] == "assistant" or msg["role"] == "model" else "user"
            history.append({"role": role, "parts": [msg["content"]]})
        return history

    def save_report(self, report: MonthlyDigest) -> str:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(os.path.dirname(__file__), OUTPUT_CONFIG["output_dir"])
        os.makedirs(output_dir, exist_ok=True)

        html_filename = OUTPUT_CONFIG["html_filename_template"].format(timestamp=timestamp)
        html_path = os.path.join(output_dir, html_filename)
        
        # Simple HTML wrapper
        html_content = f"""
        <html>
        <head><title>{report.title}</title>
        <style>body{{font-family:sans-serif;max-width:800px;margin:auto;padding:20px;}} a{{color:#0056b3;}}</style>
        </head>
        <body>
        {report.content.replace(chr(10), '<br>')}
        <hr>
        <h3>Validated Sources Database</h3>
        <ul>
        {''.join([f'<li><a href="{s.url}">{s.title}</a></li>' for s in report.sources])}
        </ul>
        </body></html>
        """
        
        # Regex to make sure plain URLs in text become clickable
        url_pattern = re.compile(r'(?<!href=")(https?://[^\s<]+)')
        html_content = url_pattern.sub(r'<a href="\1">\1</a>', html_content)

        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        print(f"✅ Report saved to {html_path}")
        return html_path

def main():
    topic = "MarTech trends in Nordics"
    try:
        gen = Generator()
        report = gen.generate_report(topic)
        if report:
            file_path = gen.save_report(report)
            if input(f"Send {file_path}? (y/n): ").lower() == 'y':
                send_html_file(file_path, report.title)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
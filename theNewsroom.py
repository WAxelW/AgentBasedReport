import datetime
from ddgs import DDGS
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import json
from typing import List, Dict, Any
import re
from dataclasses import dataclass
from urllib.parse import urlparse
from pathlib import Path
import os
from url_normalize import url_normalize
from SendNewsletter import send_html_file
import time

# Model config
GEMINI_MODEL = "gemini-3-pro-preview"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Search params
SEARCH_CONFIG = {
    "min_sources": 10,
    "target_word_count": 800,
    "search_depth": 10,
    "max_results_per_search": 25,
}

# Output details
OUTPUT_CONFIG = {
    "output_dir": "Output",
    "html_filename_template": "monthly_digest_{timestamp}.html",
    "txt_filename_template": "monthly_digest_{timestamp}.txt"
}

# Config research assistant
RESEARCHER_PROMPT = """
You are an expert Market Research Analyst working for Differ Strategy in Stockholm.
Your role is to scour the web for high-impact business intelligence to support senior management consultants.

CORE OBJECTIVE:
Iteratively search for and gather raw information on the user's specified topic. You are the "eyes and ears" of the consultancy.

SEARCH GUIDELINES:
1. GEOGRAPHY FIRST: Prioritize sources from Sweden (SE), Norway (NO), Denmark (DK), and Finland (FI). Use Swedish search terms if necessary to dig deeper.
2. SOURCE QUALITY: Look for reputable business press (e.g., Dagens Industri, Affärsvärlden, Breakit, Resumé), industry reports, and official press releases from major Nordic companies.
3. CONTENT FOCUS: Look for strategic shifts, mergers & acquisitions, digital transformation, customer behavior changes, and sustainability initiatives.
4. TIMELINESS: Focus strictly on developments from the last 30 days.

YOUR BEHAVIOR:
- Use the tool `SEARCH: [query]` to request searches.
- Analyze the results. If a lead looks promising but lacks detail, perform a follow-up search.
- Do not summarize yet; your job is to capture the raw data/snippets.
"""

# Config quality assurance editor
EDITOR_PROMPT = EDITOR_SYSTEM_PROMPT = """
You are the Chief Editor and Quality Assurance Lead at Differ Strategy.
Your goal is to filter a large set of raw search data into a "Curated Fact Sheet" that is actionable for senior management consultants.

You have a low tolerance for noise, PR fluff, and irrelevant information.

YOUR CRITERIA FOR VALIDATION:
1. STRATEGIC RELEVANCE: Does this news impact the strategic landscape? (e.g., a new logo is noise; a new business model is signal).
2. NORDIC RELEVANCE: Is this relevant to a consultant based in Stockholm? If it is a global trend, does it have a clear local angle?
3. RECENCY: Discard ANY information older than 30 days.
4. CREDIBILITY: Discard rumor mills or low-quality blogs.

Your job is NOT to write or edit any text, simply to filter out the irrelevant results

OUTPUT FORMAT:
Provide a structured Markdown list of the vetted findings, including all the text and the URL source etc.

If the raw data contains nothing of value, state that clearly.
"""

# Config the professional copy writer
WRITER_PROMPT = WRITER_SYSTEM_PROMPT = """
You are a Senior Strategy Consultant and Communications Expert at Differ.
Your task is to synthesize a list of validated facts into a professional "Monthly Business Digest".

TONE OF VOICE & STYLE (The "Differ Way"):
- Professional yet approachable. Avoid overly academic jargon, but maintain high business acumen.
- Growth-Oriented: Frame news in terms of opportunities, sustainable growth, and customer value.
- Pragmatic: Focus on tangible impact ("What does this mean?") rather than just reporting the news.
- Structure: Clear, scannable, and efficient. Your readers are busy consultants in Stockholm.

REPORT STRUCTURE:
1. HEADER: A catchy, professional title, reflecting this months content.
2. EXECUTIVE SUMMARY: A 3-4 sentence synthesis of the month's overall sentiment. What is the big picture?
3. KEY DEVELOPMENTS (The Core):
   - Group facts into logical themes (e.g., "Digital Transformation", "Sustainability", "Market Moves").
   - Use bullet points.
   - For every claim, you MUST hyperlink the source using <a href="URL">Source</a>.
4. CONSULTANT'S TAKE: A short concluding section connecting the insights to Differ's core areas (Strategy, Insight, or Creative).

CONSTRAINTS:.
- Do not hallucinate. Use ONLY the provided Curated Facts.
- Language: English (Standard Business English).
"""

@dataclass
class SearchResult:
    """Structured search result data"""
    title: str
    url: str
    snippet: str
    source: str
    query: str

@dataclass
class MonthlyDigest:
    """Structured monthly digest data"""
    title: str
    period: str
    generated_at: str
    search_count: int
    total_results: int
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
        self.search_results: List[str] = []
        self.messages: List[Dict[str, str]] = []
        self.performed_searches: List[str] = []
        self.seen_urls: set[str] = set()
        self.allowed_domains: set[str] = set()
        self.load_domains()

    def get_agent(self, role: str) -> genai.GenerativeModel:
        """Helper to create a fresh model instance with a specific persona"""
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        return genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=role,
            safety_settings=safety_settings
        )

    def load_domains(self) -> None:

        script_dir = Path(__file__).parent
        file_path = script_dir / "domainFilter.json"

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            domains: set[str] = set()
            for category, items in data.items():
                if isinstance(items, list):
                    for obj in items:
                        d = (obj.get("domain") or "").strip().lower()
                        if d:
                            domains.add(d)
            self.allowed_domains = domains
            print(f"✅ [DEBUG] Loaded {len(self.allowed_domains)} allowed domains for filtering")
        except FileNotFoundError:
            print(f"⚠️ [DEBUG] domainFilter.json not found, proceeding without domain filter.")

    def extract_search_queries(self, agent_response: str) -> List[str]:
        queries = [q.strip() for q in re.findall(r'(?i)SEARCH:\s*([^\n]+)', agent_response)]
        print(f"🔍 [DEBUG] Extracted {len(queries)} search queries: {queries}")
        unique_queries = []
        for q in queries:
            if q not in self.performed_searches:
                self.performed_searches.append(q)
                unique_queries.append(q)
        return unique_queries

    def perform_search(self, query: str) -> List[SearchResult]:
        with PerfTimer(f"DDGS Search: '{query}"):
            try:
                with DDGS() as ddgs:
                    all_results = []
                    
                    for result in ddgs.text(query, max_results=SEARCH_CONFIG["max_results_per_search"]):
                        search_result = SearchResult(
                            title=result.get("title", ""),
                            url=result.get("href", ""),
                            snippet=result.get("body", ""),
                            source="DDGS",
                            query=query,
                        )
                        all_results.append(search_result)

                    print(f"✅ [DEBUG] Search completed: {len(all_results)} results")
                    return all_results
                
            except Exception as e:
                print(f"❌ [DEBUG] Search error: {e}")
                return []
        
    def validate_domain(self, url: str) -> bool:
        if not self.allowed_domains:
            return True
        try:
            host = urlparse(url).netloc.lower()
            if host.startswith("www."):
                host = host[4:]
            for d in self.allowed_domains:
                if host == d or host.endswith("." + d):
                    return True
            return False
        except Exception:
            return True
    
    # --- STEP 1: THE RESEARCHER ---
    def call_researcher(self, topic: str, period: str) -> str:
        """Orchestrates the iterative search process using the Researcher Agent."""
        print(f"\n🚀 --- STARTING STEP 1: RESEARCHER ---")
        researcher_agent = self.get_agent(RESEARCHER_PROMPT)
                
        initial_prompt = f"""
            Start your research. Topic: {topic}. Period: {period}.
            You must commit {SEARCH_CONFIG['search_depth']} searches.
            Begin by requesting your first search query using the format: SEARCH: [query].
            """
        
        self.researcher_log.append({"role": "user", "content": initial_prompt})
        
        # Konvertera och skicka historiken
        history = self.gemini_crunch(self.researcher_log)
        with PerfTimer("Gemini: initial query request"):
            response = researcher_agent.generate_content(history)
        self.researcher_log.append({"role": "model", "content": response.text})

        search_count = 0
        start_time = time.time()
        
        while search_count < SEARCH_CONFIG['search_depth']:
            print(f"🤖 [DEBUG] Starting iteration {search_count + 1}/{SEARCH_CONFIG['search_depth']}")
            print(self.researcher_log)

            # 1. Hämta nästa sökfråga
            queries = self.extract_search_queries(response.text)
            if not queries:
                 print("⚠️ [R] Agent did not request search, forcing new request.")
                 self.researcher_log.append({"role": "user", "content": "Please provide a new SEARCH query now. Format: SEARCH: [query]"})
                 response = researcher_agent.generate_content(self.gemini_crunch(self.researcher_log))
                 self.researcher_log.append({"role": "model", "content": response.text})
                 continue
            
            # 2. Utför sökningar
            for q in queries:
                if search_count >= SEARCH_CONFIG['search_depth']:
                    break
                    
                search_results = self.perform_search(q)
                
                deduped: List[SearchResult] = []
                
                for r in search_results:
                    # 1. Normalisera URL:en för att ta bort spårningsparametrar
                    try:
                        normalized_url = url_normalize(r.url)
                    except Exception as e:
                        # Hoppa över om URL-normaliseringen misslyckas helt
                        print(f"⚠️ URL normalization failed for {r.url}: {e}")
                        continue
                        
                    # 2. Kontrollera om URL:en redan setts ELLER om domänen är ogiltig
                    if normalized_url in self.seen_urls or not self.validate_domain(normalized_url):
                        continue
                        
                    # 3. Om godkänd: Lägg till i listan, uppdatera set och resultat
                    self.seen_urls.add(normalized_url)
                    r.url = normalized_url  # Uppdatera SearchResult-objektet med den rena URL:en
                    deduped.append(r)

                if deduped:
                    search_count += 1
                    self.search_results.extend(deduped)
                    results_text = "\n".join([f"- {r.title} ({r.url}): {r.snippet}" for r in deduped])
                    self.researcher_log.append({"role": "user", "content": f"SEARCH RESULTS for query '{q}':\n{results_text}"})
                    print(f"✅ [R] Search {search_count} completed with {len(deduped)} valid results.")
                    print(f"   📝 Adding {len(results_text)} chars of context to Agent's memory.")
                else:
                    self.researcher_log.append({"role": "user", "content": f"SEARCH RESULTS for query '{q}': No valid results found for this query, try another."})
                    print(f"❌ [R] Search for '{q}' returned no valid results after filtering.")

            # 3. Be Agenten att fortsätta
            if search_count < SEARCH_CONFIG['search_depth']:
                self.researcher_log.append({"role": "user", "content": f"Provide the next SEARCH query. ({search_count}/{SEARCH_CONFIG['search_depth']} completed)"})

                content = None

                with PerfTimer("Gemini: next query request"):
                    response = researcher_agent.generate_content(self.gemini_crunch(self.researcher_log))
                try:
                    # Försök att hämta texten som vanligt
                    content = response.text 
                except ValueError:
                    # Fånga felet och inspektera orsaken
                    if response.candidates and response.candidates[0].finish_reason == 1:
                        
                        # Hämta den faktiska säkerhetsbedömningen
                        ratings = response.candidates[0].safety_ratings
                        
                        print("\n--- SAFETY BLOCK DETECTED ---")
                        print("Model generated content but it was filtered.")
                        for rating in ratings:
                            if rating.blocked:
                                print(f"BLOCKED: {rating.category.name} (Threshold: {rating.threshold.name})")
                        print("-----------------------------\n")
                    else:
                        raise

                self.researcher_log.append({"role": "model", "content": content})
        
        total_duration = time.time() - start_time
        print(f"🛑 [R] Max search depth ({SEARCH_CONFIG['search_depth']}) reached after {total_duration: .2f}s.")
        
        # Returnera all rådata för nästa steg
        raw_data_text = "\n\n---\n\n".join([
            f"Source: {r.source}\nTitle: {r.title}\nURL: {r.url}\nSnippet: {r.snippet}" 
            for r in self.search_results
        ])
        return raw_data_text
    
    # --- STEP 2: THE EDITOR ---
    def call_editor(self, raw_data_text: str, topic: str, period: str) -> str:
        """Uses the Editor Agent to validate and filter raw data."""
        print(f"\n🚀 --- STARTING STEP 2: EDITOR ---")
        data_size = len(raw_data_text)
        print(f"   📦 Input Payload: {data_size} characters (~{data_size // 4} tokens)")
        
        editor_agent = self.get_agent(EDITOR_PROMPT)
        
        task = f"""
        Analyze the following {len(self.search_results)} search results for the 'Monthly Digest' on topic '{topic}' for the period of {period}. 
        
        Your only task is to check each source and the information fetched from that source to see if it matches the topic and most importantly, the period
        
        INPUT RAW DATA:
        {raw_data_text}
        
        OUTPUT: a list of each source and the content fetched from that source
        """
        with PerfTimer("Gemini: validating and filtering"):
            response = editor_agent.generate_content(task)
        print(f"✅ [E] Validation complete. Output length: {len(response.text)} chars.")
        return response.text # Curated Fact Sheet
    
# --- STEP 3: THE WRITER ---
    def call_writer(self, curated_facts: str, topic: str, period: str) -> str:
        """Uses the Writer Agent to synthesize the final report."""
        print(f"\n🚀 --- STARTING STEP 3: WRITER ---")
        
        # Se till att get_agent sätter BLOCK_NONE (se punkt 2 nedan)
        writer_agent = self.get_agent(WRITER_PROMPT)
        
        writer_task = f"""
        You are writing the final 'Monthly Digest' for {period} on the topic '{topic}'.
        
        CRITICAL: Use ONLY the following 'Curated Facts' to write the report. Do not add external knowledge or new searches.
        
        Curated Facts:
        {curated_facts}
        
        Output the final report using the required structure:
        1. MONTHLY_DIGEST: [start of report]
        2. Executive Summary
        3. Main body (bullet points with analysis and references)
        4. Source References (numbered list of URLs used from the Curated Facts)
        
        Ensure the output length is near {SEARCH_CONFIG['target_word_count']} words.
        """
        
        content = ""
        
        # Anropa modellen
        with PerfTimer("Gemini: writing final report"):
            response = writer_agent.generate_content(writer_task)
            
        # SÄKERHETSKONTROLL & FELHANTERING
        try:
            content = response.text
        except ValueError:
            # Hantera säkerhetsblockering
            if response.candidates and response.candidates[0].finish_reason == 1:
                print("\n❌ [CRITICAL] Writer output blocked by safety filters!")
                print("   Inspect the following ratings to see why:")
                
                # Skriv ut exakt vad som triggade filtret
                for rating in response.candidates[0].safety_ratings:
                    if rating.blocked:
                        print(f"   🚫 BLOCKED: {rating.category.name} (Threshold: {rating.threshold.name})")
                
                # Returnera en "Nöd-rapport" så vi inte förlorar data
                return f"""
                <h1>Generation Failed Due to Safety Filters</h1>
                <p>The AI model refused to generate the report for '{topic}' due to safety constraints.</p>
                <p><strong>Curated Facts intended for this report:</strong></p>
                <pre>{curated_facts}</pre>
                """
            else:
                # Om det är något annat fel, kasta det vidare
                raise

        # Extrahera innehållet efter MONTHLY_DIGEST:
        match = re.search(r'MONTHLY_DIGEST:\s*(.*)', content, re.DOTALL | re.IGNORECASE)
        final_digest_content = match.group(1).strip() if match else content
        
        print(f"✅ [W] Final digest compiled. Length: {len(final_digest_content)} chars.")
        return final_digest_content
    
    # --- ORCHESTRATION ---
    def generate_report(self, topic: str):
        current_date = datetime.date.today()
        period = f"{current_date.strftime('%B %Y')}"

        # 1. Researcher: Gather data and store in self.search_results
        raw_data = self.call_researcher(topic, period)
        
        # 2. Editor: Validate and filter
        curated_facts = self.call_editor(raw_data, topic, period)
        
        # 3. Writer: Compile the final report
        digest_content = self.call_writer(curated_facts, topic, period)
        word_count = len(digest_content.split())
        
        # Skapa MonthlyDigest objektet
        return MonthlyDigest(
            title=f"Report on {topic} for {period}",
            period=period,
            generated_at=datetime.datetime.now().date().strftime("%d %b %Y"),
            search_count=SEARCH_CONFIG['search_depth'],
            total_results=len(self.search_results),
            content=digest_content,
            sources=self.search_results,
            metadata={
                "AI model": GEMINI_MODEL,
                "architecture": "Newsroom (R->E->W)",
                "search_config": SEARCH_CONFIG,
                "word_count": word_count,
            }
        )

    def gemini_crunch(self, messages: List[Dict]) -> List[Dict]:
        """Converts internal OpenAI-style list to Gemini content history."""
        gemini_history = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            # System roles hanteras av get_agent. Vi konverterar assistent till model.
            if role == "assistant":
                role = "model"
            elif role == "system":
                # System messages inside the conversation history are treated as user input for the Researcher
                role = "user" 
            
            gemini_history.append({"role": role, "parts": [content]})
        return gemini_history
    
    def save_report(self, report: MonthlyDigest) -> str:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, OUTPUT_CONFIG["output_dir"])
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save HTML
        html_filename = OUTPUT_CONFIG["html_filename_template"].format(timestamp=timestamp)
        html_path = os.path.join(output_dir, html_filename)
        
        print("Running linkify...")
        html_content = self.create_html(report)
        html_content = self.linkify_sources(html_content)
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        # Save TXT
        txt_filename = OUTPUT_CONFIG["txt_filename_template"].format(timestamp=timestamp)
        txt_path = os.path.join(output_dir, txt_filename)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(report.title + "\n")
            f.write("=" * 60 + "\n\n")
            f.write(report.content)

        print(f"✅ [DEBUG] Report saved to {html_path}")
        return html_path
    
    def create_html(self, report: MonthlyDigest) -> str:
        # (HTML creation logic same as before, simplified for brevity in this snippet)
        # Using f-string for metadata display
        metadata_html = f"""
        Model used: {report.metadata.get('AI model')} | 
        Generated: {report.generated_at} | 
        Sources: {report.total_results}
        """
        
        html_text = f"""<!DOCTYPE html>
            <html lang="en">
                <head>
                    <meta charset="UTF-8" />
                    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
                    <title>{report.title}</title>
                    <style>
                        body {{ font-family: sans-serif; max-width: 900px; margin: 40px auto; padding: 20px; line-height: 1.6; color: #333; }}
                        h1 {{ color: #003732; }}
                        a {{ color: #0064AA; }}
                        .meta {{ font-size: 0.9em; color: #666; margin-bottom: 30px; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
                    </style>
                </head>
                <body>
                    <h1>{report.title}</h1>
                    <div class="meta">{metadata_html}</div>
                    <div>{report.content.replace(chr(10), '<br>')}</div>
                </body>
            </html>"""
        return html_text
    
    def linkify_sources(self, html_text: str):
        url_regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
        all_urls = re.findall(url_regex, html_text)
        # Simple naive replacement (careful not to double replace in real prod)
        for url in all_urls:
             # Check if already inside an href to avoid breaking existing tags
            if f'"{url[0]}"' not in html_text:
                html_text = html_text.replace(url[0], f'<a href="{url[0]}" target="_blank">{url[0]}</a>')
        return html_text

def main():
    topic = "MarTech trends"
    try:
        print("Script initiated...")
        generator = Generator()
        report = generator.generate_report(topic)

        if report:
            print(f"✅ [DEBUG] {report.title} generated successfully.")
            file = generator.save_report(report)
            
            # Send report
            confirmation = input(f"Do you want to send the HTML report '{file}'? (Y/N): ").strip().lower()
            if confirmation == 'y':
                send_html_file(file, f"Differ Monthly Digest - {report.period}")
            else:
                print("Report not sent.")
        else:
            print(f"❌ [DEBUG] Report generation failed.")
            
    except Exception as e:
        print(f"❌ [CRITICAL ERROR] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
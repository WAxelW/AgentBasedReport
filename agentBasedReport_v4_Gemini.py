import datetime
from ddgs import DDGS
import google.generativeai as genai  # Ändrat från OpenAI
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import json
from typing import List, Dict, Any
import re
from dataclasses import dataclass
from urllib.parse import urlparse
from url_normalize import url_normalize
import os
from SendNewsletter import send_html_file
import time

# Konfiguration för Gemini
GEMINI_MODEL = "gemini-3-pro-preview" # Eller "gemini-1.5-pro" för mer komplex analys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

SEARCH_CONFIG = {
    "min_sources": 10,
    "target_word_count": 800,
    "search_depth": 10,
    "max_results_per_search": 25,
}

OUTPUT_CONFIG = {
    "output_dir": "Output",
    "html_filename_template": "monthly_digest_{timestamp}.html",
    "txt_filename_template": "monthly_digest_{timestamp}.txt"
}

DIFFER_AGENT_SYSTEM_PROMPT = """
       You are a senior management consultant at Differ Strategy.
       You specialize in helping companies achieve sustainable growth by combining deep customer insight, innovative strategy, and creative execution.
       You are pragmatic, structured, and results-oriented.
       You always aim to connect analysis with tangible impact.
       Your communication style is professional yet clear and engaging.

       Refer to the following information as your knowledge base and style guide:

        Differ’s Core Identity
        - Company type: Management consulting firm based in Stockholm.
        - Focus: Sustainable growth for clients by combining insights, strategy, and creativity.
        - Approach: Blend of analytical rigor, customer insight, and creative problem solving.
        - Tone of voice: Professional but approachable, emphasizing growth, customer-centricity, and measurable impact.

        Value Propositions
        - Customer Experience & Insight
            - Deep understanding of customer needs.
            - Turn insights into growth opportunities.
            - Map customer journeys and improve touchpoints.
        - Brand & Communication
            - Help brands differentiate and strengthen their position.
            - Build engagement and emotional connection with customers.
            - Use storytelling and creative concepts to make strategies actionable.
        - Innovation & Growth
            - Identify and develop new business opportunities.
            - Design and launch new products, services, or models.
            - Balance short-term results with long-term growth.
            
        Methodology & Ways of Working
        - Combine analysis, creativity, and customer focus.
        - Bridge insight → strategy → activation.
        - Co-create with clients and their customers.
        - Pragmatic, hands-on consultants who deliver real business value, not just slide decks.
        
        Example Cases (Condensed)
        - Swedavia: Developed insights and strategies for future airport customer experiences.
        - Scania: Defined brand positioning to support transformation in transport solutions.
        - Vattenfall: Helped shape offerings in sustainability and customer engagement.
        - Telia: Supported development of customer-centric innovations.
        - Other sectors: Retail, finance, tech, energy — always with a growth/customer focus.

        Differ’s Insights & Thought Leadership
        - Articles on sustainable growth, customer engagement, and brand strategy.
        - Emphasis on future-proofing businesses.
        - Position Differ as both thinkers (strategic) and doers (activation).
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

class Generator:

    def __init__(self):
        # Konfigurera Gemini
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        genai.configure(api_key=GOOGLE_API_KEY)
        
        # Initiera modell med System Prompt direkt i konstruktorn
        self.model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=DIFFER_AGENT_SYSTEM_PROMPT
        )
        
        self.search_results: List[str] = []
        self.messages: List[Dict[str, str]] = [] # Vi behåller denna struktur för logikens skull
        self.performed_searches: List[str] = []
        self.seen_urls: set[str] = set()
        self.allowed_domains: set[str] = set()
        self.load_domains()
        self.initialize_agent()

    def load_domains(self) -> None:
        try:
            with open("domainFilter.json", "r", encoding="utf-8") as f:
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

    def initialize_agent(self):
        print(f"🤖 [DEBUG] Initializing agent with model {GEMINI_MODEL}")
        try:
            # Test call. Notera att vi skickar en enkel user prompt för att testa anslutningen.
            response = self.model.generate_content("Respond with ACKNOWLEDGED if you are ready.")
            print(f"✅ [DEBUG] Agent initialized: {response.text.strip()}")
        except Exception as e:
            print(f"❌ [DEBUG] Error accessing Gemini model {GEMINI_MODEL}: {e}")
            raise

    def generate_report(self, topic: str):
        print(f"🤖 [DEBUG] Starting report generation process")
        print(f"🤖 [DEBUG] Generating report on topic: {topic}")
        current_date = datetime.date.today()
        period = f"{current_date.strftime('%B %Y')}"

        initial_prompt = self.create_initial_prompt(period, topic)

        # Gemini föredrar User-roll för uppgiftsbeskrivningen när System Prompt redan är satt
        self.messages.append({"role": "user", "content": initial_prompt})
        
        response_text = self.call_model(self.messages)
        print(f"🤖 [DEBUG] Agent response to task brief: {response_text}")
        
        # Spara modellens svar i historiken
        self.messages.append({"role": "assistant", "content": response_text})

        search_count = 0
        failed_searches = 0

        while search_count < SEARCH_CONFIG['search_depth']:
            print(f"🤖 [DEBUG] Starting iteration {search_count + 1}/{SEARCH_CONFIG['search_depth']}")
            
            prompt = "Request a search query using the format: SEARCH: [your query]. A good search query contains up to 5 key words."
            self.messages.append({"role": "user", "content": prompt})
            
            agent_response = self.call_model(self.messages)
            self.messages.append({"role": "assistant", "content": agent_response})

            print(f"🔄 [DEBUG] Agent response received: {len(agent_response)} chars")
            print(f"🔄 [DEBUG] Response preview: {agent_response}")

            if "SEARCH:" in agent_response:
                print(f"🔄 [DEBUG] Agent requested search")
                queries = self.extract_search_queries(agent_response)
                if not queries:
                    print(f"⚠️ [DEBUG] No new search queries extracted, skipping search")
                    continue

                for idx, q in enumerate(queries, start=1):
                    if search_count >= SEARCH_CONFIG['search_depth']:
                        break

                    print(f"🔍 [DEBUG] Processing queued search {idx}/{len(queries)}: {q}")
                    search_results = self.perform_search(q)

                    # Remove duplicate results
                    deduped: List[SearchResult] = []
                    for r in search_results:
                        normalized_url = url_normalize(r.url)
                        if not normalized_url or normalized_url in self.seen_urls or not self.validate_domain(normalized_url):
                            continue
                        self.seen_urls.add(normalized_url)
                        r.url = normalized_url
                        deduped.append(r)
                    
                    if deduped:
                        search_count += 1
                        print(f"✅ [DEBUG] Search {search_count} completed with {len(deduped)} valid results")
                        results_text = "\n".join([f"- {r.title} ({r.url}): {r.snippet}" for r in deduped])
                        self.messages.append({"role": "user", "content": f"SEARCH RESULTS for query '{q}':\n{results_text}"})
                        self.search_results.extend(deduped)
                    else:
                        failed_searches += 1
                        print(f"⚠️ [DEBUG] Search for '{q}' returned no valid results")
                        self.messages.append({"role": "user", "content": f"SEARCH RESULTS for query '{q}': No valid results found."})
                
            else:
                print(f"🔄 [DEBUG] Agent response unclear, providing guidance")
                self.messages.append({"role": "user", "content": "I can't parse that response. Please try again using SEARCH: [your query]."})
        
        print(f"🤖 [DEBUG] Max search depth reached. Forcing completion.")
        digest = self.force_completion(self.messages)
        word_count = len(digest.split())

        return MonthlyDigest(
            title=f"Report on {topic} for {period}",
            period=period,
            generated_at=datetime.datetime.now().date().strftime("%d %b %Y"),
            search_count=search_count,
            total_results=len(self.search_results),
            content=digest,
            sources=self.search_results,
            metadata={
                "AI model": GEMINI_MODEL,
                "search engine": "DDGS",
                "search_config": SEARCH_CONFIG,
                "word_count": word_count,
                }
        )
                
    def create_initial_prompt(self, period: str, topic: str) -> str:
        """Create initial prompt for agent"""
        return f"""
            You have been given a task by the CEO of Differ to create a professional monthly business digest for {period}.
            To do this you must research and analyze the most significant business/news/research developments from the last 30 days, focusing on {topic} in mainly Swedish and Nordic markets.

            CRITICAL REQUIREMENTS:
            - Target word count: {SEARCH_CONFIG['target_word_count']} words
            - Minimum {SEARCH_CONFIG['min_sources']} sources required
            - Professional, but digestible tone
            - Include specific source references for all claims

            HOW TO CONDUCT RESEARCH:
            1. Give me queries of the format "SEARCH: [your query]"
            2. I will conduct the search and provide you with search results
            3. Analyze the results and request more searches if needed
            4. After comitting {SEARCH_CONFIG['search_depth']} searches, provide your digest using the format: "MONTHLY_DIGEST: [your report]"
            5. I will format a report based on your output

            ANALYSIS INSTRUCTIONS:
            1. Critically evaluate the sources for credibility, relevance and trustworthiness
            2. If the source does not meet high standards, disregard it
            3. Extract key developments, dates, companies, and business impacts
            4. Note the source URL for each claim you make and refer to it as the source

            DIGEST FORMAT:
            Your final output should contain the following sections:
            - Executive Summary: a brief overview of the key developments the past month
            - Main body: list the most interesting developments in bullet points with short summaries and references
            - Source References (numbered list of URLs used - this section is NOT counted toward word count)

            To ensure that you have understood your task, please give me a brief summary of what you need to do.
            """

    def call_model(self, messages: List[Dict], retries=0) -> str:
        """
        Converts OpenAI style messages list to Gemini content format and calls the API.
        """
        gemini_history = []
        
        # Konvertera OpenAI-historik till Gemini-historik
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            # Gemini använder 'user' och 'model'. 'system' hanteras i init.
            # Om vi stöter på 'system' här (utöver initial prompt), behandla det som 'user'.
            if role == "system":
                if content == DIFFER_AGENT_SYSTEM_PROMPT:
                    continue # Hoppa över denna då den sattes i konstruktorn
                role = "user"
            elif role == "assistant":
                role = "model"
            
            gemini_history.append({"role": role, "parts": [content]})

        try:
            # Säkerhetsinställningar för att undvika att affärsnyheter blockeras
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }

            response = self.model.generate_content(
                gemini_history,
                safety_settings=safety_settings
            )
            
            if not response.text:
                raise ValueError("Empty response text")
                
            return response.text.strip()

        except Exception as e:
            # Hantera 'overloaded' eller andra fel
            if retries < 5:
                print(f"⚠️ [DEBUG] Gemini API error (attempt {retries+1}/5): {e}. Retrying...")
                time.sleep(2 * (retries + 1)) # Exponential backoff
                return self.call_model(messages, retries + 1)
            else:
                print(f"❌ [DEBUG] Max retries reached. Error: {e}")
                return "ERROR: Could not generate response."
    
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
        print(f"🔍 [DEBUG] Starting search for: {query}")
        try:
            with DDGS() as ddgs:
                print(f"🔍 [DEBUG] DDGS initialized, starting search...")
                all_results = []
                # DDGS kan vara känsligt för för många anrop, en liten paus kan hjälpa
                time.sleep(1) 
                
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
        
    def force_completion(self, messages: List[Dict]) -> str:
        messages.append({"role": "user", "content": f"""DIGEST FORMAT:
            You must provide your final output now. It should contain the following sections:
            - Executive Summary
            - Main body (bullet points with summaries)
            - Source References (numbered list of URLs)
            
            Your response should be around {SEARCH_CONFIG['target_word_count']} words long.
            DO NOT include any next steps or recommendations, only facts and analysis.
            """})
        
        return self.call_model(messages)

    def save_report(self, report: MonthlyDigest) -> str:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, OUTPUT_CONFIG["output_dir"])
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save HTML
        html_filename = OUTPUT_CONFIG["html_filename_template"].format(timestamp=timestamp)
        html_path = os.path.join(output_dir, html_filename)
        
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
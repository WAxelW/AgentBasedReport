import datetime
from ddgs import DDGS
from openai import OpenAI
import json
from typing import List, Dict, Any
import re
from dataclasses import dataclass
from urllib.parse import urlparse
from url_normalize import url_normalize
import os
from SendNewsletter import send_html_file


OPENAI_MODEL = "gpt-5-mini"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

SEARCH_CONFIG = {
    "min_sources": 10,
    "target_word_count": 800,
    "search_depth": 10,
    "max_results_per_search": 25,
}

OUTPUT_CONFIG = {
    "output_dir": "Output",  # This will be relative to the script location
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

        If you have understood, respond with "ACKNOWLEDGED".
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
    content: str  # The main digest content
    sources: List[SearchResult]
    metadata: Dict[str, Any]

class Generator:

    def __init__(self):
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        self.search_results: List[str] = []
        self.messages: List[Dict[str, str]] = []
        self.performed_searches: List[str] = []
        self.seen_urls: set[str] = set()
        self.allowed_domains: set[str] = set()
        self.load_domains()
        self.initialize_agent()

    def load_domains(self) -> None:
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

    def initialize_agent(self):
        print(f"🤖 [DEBUG] Initializing agent with model {OPENAI_MODEL}")
        # Test call to OpenAI to ensure connectivity
        try:
            self.messages.append({"role": "system", "content": DIFFER_AGENT_SYSTEM_PROMPT})
            response = self.call_model(self.messages, "", 0)  # Warm-up call
            print(f"✅ [DEBUG] Agent initialized: {response.choices[0].message.content.strip()}")
        except Exception as e:
            print(f"❌ [DEBUG] Error accessing OpenAI model {OPENAI_MODEL}: {e}")
            raise

    def generate_report(self, topic: str):
        print(f"🤖 [DEBUG] Starting report generation process")
        print(f"🤖 [DEBUG] Generating report on topic: {topic}")
        current_date = datetime.date.today()
        period = f"{current_date.strftime('%B %Y')}"

        initial_prompt = self.create_initial_prompt(period, topic)

        self.messages.append({"role": "system", "content": initial_prompt})
        response = self.call_model(self.messages, period, 0)
        print(f"🤖 [DEBUG] Agent response to task brief: {response.choices[0].message.content.strip()}")

        search_count = 0
        failed_searches = 0

        while search_count < SEARCH_CONFIG['search_depth']:
            print(f"🤖 [DEBUG] Starting iteration {search_count + 1}/{SEARCH_CONFIG['search_depth']}")
            self.messages.append({"role": "user", "content": "Request a search query using the format: SEARCH: [your query]. A good search query conatins up to 5 key words."})
            response = self.call_model(self.messages, period, search_count)
            agent_response = response.choices[0].message.content.strip()
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
                        print(f"🔍 [DEBUG] Max searches reached; skipping remaining {len(queries) - (idx - 1)} queued quer{'y' if len(queries) - (idx - 1) == 1 else 'ies'}")
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
                    
                    # Only consider valid results
                    if deduped:
                        search_count += 1
                        print(f"✅ [DEBUG] Search {search_count} completed with {len(deduped)} valid results")
                        results_text = "\n".join([f"- {r.title} ({r.url}): {r.snippet}" for r in deduped])
                        print(f"🔍 [DEBUG] Adding results to conversation (length: {len(results_text)} chars)")
                        print(f"🔍 [DEBUG] Results preview: {results_text[:500]}{'...' if len(results_text) > 500 else ''}")
                        self.messages.append({"role": "user", "content": f"SEARCH RESULTS for query '{q}':\n{results_text}"})
                        self.search_results.extend(deduped)
                    else:
                        failed_searches += 1
                        print(f"⚠️ [DEBUG] Search for '{q}' returned no valid results after filtering")
                        self.messages.append({"role": "user", "content": f"SEARCH RESULTS for query '{q}': No valid results found."})
                
                print(f"🔄 [DEBUG] Search request handled. Successful searches: {search_count}. Failed searches: {failed_searches}.")

            else:
                print(f"🔄 [DEBUG] Agent response unclear, providing guidance")
                self.messages.append({"role": "user", "content": "I can't parse that response. Please try again using SEARCH: [your query] to request web searches. I will provide you with search results. When you have enough information, provide your digest using: MONTHLY_DIGEST: [your report]"})
        
        print(f"🤖 [DEBUG] Max search depth reached. Forcing completion.")
        response = self.force_completion(self.messages, period, search_count)
        digest = response.choices[0].message.content.strip()
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
                "OpenAI model": OPENAI_MODEL,
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
            - Minimum {SEARCH_CONFIG['min_sources']} sources required to ensure an interesting report
            - Professional, but digestible tone that is easy to read and understand
            - Include specific source references for all claims (references are separate from word count)

            HOW TO CONDUCT RESEARCH:
            1. Give me queries of the format "SEARCH: [your query]" to request web searches
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

    def call_model(self, messages: List[Dict], period, search_count) -> Any:

        response = self.openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
        )

        retries = 0

        if len(response.choices[0].message.content.strip()) == 0:
            retries += 1
            if retries > 5:
                print(f"🤖 [DEBUG] OpenAI API response is empty, max retries reached.")
                return self.force_completion(messages, period, search_count, retries)
            else:
                print(f"🤖 [DEBUG] OpenAI API response is empty, retrying...")
                messages.append({"role": "user", "content": "Your response contains no content. Please try again, make sure to follow the instructions and provide a response. Retry counter: {retries}/5"})
                return self.call_model(messages, period, search_count, retries)
            
        return response
    
    def extract_search_queries(self, agent_response: str) -> List[str]:
        queries = [q.strip() for q in re.findall(r'(?i)SEARCH:\s*([^\n]+)', agent_response)]
        print(f"🔍 [DEBUG] Extracted {len(queries)} search queries: {queries}")
        for q in queries:
            if q in self.performed_searches:
                queries.remove(q)
            else:
                self.performed_searches.append(q)
        return queries

    def perform_search(self, query: str) -> List[SearchResult]:
        print(f"🔍 [DEBUG] Starting search for: {query}")
        try:
            with DDGS() as ddgs:
                print(f"🔍 [DEBUG] DDGS initialized, starting search...")
                all_results = []
                result_count = 0
                
                for result in ddgs.text(query, max_results=SEARCH_CONFIG["max_results_per_search"]):
                    result_count += 1
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
            import traceback
            traceback.print_exc()
            return []
        
    def validate_domain(self, url: str) -> bool:
        """Check if URL's host matches an allowed domain or its subdomain."""
        if not self.allowed_domains:
            return True  # no filter configured
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
        
    def force_completion(self, messages: List[Dict], period, search_count) -> Any:
        messages.append({"role": "user", "content": f"""DIGEST FORMAT:
            You must provide your final output now. It should contain the following sections:
            - Executive Summary: a brief overview of the key developments the past month
            - Main body: list the most interesting developments in bullet points with short summaries and references. Make sure to pick the sources that are the most relevant to the subject!
            - Source References (numbered list of URLs used - this section is NOT counted toward word count)
            Your response should be around {SEARCH_CONFIG['target_word_count']} words long and include at least (but not limited to) {SEARCH_CONFIG['min_sources']} sources.
            DO NOT include any next steps or recommendations, only facts and analysis.
            You also do not need to provide a title, as I will add that later.
            """})
        response = self.call_model(messages, period, search_count)
        if len(response.choices[0].message.content.strip()) == 0:
            print(f"❌ [DEBUG] Forced completion response is empty, retrying once.")
            messages.pop()  # remove last forced completion prompt
            messages.append({"role": "user", "content": "Your response contains no content. Please try again, make sure to follow the instructions and provide a response."})
            response = self.call_model(messages, period, search_count)
        else:
            print(f"✅ [DEBUG] Forced completion response received ({len(response.choices[0].message.content.strip())} chars).")
            return response

    def save_report(self, report: MonthlyDigest) -> None:
        """Save digest in both HTML and TXT formats"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, OUTPUT_CONFIG["output_dir"])

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

        print(f"✅ [DEBUG] Report saved")
        return html_path
    
    def create_html(self, report: MonthlyDigest) -> str:
        """Create a simple HTML representation of the report"""

        html_text = f"""<!DOCTYPE html>
            <html lang="en">
                <head>
                    <meta charset="UTF-8" />
                    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
                    <title>{report.title}</title>
                    <style>
                        /* reset och layout */
                        html, body {{
                        height: 100%;
                        margin: 0;           /* remove default outer margin */
                        padding: 0;
                        box-sizing: border-box;
                        background-color: #FFFFFF; /* page background outside the report */
                        }}

                        /* själva "kortet" som innehåller rapporten */
                        .report-wrapper {{
                        box-sizing: border-box;
                        width: 100%;
                        max-width: 1200px;          /* läsbar kolumn på stora skärmar */
                        margin: 24px auto;          /* centrerat med lite yttre luft */
                        padding: 28px;              /* inre luft i rutan */
                        background-color: #FFFFFF;  /* bakgrund */
                        color: #000000;
                        border-radius: 10px;         /* valfritt: rundar hörn något */
                        line-height: 1.6;
                        }}

                        /* typsnitt & rubriker */
                        body {{
                        font-family: Archivo light;
                        font-size: 16px;            /* basstorlek */
                        }}

                        h1 {{
                        margin: 0 0 18px 0;
                        font-family: Archivo expanded bold;
                        font-size: clamp(1.4rem, 2.8vw, 2.4rem); /* skalar med viewport */
                        color: #003732;
                        line-height: 1.05;
                        }}

                        /* sektioner */
                        section {{
                        margin-bottom: 1.5rem;
                        }}

                        /* länkar */
                        a, a:link {{
                        color: #0064AA;
                        background-color: transparent;
                        text-decoration: none;
                        }}

                        a:visited {{
                        color: #0034AA;
                        text-decoration: none;
                        }}

                        a:hover, a:focus {{
                        color: #0034AA;
                        text-decoration: underline;
                        }}

                        a:active {{
                        color: #0094AA;
                        text-decoration: underline;
                        }}

                        /* säkerställ att långa rader bryts och inte orsakar horisontell scroll */
                        .report-wrapper, .report-wrapper * {{
                        word-wrap: break-word;
                        overflow-wrap: break-word;
                        }}
                    </style>
                </head>
                <body>
                    <div class="report-wrapper">
                        <h1>{report.title}</h1>
                        <section>
                            Model used for generation: {report.metadata.get('OpenAI model', None)}  &nbsp;&nbsp;&nbsp;&nbsp; Generated on: {report.generated_at}  &nbsp;&nbsp;&nbsp;&nbsp; Period: {report.period}  
                            &nbsp;&nbsp;&nbsp;&nbsp; Searches: {report.search_count}  &nbsp;&nbsp;&nbsp;&nbsp; Sources considered: {report.total_results}<br><br>
                            {report.content.replace('\n', '<br>')}
                        </section>
                    </div>
                </body>
            </html>"""
        return html_text
    
    def linkify_sources(self, html_text: str):
        url_regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
        all_urls = re.findall(url_regex, html_text)

        for url in all_urls:
            html_text = html_text.replace(url[0], f'<a href="{url[0]}" target="_blank">{url[0]}</a>')

        return html_text

def main():
    topic = "MarTech trends"
    generator = Generator()
    report = generator.generate_report(topic)

    if report:
        print(f"✅ [DEBUG] {report.title} generated successfully ({len(report.content)} chars).")
        print(f"📊 Digest Statistics:")
        print(f"   • Period: {report.period}")
        print(f"   • Searches: {report.search_count}")
        print(f"   • Sources: {report.total_results}")
        print(f"   • Word Count: {report.metadata.get('word_count', 'N/A')}")
        print(f"   • AI model: {report.metadata.get('OpenAI model', None)}")

        # Save to files
        file = generator.save_report(report)

        # Send report
        confirmation = input(f"Do you want to send the HTML report '{file}'? (Y/N): ").strip().lower()
        if confirmation == 'y':
            send_html_file(file, f"Differ Monthly Digest - {report.period}")
        else:
            print("Report not sent.")
    
    else:
        print(f"❌ [DEBUG] Report generation failed.")

if __name__ == "__main__":
    main()
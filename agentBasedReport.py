"""
Monthly Digest Generator using AI Agent and DDGS Search
Generates comprehensive monthly business digests with real-time web search
"""

import os
import time
import datetime
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
from dataclasses import dataclass
from ddgs import DDGS
from openai import OpenAI
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

# OpenAI Model Configuration
# Change this variable to switch between different OpenAI models
OPENAI_MODEL = "gpt-5-mini"
OPENAI_API_KEY = "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"

SEARCH_CONFIG = {
    "max_searches": 10,  # Increased for more comprehensive research
    "max_results_per_search": 10,  # Increased for more sources per search
    "max_tokens": 3000,  # Increased for more comprehensive reports
    "target_word_count": 800,  # Target 450-550 words
    "min_word_count": 700,
    "max_word_count": 1000,
    "min_sources": 10,  # Minimum sources required
    "max_sources": 30,   # Maximum sources to include
}

OUTPUT_CONFIG = {
    "output_dir": "Output",  # This will be relative to the script location
    "html_filename_template": "monthly_digest_{timestamp}.html",
    "txt_filename_template": "monthly_digest_{timestamp}.txt"
}

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SearchResult:
    """Structured search result data"""
    title: str
    url: str
    snippet: str
    source: str
    query: str
    timestamp: str

@dataclass
class MonthlyDigest:
    """Structured monthly digest data"""
    title: str
    period: str
    generated_at: str
    search_count: int
    total_results: int
    retries: int
    content: str  # The main digest content
    sources: List[SearchResult]
    metadata: Dict[str, Any]

# ============================================================================
# AI AGENT CONFIGURATION
# ============================================================================

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

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

class MonthlyDigestGenerator:
    """Main class for generating monthly digests"""
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        self.search_results: List[SearchResult] = []
        self.seen_urls: set[str] = set()
        self.allowed_domains: set[str] = set()
        # Track cumulative token usage
        self.token_usage = {"prompt": 0, "completion": 0, "total": 0}
        self.ensure_output_directory()
        self._load_domain_filter()

    def _normalize_url(self, url: str) -> str:
        """Normalize URL for deduplication: lowercase host, strip fragments, trailing slashes,
        and remove tracking params (utm_*, fbclid, gclid, ref, ref_src)."""
        if not url:
            return ""
        try:
            parsed = urlparse(url.strip())
            scheme = parsed.scheme.lower() or "http"
            netloc = parsed.netloc.lower()
            path = parsed.path.rstrip('/')
            # filter query params
            tracking_prefixes = ("utm_",)
            tracking_keys = {"fbclid", "gclid", "ref", "ref_src"}
            query_pairs = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=False)
                           if not (k.lower().startswith(tracking_prefixes) or k.lower() in tracking_keys)]
            query = urlencode(query_pairs, doseq=True)
            normalized = urlunparse((scheme, netloc, path, "", query, ""))
            return normalized
        except Exception:
            return url.strip()

    def _load_domain_filter(self) -> None:
        """Load allowed domains from domainFilter.json if present."""
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(script_dir, "domainFilter.json")
            if not os.path.exists(path):
                print("⚠️ [DEBUG] domainFilter.json not found - skipping domain filtering")
                return
            with open(path, "r", encoding="utf-8") as f:
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
        except Exception as e:
            print(f"⚠️ [DEBUG] Failed to load domainFilter.json: {e}")
            self.allowed_domains = set()

    def _is_domain_allowed(self, url: str) -> bool:
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

    def ensure_output_directory(self) -> None:
        """Ensure Output directory exists relative to script location"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, OUTPUT_CONFIG["output_dir"])
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 [DEBUG] Output directory: {output_dir}")
    
    def perform_web_search(self, query: str) -> List[SearchResult]:
        """Perform web search using DDGS with structured results and source vetting"""
        start_time = time.time()
        try:
            print(f"🔍 [DEBUG] Starting search for: {query}")
            print(f"🔍 [DEBUG] Initializing DDGS...")
            
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
                        timestamp=datetime.datetime.now().isoformat()
                    )
                    all_results.append(search_result)
                
                elapsed = time.time() - start_time
                print(f"✅ [DEBUG] Search completed: {len(all_results)} results in {elapsed:.2f}s")
                return all_results
                
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ [DEBUG] Search error after {elapsed:.2f}s: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def format_search_results_for_agent(self, results: List[SearchResult], query: str) -> str:
        """Format search results for agent analysis"""
        if not results:
            return "❌ No results found for this query."
        
        formatted = f"📰 SEARCH RESULTS for '{query}':\n"
        formatted += f"Found {len(results)} sources:\n\n"
        
        for i, result in enumerate(results, 1):
            formatted += f"**{i}. {result.title}**\n"
            formatted += f"   🔗 Source: {result.url}\n"
            formatted += f"   📄 Content: {result.snippet}\n\n"
        
        formatted += "📋 ANALYSIS INSTRUCTIONS:\n"
        formatted += "- Critically evaluate the sources for credibility, relevance and trustworthiness\n"
        formatted += "- If the source does not meet high standards, disregard it\n"
        formatted += "- Extract key developments, dates, companies, and business impacts\n"
        formatted += "- Note the source URL for each claim you make and refer to it as the source\n"
        formatted += "- Focus on strategic implications for business leaders\n"
        formatted += "- Continue with more searches if necessary or provide your professional digest"

        print(f"🔍 [DEBUG] Formatted search results for agent: {formatted}\n")
        
        return formatted
    
    def _extract_search_queries(self, agent_response: str) -> List[str]:
        """Extract all SEARCH: queries from the agent response (order preserved)."""
        import re
        if not agent_response:
            return []
        # capture any "SEARCH: ..." occurrence up to the newline (handles multiple per response)
        queries = [q.strip() for q in re.findall(r'(?i)SEARCH:\s*([^\n]+)', agent_response)]
        # drop empties and duplicates while preserving order
        seen = set()
        deduped = []
        for q in queries:
            if not q or q.lower() in seen:
                continue
            seen.add(q.lower())
            deduped.append(q)
        if deduped:
            print(f"🔍 [DEBUG] Extracted {len(deduped)} search quer{'y' if len(deduped)==1 else 'ies'}: {deduped}")
        else:
            print("🔍 [DEBUG] No SEARCH queries extracted")
        return deduped

    def _handle_search_request(self, agent_response: str, messages: List[Dict], search_count: int, failed_searches: int) -> tuple[int, int]:
        """Handle one or more agent search requests in a single response."""
        start_time = time.time()
        try:
            queries = self._extract_search_queries(agent_response)
            if not queries:
                print(f"🔍 [DEBUG] No valid search query found in response")
                return search_count, failed_searches + 1

            max_allowed = SEARCH_CONFIG["max_searches"]
            for idx, search_query in enumerate(queries, start=1):
                if search_count >= max_allowed:
                    print(f"🔍 [DEBUG] Max searches ({max_allowed}) reached; skipping remaining {len(queries) - (idx - 1)} queued quer{'y' if len(queries) - (idx - 1) == 1 else 'ies'}")
                    break

                print(f"🔍 [DEBUG] Processing queued search {idx}/{len(queries)}: {search_query}")
                search_results = self.perform_web_search(search_query)

                # Deduplicate by normalized URL and apply domain filtering
                deduped: List[SearchResult] = []
                for r in search_results:
                    norm = self._normalize_url(r.url)
                    if not norm or norm in self.seen_urls or not self._is_domain_allowed(norm):
                        continue
                    self.seen_urls.add(norm)
                    r.url = norm
                    deduped.append(r)

                # Count only successful searches as iterations
                if deduped:
                    search_count += 1
                    print(f"🔍 [DEBUG] Search success (counted). {len(deduped)} results after de-dup and filtering.")
                    results_text = self.format_search_results_for_agent(deduped, search_query)
                    print(f"🔍 [DEBUG] Adding results to conversation (length: {len(results_text)} chars)")
                    messages.append({"role": "user", "content": results_text})
                    self.search_results.extend(deduped)
                else:
                    failed_searches += 1
                    print(f"🔍 [DEBUG] Search failed/no usable results (NOT counted). Failed searches: {failed_searches}")
                    messages.append({"role": "user", "content": "❌ No results found. Try a different search query."})

            elapsed = time.time() - start_time
            print(f"🔍 [DEBUG] Search request handled in {elapsed:.2f}s (search_count: {search_count}, failed_searches: {failed_searches})")
            return search_count, failed_searches

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ [DEBUG] Error handling search request after {elapsed:.2f}s: {e}")
            import traceback
            traceback.print_exc()
            return search_count, failed_searches + 1

    def generate_digest(self) -> Optional[MonthlyDigest]:
        """Generate comprehensive monthly digest"""
        overall_start_time = time.time()
        try:
            current_date = datetime.date.today()
            period = f"{current_date.strftime('%B %Y')}"
            
            print(f"📰 [DEBUG] Starting monthly digest generation for {period}")
            print("=" * 60)
            
            # Initialize conversation
            initial_prompt = self._create_initial_prompt(period)
            print(f"📰 [DEBUG] Initial prompt created ({len(initial_prompt)} chars)")
            
            messages = [
                {"role": "system", "content": DIFFER_AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": initial_prompt}
            ]
            print(f"📰 [DEBUG] Conversation initialized with {len(messages)} messages")
            
            search_count = 0
            failed_searches = 0
            retries = 0
            
            # Research loop
            loop_start_time = time.time()
            while search_count < SEARCH_CONFIG["max_searches"]:
                iteration_start = time.time()
                print(f"🔄 [DEBUG] Starting iteration {search_count + 1}/{SEARCH_CONFIG['max_searches']}")
                
                response = self._get_agent_response(messages, period, search_count, retries)
                agent_response = response.choices[0].message.content.strip()
                messages.append({"role": "assistant", "content": agent_response})
                
                print(f"🔄 [DEBUG] Agent response received: {len(agent_response)} chars")
                print(f"🔄 [DEBUG] Response preview: {agent_response}")
                
                if "SEARCH:" in agent_response:
                    print(f"🔄 [DEBUG] Agent requested search")
                    search_count, failed_searches = self._handle_search_request(
                        agent_response, messages, search_count, failed_searches
                    )
                elif "MONTHLY_DIGEST:" in agent_response:
                    print(f"🔄 [DEBUG] Agent provided digest")
                    print(f"🔄 [DEBUG] Digest response length: {len(agent_response)} characters")
                    print(f"🔄 [DEBUG] Digest response preview: {agent_response}")
                    
                    if search_count < 4:  # Require minimum 4 searches for comprehensive research
                        print(f"🔄 [DEBUG] Insufficient searches ({search_count}), requesting more")
                        messages.append({"role": "user", "content": 
                            f"❌ You've only conducted {search_count} searches. You need at least 4 searches to create a comprehensive digest. Please conduct more searches before providing your final digest."})
                    else:
                        print(f"🔄 [DEBUG] Sufficient searches completed, validating digest")
                        digest = self._create_digest_from_response(
                            agent_response, period, search_count, retries
                        )
                        if digest is None:  # Validation failed
                            print(f"🔄 [DEBUG] Validation failed, checking reason")
                            source_count = len(self.search_results)
                            if source_count < SEARCH_CONFIG["min_sources"]:
                                print(f"🔄 [DEBUG] Insufficient sources ({source_count}), requesting more searches")
                                messages.append({"role": "user", "content": 
                                    f"❌ You only have {source_count} sources. You need at least {SEARCH_CONFIG['min_sources']} sources for a comprehensive digest. Please conduct more searches to gather additional information."})
                            else:
                                print(f"🔄 [DEBUG] Word count validation failed, requesting rewrite")
                                messages.append({"role": "user", "content": 
                                    f"❌ Your digest does not meet the word count requirements. You must provide a digest between {SEARCH_CONFIG['min_word_count']} and {SEARCH_CONFIG['max_word_count']} words for the MAIN CONTENT ONLY (excluding source references). Use the entire conversation history (all SEARCH RESULTS messages we sent earlier) as your notes and provide your final report now using MONTHLY_DIGEST: [your report]."})
                        else:
                            return digest
                else:
                    print(f"🔄 [DEBUG] Agent response unclear, providing guidance")
                    messages.append({"role": "user", "content": self._get_guidance_message() + "\n\nUse the conversation history above (SEARCH RESULTS messages) as your working notes. When ready, provide your digest with: MONTHLY_DIGEST: [your report]."})
                
                iteration_time = time.time() - iteration_start
                total_time = time.time() - loop_start_time
                print(f"🔄 [DEBUG] Iteration {search_count} completed in {iteration_time:.2f}s (total: {total_time:.2f}s)")
                print(f"🔄 [DEBUG] Current status: {search_count} searches, {failed_searches} failures")
                print("-" * 50)
            
            # Force completion if max searches reached
            print(f"📰 [DEBUG] Max searches reached, forcing completion")
            return self._force_digest_completion(messages, period, search_count, retries)
            
        except Exception as e:
            elapsed = time.time() - overall_start_time
            print(f"❌ [DEBUG] Error generating digest after {elapsed:.2f}s: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            total_elapsed = time.time() - overall_start_time
            print(f"📰 [DEBUG] Total digest generation time: {total_elapsed:.2f}s")
    
    def _create_initial_prompt(self, period: str) -> str:
        """Create initial prompt for agent"""
        return f"""
            You have been given a task by the CEO of Differ to create a professional monthly business digest for {period}.

            Your task: Research and analyze the most significant business developments from the last 30 days, focusing on the Swedish and Nordic markets.

            CRITICAL REQUIREMENTS:
            - Target word count: {SEARCH_CONFIG['target_word_count']} words (range: {SEARCH_CONFIG['min_word_count']}-{SEARCH_CONFIG['max_word_count']}) for MAIN CONTENT ONLY (excludes source references)
            - Minimum {SEARCH_CONFIG['min_sources']} sources required (maximum {SEARCH_CONFIG['max_sources']})
            - Professional, but digestible tone that is easy to read and understand
            - Include specific source references for all claims (references are separate from word count)

            HOW TO CONDUCT RESEARCH:
            1. Use the format "SEARCH: [your query]" to request web searches
            2. I will provide you with search results but I can only process one query at a time
            3. Analyze the results and request more searches if needed
            4. When you have enough information, provide your digest using the format: "MONTHLY_DIGEST: [your report]"

            DIGEST FORMAT:
            - Executive Summary (2-3 sentences)
            - Main body: list the most interesting developments in bullet points with short summaries and references
            - Source References (numbered list of URLs used - this section is NOT counted toward word count)
            """
    
    def _get_agent_response(self, messages: List[Dict], period, search_count, retries) -> Any:
        """Get response from OpenAI agent"""
        start_time = time.time()
        try:
            print(f"🤖 [DEBUG] Calling OpenAI API...")
            print(f"🤖 [DEBUG] Message count: {len(messages)}")
            print(f"🤖 [DEBUG] Last message: {messages[-1]['content']}")
            
            response = self.openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                max_completion_tokens=SEARCH_CONFIG["max_tokens"],
            )

            if len(response.choices[0].message.content.strip()) == 0:
                retries += 1
                if retries > 5:
                    print(f"🤖 [DEBUG] OpenAI API response is empty, max retries reached.")
                    return self._force_digest_completion(messages, period, search_count, retries)
                print(f"🤖 [DEBUG] OpenAI API response is empty, retrying...")
                messages.append({"role": "user", "content": "Your response contains no content. Please try again, make sure to follow the instructions and provide a response."})
                messages.append({"role": "assistant", "content": messages[-1]['content'] + f" Retries: {retries}"})
                return self._get_agent_response(messages, period, search_count, retries)
            
            elapsed = time.time() - start_time
            print(f"🤖 [DEBUG] OpenAI API response received in {elapsed:.2f}s")

            # Token usage tracking (per-call and cumulative)
            usage = getattr(response, "usage", None)
            # Defensive extraction for multiple schema variants
            prompt_tokens = None
            completion_tokens = None
            total_tokens = None
            if usage is not None:
                prompt_tokens = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None)
                completion_tokens = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None)
                total_tokens = getattr(usage, "total_tokens", None)
                # Fallback compute total if missing
                if total_tokens is None and (prompt_tokens is not None or completion_tokens is not None):
                    total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

            # Update cumulative totals safely
            p = int(prompt_tokens or 0)
            c = int(completion_tokens or 0)
            t = int(total_tokens or (p + c))
            self.token_usage["prompt"] += p
            self.token_usage["completion"] += c
            self.token_usage["total"] += t
            print(f"🤖 [DEBUG] Token usage this call: prompt={p}, completion={c}, total={t}; "
                  f"Cumulative: prompt={self.token_usage['prompt']}, completion={self.token_usage['completion']}, total={self.token_usage['total']}")

            return response
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ [DEBUG] OpenAI API error after {elapsed:.2f}s: {e}")
            import traceback
            traceback.print_exc()
            raise
       
    def _create_digest_from_response(self, agent_response: str, period: str, search_count: int, retries: int) -> MonthlyDigest:
        """Create structured digest from agent response - simplified"""
        digest_content = agent_response.split("MONTHLY_DIGEST:")[1].strip()
        
        # Simple word count validation
        word_count = len(digest_content.split())
        source_count = len(self.search_results)
        print(f"✅ Digest completed after {search_count} searches ({word_count} words, {source_count} sources)")
        
        # Basic validation
        if word_count < SEARCH_CONFIG["min_word_count"]:
            print(f"⚠️ Warning: Digest is {word_count} words, below minimum of {SEARCH_CONFIG['min_word_count']}")
            return None
        elif source_count < SEARCH_CONFIG["min_sources"]:
            print(f"⚠️ Warning: Only {source_count} sources found, below minimum of {SEARCH_CONFIG['min_sources']}")
            return None
        
        return MonthlyDigest(
            title=f"Monthly Business Digest - {period}",
            period=period,
            generated_at=datetime.datetime.now().isoformat(),
            search_count=search_count,
            total_results=len(self.search_results),
            retries=retries,
            content=digest_content,
            sources=self.search_results,
            metadata={
                "generator": "Differ AI Agent",
                "search_engine": "DDGS",
                "model": OPENAI_MODEL,
                "version": "1.0",
                "word_count": word_count,
                "source_evaluation": "AI Agent Judged"
            }
        )
    
    def _force_digest_completion(self, messages: List[Dict], period: str, search_count: int, retries: int) -> Optional[MonthlyDigest]:
        """Force digest completion when max searches reached"""
        messages.append({"role": "user", "content": 
            f"""You have reached the maximum number of searches ({search_count}).
            You must now provide your monthly digest.
            The digest must be between {SEARCH_CONFIG['min_word_count']} and {SEARCH_CONFIG['max_word_count']} words for the main content (excluding source references).

            Include:
            - Executive Summary (2-3 sentences)
            - Main body: list the most interesting developments in bullet points with short summaries and references
            - Source References (numbered list of URLs used - this section is NOT counted toward word count)

            Provide your digest now using "MONTHLY_DIGEST: [your report]".
            Use the entire conversation history (every SEARCH RESULTS message) as your evidence base.
            """})
        
        response = self._get_agent_response(messages, period, search_count, retries)
        final_response = response.choices[0].message.content.strip()
        if len(final_response) == 0:
            if retries > 5:
                print(f"❌ [DEBUG] Final agent response is empty, max retries reached.")
                raise Exception("Final agent response is empty after maximum retries.")
            else:
                retries += 1
                print(f"❌ [DEBUG] Final agent response is empty, retrying... Retries: {retries}")
                messages.append({"role": "user", "content": "Your response contains no content. Please try again, make sure to follow the instructions and provide a response. Retries: {retries}"})
                return self._force_digest_completion(messages, period, search_count, retries)
        else:
            # Debug: Show what the agent actually returned
            print(f"🔍 [DEBUG] Final agent response length: {len(final_response)} characters")
            return self._create_digest_from_response(final_response, period, search_count, retries)
    
    def _get_guidance_message(self) -> str:
        """Get guidance message for agent"""
        return f"""I can't parse that response. Please try again using SEARCH: [your query] to request web searches.
        I will provide you with search results.
        When you have enough information, provide your digest using: "MONTHLY_DIGEST: [your report]"
        Start with a simple search query using "SEARCH: [your query]" format.
        """
    
    def _format_swedish_period(self, period: str) -> str:
        """Return 'Oktober 2025' style from 'October 2025'."""
        months = {
            "january": "Januari", "february": "Februari", "march": "Mars",
            "april": "April", "may": "Maj", "june": "Juni",
            "july": "Juli", "august": "Augusti", "september": "September",
            "october": "Oktober", "november": "November", "december": "December",
        }
        try:
            m, y = period.split()
            return f"{months.get(m.lower(), m)} {y}"
        except Exception:
            return period

    def _parse_digest_content(self, content: str) -> dict:
        """Extract summary (2–3 sentences), bullets, strategic implications."""
        import re
        text = (content or "").replace("\r\n", "\n").replace("\r", "\n").strip()

        # Collect bullets, implications and paragraphs
        bullets, implications, paragraphs = [], [], []
        for raw in text.split("\n"):
            s = raw.strip()
            if not s:
                continue
            if re.match(r"^(\-|\*|•|–|—|\d+[\.)])\s+", s):
                bullets.append(re.sub(r"^(\-|\*|•|–|—|\d+[\.)])\s+", "", s))
                continue
            if s.lower().startswith("strategic implication"):
                # Keep only the message after the colon if present
                parts = s.split(":", 1)
                implications.append(parts[1].strip() if len(parts) > 1 else s)
                continue
            paragraphs.append(s)

        # Summary: first non-empty paragraph -> first 2–3 sentences
        summary_src = next((p for p in paragraphs if p), "")
        sentences = re.split(r"(?<=[\.!\?])\s+", summary_src)
        summary = " ".join(sentences[:3]).strip()

        # Split bullets into key and main
        key_bullets = bullets[:4]
        main_bullets = bullets[4:]

        return {
            "summary": summary,
            "key_bullets": key_bullets,
            "main_bullets": main_bullets,
            "implications": implications,
        }

    def _generate_html_digest(self, digest: MonthlyDigest) -> str:
        """Generate HTML in the requested layout and styling."""
        import html as _html
        import datetime
        import re

        def _esc(s: str) -> str:
            return _html.escape(s or "", quote=True)

        url_re = re.compile(r"(https?://[^\s<>'\"]+)")
        def _linkify_safe(text: str) -> str:
            if not text:
                return ""
            parts, last = [], 0
            for m in url_re.finditer(text):
                parts.append(_esc(text[last:m.start()]))
                url = m.group(0)
                trail = ""
                while url and url[-1] in ".,;:)]}":
                    trail = url[-1] + trail
                    url = url[:-1]
                parts.append(f'<a href="{_esc(url)}" target="_blank">{_esc(url)}</a>' + _esc(trail))
                last = m.end()
            parts.append(_esc(text[last:]))
            return "".join(parts)

        # Meta
        title = _esc(digest.title)
        period = _esc(digest.period)
        try:
            gen_dt = datetime.datetime.fromisoformat(digest.generated_at)
            generated_str = gen_dt.strftime("%B %d, %Y")
        except Exception:
            generated_str = _esc(digest.generated_at or "")

        sources_count = digest.total_results if hasattr(digest, "total_results") else len(getattr(digest, "sources", []))
        word_count = digest.metadata.get("word_count", "N/A") if digest.metadata else "N/A"

        # Parse content into sections
        parts = self._parse_digest_content(digest.content or "")
        summary = parts["summary"]
        key_bullets = parts["key_bullets"]
        main_bullets = parts["main_bullets"]
        implications = parts["implications"]

        def render_bullets(items: list[str]) -> str:
            return "".join(
                f'<div class="bullet"><span class="bullet-dot"></span><span class="bullet-text">{_linkify_safe(it)}</span></div>'
                for it in items if it
            )

        # Highlight title (Swedish month + region), with conditional disclaimer
        sv_period = self._format_swedish_period(digest.period)
        region_suffix = " — Sverige &amp; Norden"
        disclaimer_needed = sources_count < SEARCH_CONFIG.get("min_sources", 10)
        disclaimer_text = (
            "Detta digest är ett analytiskt, plausibelt underlag framtaget utan direkt åtkomst till nyhetskällor för perioden. "
            "För publicering eller beslutstagande behöver vi verifiera alla påståenden mot minst 10 källor."
            if disclaimer_needed else
            "Detta digest är AI-assisterat och baseras på sammanställda källor. Verifiera kritiska beslut mot originalkällor."
        )
        highlighted_title = f"{_esc(sv_period)}{region_suffix} ({'hypotetisk, ej källbelagd' if disclaimer_needed else 'källbelagd, AI-assisterad'})"

        # Optional KPI grid (provide via digest.metadata['kpis'] = [str, ...])
        kpis = digest.metadata.get("kpis", []) if digest.metadata else []
        kpi_html = ""
        if kpis:
            boxes = "".join(f'<div class="kpi-box">{_linkify_safe(k)}</div>' for k in kpis)
            kpi_html = f'''
        <div class="section">
            <div class="content">
                <div class="section-title">Mätbara KPI-förslag</div>
                <div class="kpi-grid">{boxes}</div>
            </div>
        </div>'''

        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
    <title>{title}</title>
    <style>
        :root {{
            --differ-dark: #003732;
            --differ-green: #328278;
            --differ-blue: #0064AA;
            --differ-red: #FA6464;
            --differ-grey: #EBE6DC; /* light tan used on dark bg */
        }}

        /* Basic layout */
        html,body{{height:100%}}
        body {{
            margin: 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #000000;
            color: var(--differ-grey);
            -webkit-font-smoothing:antialiased;
            -moz-osx-font-smoothing:grayscale;
        }}
        .container {{
            max-width: 760px;
            margin: 0 auto;
            padding: 24px 16px 48px;
        }}

        /* Title */
        .title {{
            margin: 0 0 12px 0;
            font-weight: 600;
            color: var(--differ-green);
            font-size: 28px;
        }}

        /* Sections & dividers */
        .section {{
            border-bottom: 1px solid rgba(255,255,255,0.12); /* subtle white divider */
        }}
        /* Sections with white background (meta, key developments, etc.) */
        .section.white-bg {{
            background: #ffffff;
            color: #333333;
            border-radius: 8px;
            padding: 12px 16px;
            border-bottom: none; /* skip dividing line for white-bg sections */
        }}
        .section.white-bg .content,
        .section.white-bg .bullet-text,
        .section.white-bg .meta-item {{
            color: #333333;
        }}
        .section.white-bg .bullet-dot {{ background: var(--differ-green); }}
        .section.white-bg .content a {{ color: var(--differ-blue); }}

        /* Meta grid styling (Period, Generated, Sources, Word Count) */
        .meta-grid {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 10px;
            align-items: start;
        }}
        .meta-item span {{
            display: block;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: .04em;
            color: #000000; /* black labels on white */
            margin-bottom: 2px;
            font-weight: 700;
        }}
        .meta-item {{
            color: #666666; /* grey answers */
            font-size: 14px;
            margin: 0;
        }}
        /* Decorative white lines for specific section */
        .highlighted-title {{
          border-top: 1px solid white;
          border-bottom: 1px solid white;
          padding: 8px 0;
          font-weight: 700;
          color: var(--differ-green);
        }}

        /* Content & headings */
        .content {{
            white-space: pre-wrap;
            line-height: 1.45;
            color: var(--differ-grey);
            font-size: 15px;
        }}
        .section-title {{
            font-weight: 700;
            color: var(--differ-green); /* unified green headers by default */
            margin: 0 0 2px 0; /* minimal gap to following content */
            font-size: 16px;
            line-height: 1.2;
            margin-bottom: 0px;
        }}
        .section-title.red {{
            color: var(--differ-red); /* special red title */
            margin-bottom: 10px;
        }}

        /* Bullets - tighter spacing + larger circular bullet */
        .bullet {{
            margin: 4px 0;
            display: flex;
            align-items: flex-start;
            gap: 10px;
        }}
        .bullet-dot {{
            height: 8px;
            background: var(--differ-green);
            border-radius: 50%;
            display: inline-block;
            flex: 0 0 8px;
            margin-top: 6px; /* visually align with first text line */
            font-size: 0;    /* hide any content in the span */
        }}
        .bullet-text {{
            color: var(--differ-grey);
            line-height: 1.3;
            font-size: 15px;
            margin: 0;
            margin-bottom: 5px;
        }}

        /* Links */
        .content a {{ color: var(--differ-blue); text-decoration: none; }}
        .content a:hover {{ text-decoration: underline; }}

        /* KPI boxes (two per row) */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin-top: 8px;
        }}
        .kpi-box {{
            background: var(--differ-green);
            color: #ffffff;
            padding: 12px;
            border-radius: 8px;
            font-size: 14px;
            line-height: 1.4;
        }}
        /* Make last lonely box span whole width for balance */
        .kpi-grid .kpi-box:nth-last-child(1):nth-child(odd) {{
            grid-column: 1 / -1;
        }}

        /* Sources headings */
        .sources h2 {{
            margin: 0 0 6px 0;
            color: var(--differ-green);
            font-size: 16px;
        }}
        .source-item {{ padding: 6px 0; color: var(--differ-grey); }}
        .source-item a {{ color: var(--differ-blue); text-decoration: none; word-break: break-all; }}

        /* Footer */
        .footer {{
            background: var(--differ-green);
            color: #ffffff;
            text-align: center;
            font-size: 12px;
            margin-top: 40px;
            padding: 12px;
            border-radius: 6px;
        }}

        /* Responsive */
        @media (max-width: 680px) {{
            .meta-grid {{ grid-template-columns: 1fr; }}
            .kpi-grid {{ grid-template-columns: 1fr; }}
        }}

        @media (max-width: 480px) {{
            .title {{ font-size: 22px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1 class="title">{title}</h1>

        <!-- Meta Section (white background) -->
        <div class="section white-bg meta">
            <div class="meta-grid">
                <div class="meta-item"><span>Period</span>{period}</div>
                <div class="meta-item"><span>Generated</span>{generated_str}</div>
                <div class="meta-item"><span>Sources</span>{sources_count}</div>
                <div class="meta-item"><span>Word Count</span>{word_count}</div>
            </div>
        </div>
            </div>
        </div>

        <!-- Huvudbody — intressanta utvecklingar (dark) -->
        <div class="section">
            <div class="content">
                <div class="section-title">Huvudbody — intressanta utvecklingar</div>{render_bullets(main_bullets)}
            </div>
        </div>'''

        # Strategic implications (only if present)
        if implications:
            html += f'''
        <!-- Strategic implications — konkreta rekommendationer -->
        <div class="section">
            <div class="content">
                <div class="section-title">Strategic implications — konkreta rekommendationer för klienter</div>
                {render_bullets(implications)}
            </div>
        </div>'''

        # Optional KPI grid if provided
        if kpi_html:
            html += kpi_html

        # Next steps (static CTA)
        html += f'''
        <!-- Next steps from Differ -->
        <div class="section">
            <div class="content">
                <div class="section-title">Nästa steg från Differ (förslag)</div><div class="bullet"><span class="bullet-dot"></span><span class="bullet-text">Om du vill ha full verifierad digest: skicka 10–30 källor (titel + URL + 1–2 raders utdrag). Vi levererar en uppdaterad version med numrerade referenser och eventuella korrigeringar.</span></div><div class="bullet"><span class="bullet-dot"></span><span class="bullet-text">Alternativt: be oss producera en prioriterad 1-sidig handlingsplan för ert företag baserat på ovan (example / draft).</span></div>
                <div style="margin-top:8px;">Behöver du att vi hjälper till att formulera exakta sökfraser eller samlar källor åt dig (arbetssteg där du klistrar in sökresultat)? Vi stöttar gärna — på Differ rekommenderar vi verifiering innan beslut.
                </div>
            </div>
        </div>

        <!-- Footer (green background) -->
        <div class="footer">Generated by Differ • {getattr(digest, 'search_count', '')} searches • {word_count} words</div>
    </div>
</body>
</html>'''

        return html

    def save_digest(self, digest: MonthlyDigest) -> Dict[str, str]:
        """Save digest in both HTML and TXT formats"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, OUTPUT_CONFIG["output_dir"])
        
        # Save HTML
        html_filename = OUTPUT_CONFIG["html_filename_template"].format(timestamp=timestamp)
        html_path = os.path.join(output_dir, html_filename)
        
        html_content = self._generate_html_digest(digest)
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Save TXT
        txt_filename = OUTPUT_CONFIG["txt_filename_template"].format(timestamp=timestamp)
        txt_path = os.path.join(output_dir, txt_filename)
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"{digest.title}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Period: {digest.period}\n")
            f.write(f"Generated: {digest.generated_at}\n")
            f.write(f"Searches conducted: {digest.search_count}\n")
            f.write(f"Total results: {digest.total_results}\n")
            f.write(f"Word count: {digest.metadata.get('word_count', 'N/A')}\n")
            f.write(f"Trusted sources only: {digest.metadata.get('trusted_sources_only', False)}\n\n")
            
            # Write main content
            f.write("## Digest Content\n")
            f.write(digest.content + "\n\n")
            
            # Write source references
            f.write("## Source References\n")
            f.write("All information sourced from verified professional publications:\n\n")
            for i, source in enumerate(digest.sources, 1):
                f.write(f"{i}. {source.title}\n")
                f.write(f"   URL: {source.url}\n")
                f.write(f"   Query: {source.query}\n\n")
        
        return {"html": html_path, "txt": txt_path}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    main_start_time = time.time()
    print("🚀 [DEBUG] Monthly Digest Generator Starting")
    print("=" * 60)
    
    try:
        print("🚀 [DEBUG] Initializing MonthlyDigestGenerator...")
        generator = MonthlyDigestGenerator()
        print("🚀 [DEBUG] Generator initialized successfully")
        
        print("🚀 [DEBUG] Starting digest generation...")
        digest = generator.generate_digest()
        
        if digest:
            print("\n✅ [DEBUG] Monthly Digest Generated Successfully!")
            print("=" * 60)
            
            # Save digest
            file_paths = generator.save_digest(digest)
            
            print(f"📊 Digest Statistics:")
            print(f"   • Period: {digest.period}")
            print(f"   • Searches: {digest.search_count}")
            print(f"   • Sources: {digest.total_results} (target: {SEARCH_CONFIG['min_sources']}-{SEARCH_CONFIG['max_sources']})")
            print(f"   • Word Count: {digest.metadata.get('word_count', 'N/A')} (target: {SEARCH_CONFIG['min_word_count']}-{SEARCH_CONFIG['max_word_count']})")
            print(f"🤖 [DEBUG] Total token usage: prompt={generator.token_usage['prompt']}, completion={generator.token_usage['completion']}, total={generator.token_usage['total']}")
            
            # Display content preview
            print(f"\n📋 Content Preview:")
            print(f"   {digest}...")
            
        else:
            print("\n❌ [DEBUG] Failed to generate monthly digest")
            # Still report total tokens used for visibility
            print(f"🤖 [DEBUG] Total token usage: prompt={generator.token_usage['prompt']}, completion={generator.token_usage['completion']}, total={generator.token_usage['total']}")
            
    except Exception as e:
        elapsed = time.time() - main_start_time
        print(f"\n❌ [DEBUG] Main function error after {elapsed:.2f}s: {e}")
        import traceback
        traceback.print_exc()
    finally:
        total_elapsed = time.time() - main_start_time
        print(f"\n🚀 [DEBUG] Total execution time: {total_elapsed:.2f}s")

if __name__ == "__main__":
    main()

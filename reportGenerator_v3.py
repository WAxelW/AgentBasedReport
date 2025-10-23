#!/usr/bin/env python3
"""
Differ Monthly Newsletter Generator v3 (Enhanced)

- Extracts themes from Differ.se website
- Fetches articles from curated RSS sources with domain filtering
- Deduplicates URLs & canonicalizes links
- Summarizes with GPT in structured JSON format
- Builds HTML digest with proper theming
- Caches results for efficiency
"""

import json
import os
import time
import datetime
import re
import requests
from urllib.parse import urlparse, urlunparse, quote_plus, parse_qs, unquote
import feedparser
from bs4 import BeautifulSoup
from dataclasses import dataclass
from typing import List, Optional, Tuple
from openai import OpenAI

# ---------------- CONFIG ----------------
OPENAI_API_KEY = "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"
MAX_ARTICLES = 30
MAX_SUMMARIES_PER_RUN = 15
MAX_PER_THEME = 3
CACHE_FILE = os.path.join(os.path.dirname(__file__), "summaries_cache.json")
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; DifferNewsDigest/1.0)"}

# High-authority domains for content filtering
TRUSTED_DOMAINS = {
    # Major news & business
    "bbc.com", "bbc.co.uk", "reuters.com", "bloomberg.com", "wsj.com", "ft.com",
    "economist.com", "nytimes.com", "washingtonpost.com", "guardian.com",
    
    # Tech & business publications
    "techcrunch.com", "venturebeat.com", "wired.com", "theverge.com", 
    "arstechnica.com", "engadget.com", "mashable.com", "techinasia.com",
    
    # Business & consulting
    "hbr.org", "mckinsey.com", "bcg.com", "bain.com", "deloitte.com", 
    "pwc.com", "kpmg.com", "accenture.com", "ibm.com",
    
    # European sources
    "di.se", "svd.se", "dn.se", "aftonbladet.se", "expressen.se",
    "handelsblatt.com", "spiegel.de", "zeit.de", "lemonde.fr",
    
    # Software & SaaS
    "salesforce.com", "hubspot.com", "zendesk.com", "microsoft.com",
    "google.com", "amazon.com", "oracle.com", "adobe.com",
    
    # Financial & fintech
    "forbes.com", "cnbc.com", "marketwatch.com", "investopedia.com",
    "stripe.com", "paypal.com", "visa.com", "mastercard.com"
}

# ---------------- OpenAI client ----------------
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------- Dataclass ----------------
@dataclass
class Article:
    keyword: str
    title: str
    link: str
    published: Optional[datetime.datetime]
    hostname: str
    source_hint: str
    summary: Optional[str] = None
    why_it_matters: Optional[str] = None
    suggested_action: Optional[str] = None
    tags: Optional[List[str]] = None
    score: float = 0.0

# ---------------- Differ Agent Integration ----------------
DIFFER_AGENT_SYSTEM_PROMPT = """
You are an AI agent that role-plays as a real employee at Differ — a multidisciplinary consultancy and creative agency focused on sustainable growth. Use the facts and tone below as your knowledge base and style guide. When in doubt, be consultative, practical, sustainability-minded, and concise.

Knowledge & facts to use:
- Differ's purpose: enable sustainable growth — profitable and good for planet & society.
- Services: strategy, digital/technology, creative agency; typical areas: brand strategy, CX, digital marketing, martech, change management, transformation, sustainability-driven strategy.
- Organisation: Differ Strategy, Differ Technology, Differ Agency — use appropriate lens for answers.
- People & contacts: Swedish offices (Stockholm, Malmö, Norrköping), general contact info@differ.se, +46 8 519 510 00.
- Sustainability: integrate sustainability into brand and strategy, PUSH community/festival, measureable impact.

Persona & tone:
- Professional, consultative, pragmatic.
- Positive, curious; use "we/us/at Differ" in recommendations.
- Emphasize measurable impact, prioritization, feasibility.
- Default language: Swedish for Swedish users; otherwise match user language.
- Concise answers (3–6 short paragraphs/bullets) + clear next step.

Behaviours & deliverables:
- Quick assessments, prioritized action lists, workshop agendas, brief project scoping, KPIs, martech recommendations, content/SEO guidance, templates.
- Example slides/outlines labeled "example / draft".
- Include rationale, expected impact, estimated effort, suggested next step.
- Cite Differ case studies or public sources when applicable.

Safety / escalation rules:
- Do not give legal/compliance advice; refer to experts.
- Do not give legal/compliance advice; refer to experts.
- Do not access confidential client files; suggest template/process for info gathering.
- Commercial engagement: suggest ballpark guidance only, refer to info@differ.se.

Prompt template for responses:
1. One-sentence summary of user request.
2. Prioritized recommendation with rationale, impact, effort.
3. Clear next step.

Output formatting:
- Bullets for actions.
- Summary at top.
- Labels: Effort: low/medium/high, Impact: low/medium/high.
- Optional 2–3 line "how we would work" summary.
"""

def query_differ_agent_for_insights():
    """Query the Differ agent for specific recent market developments"""
    try:
        # Get current date for context
        current_date = datetime.date.today().strftime("%B %Y")
        current_month = datetime.date.today().strftime("%B")
        current_year = datetime.date.today().year
        
        # Ask the agent for specific recent developments, not generic trends
        agent_prompt = f"""
        You are a senior consultant at Differ, a Stockholm-based multidisciplinary consultancy focused on sustainable growth. Your sole task is to thoroughly investigate the current business landscape and synthesize accurate, penetrating analyses of our environment.
        
        Today's date is {current_date}. Analyze the most significant market developments from {current_month} {current_year} (the last 30 days) that directly impact our clients' business strategies.
        
        As a Stockholm-based consultant, consider:
        - European and Nordic market dynamics
        - Swedish regulatory environment and business climate
        - Global trends affecting Swedish and European businesses
        - Technology adoption patterns in the Nordic region
        - Sustainability and ESG developments relevant to European markets
        
        Focus on CONCRETE, RECENT developments in:
        - Technology breakthroughs and AI implementations
        - Regulatory changes and compliance requirements (especially EU/Nordic)
        - Major business acquisitions, partnerships, or market shifts
        - Sustainability and ESG regulatory updates
        - Digital transformation case studies and results
        - Marketing technology innovations and adoption
        
        For each development, provide:
        1. Specific event/development (not generic trends)
        2. Concrete impact on business operations
        3. Immediate action items for clients
        4. Relevant keywords for finding supporting articles
        
        Be specific about dates, companies, and concrete outcomes. Avoid generic buzzwords.
        Focus on developments that require client attention or strategic response.
        Think like a dedicated consultant whose only job is to understand the market landscape.
        
        IMPORTANT: At the end of your analysis, provide a "KEYWORDS FOR SEARCH" section with a simple list of 10-30 general but relevant search terms that would help find articles about these developments. Use broad terms that would appear in news headlines, not specific company names. Format it like this:
        
        KEYWORDS FOR SEARCH:
        - [keyword 1]
        - [keyword 2]
        - [keyword 3]
        etc.
        
        Examples of good keywords: "AI regulation", "digital transformation", "sustainability reporting", "cybersecurity", "data privacy", "automation", "cloud computing", "fintech", "martech", "ESG compliance"
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": DIFFER_AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": agent_prompt}
            ],
            max_tokens=1000,
            temperature=0.2
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Error querying Differ agent: {e}")
        return None

def extract_keywords_from_agent_insights(insights: str) -> List[str]:
    """Extract keywords from the agent's structured KEYWORDS FOR SEARCH section"""
    if not insights:
        # Fallback to predefined keywords
        return [
            "Artificial Intelligence", "AI", "Machine Learning", "Digital Transformation",
            "Strategy", "Consulting", "Business Development", "CRM", "Customer Relationship",
            "Leadership", "Innovation", "Automation", "Data Analytics",
            "Cloud Computing", "Cybersecurity", "Fintech", "MarTech", "SaaS",
            "Sustainability", "ESG", "Green Technology", "Carbon Neutral"
        ]
    
    # Look for the structured "KEYWORDS FOR SEARCH" section
    keywords_section = re.search(r'KEYWORDS FOR SEARCH:\s*\n(.*?)(?:\n\n|\n[A-Z]|$)', insights, re.DOTALL | re.IGNORECASE)
    
    if keywords_section:
        keywords_text = keywords_section.group(1)
        # Extract keywords from bullet points
        keywords = []
        for line in keywords_text.split('\n'):
            line = line.strip()
            if line.startswith('-') or line.startswith('•'):
                keyword = line[1:].strip()
                if keyword:
                    keywords.append(keyword)
        
        if keywords:
            print(f"[extract_keywords] Found {len(keywords)} structured keywords from agent")
            # Add some general fallback keywords to improve article matching
            fallback_keywords = ["AI", "technology", "business", "digital", "innovation", "strategy", "sustainability", "data", "automation", "transformation"]
            keywords.extend(fallback_keywords)
            return keywords
    
    # Fallback: extract from the main text if no structured section found
    print("[extract_keywords] No structured keywords found, extracting from main text")
    keywords = set()
    text_lower = insights.lower()
    
    # Extract meaningful business terms (4+ characters)
    words = re.findall(r'\b[a-z]{4,}\b', text_lower)
    
    # Add significant terms, preserving the agent's nuanced understanding
    for word in words:
        # Skip very common words that don't add value
        if word not in {'this', 'that', 'with', 'from', 'they', 'been', 'have', 'will', 'your', 'said', 'each', 'which', 'their', 'time', 'very', 'when', 'much', 'than', 'then', 'them', 'these', 'some', 'what', 'only', 'other', 'into', 'over', 'think', 'also', 'back', 'after', 'first', 'well', 'work', 'such', 'make', 'like', 'long', 'take', 'here', 'just', 'know', 'good', 'year', 'most', 'even', 'many', 'more', 'about', 'there', 'through', 'during', 'before', 'where', 'should', 'could', 'would'}:
            keywords.add(word.title())
    
    # Always add some general fallback keywords to improve matching
    keywords.update([
        "Artificial Intelligence", "AI", "Digital Transformation", "Strategy",
        "Sustainability", "MarTech", "Customer Experience", "Innovation",
        "Technology", "Business", "Data", "Automation", "Cloud", "Security"
    ])
    
    return list(keywords)

# ---------------- Utilities ----------------
def canonicalize(url: str) -> str:
    try:
        p = urlparse(url)
        return urlunparse(p._replace(query="", params="", fragment=""))
    except Exception:
        return url

def resolve_url(url: str, timeout: int = 8) -> str:
    try:
        # Unwrap Google consent pages to the real destination when present
        for _ in range(2):
            parsed = urlparse(url)
            host = (parsed.hostname or "").lower()
            if host == "consent.google.com":
                qs = parse_qs(parsed.query)
                cont = qs.get("continue", [None])[0]
                if cont:
                    url = unquote(cont)
                    continue
            break

        resp = requests.get(url, timeout=timeout, headers=HEADERS, allow_redirects=True)
        resp.raise_for_status()
        return resp.url
    except Exception:
        return url

# ---------------- Fetch Articles ----------------
def fetch_articles(keywords: List[str], max_articles=MAX_ARTICLES) -> List[Article]:
    rss_sources = [
        # Tech & Business (high-quality, reliable feeds)
        "https://feeds.feedburner.com/techcrunch/",
        "https://feeds.feedburner.com/arstechnica/",
        "https://feeds.feedburner.com/wired/",
        "https://feeds.feedburner.com/theverge/",
        "https://feeds.feedburner.com/engadget/",
        "https://feeds.feedburner.com/venturebeat/SZYF",
        
        # Business & Strategy (consulting firms)
        "https://feeds.feedburner.com/mckinsey/",
        "https://feeds.feedburner.com/bcg/",
        "https://feeds.feedburner.com/bain/",
        "https://feeds.feedburner.com/deloitte/",
        "https://feeds.feedburner.com/pwc/",
        
        # High-quality business publications
        "https://www.ft.com/rss/home",
        "https://feeds.feedburner.com/forbes/",
        "https://feeds.bloomberg.com/markets/news.rss",
        "https://feeds.feedburner.com/bloomberg/",
        "https://feeds.feedburner.com/reuters/",
        
        # European sources
        "https://feeds.bbci.co.uk/news/technology/rss.xml",
        "https://feeds.bbci.co.uk/news/business/rss.xml",
        "https://www.di.se/rss/",
        "https://feeds.feedburner.com/handelsblatt/",
        
        # Software & SaaS
        "https://feeds.feedburner.com/salesforce/",
        "https://feeds.feedburner.com/hubspot/",
        "https://feeds.feedburner.com/zendesk/",
        "https://feeds.feedburner.com/microsoft/",
        
        # Financial & Fintech
        "https://feeds.feedburner.com/cnbc/",
        "https://feeds.feedburner.com/marketwatch/",
        "https://feeds.feedburner.com/stripe/",
    ]
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=30)
    articles: List[Article] = []
    seen_urls = set()

    for i, source in enumerate(rss_sources):
        try:
            # Add delay between requests to avoid rate limiting
            if i > 0:
                time.sleep(1.5)  # 1.5 second delay between RSS requests
            
            resp = requests.get(source, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            feed = feedparser.parse(resp.content)
            
            if not feed.entries:
                continue
            
            for entry in feed.entries[:10]:
                published = None
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    published = datetime.datetime.fromtimestamp(time.mktime(entry.published_parsed), tz=datetime.timezone.utc)
                    if published < cutoff:
                        continue

                title_lower = getattr(entry, "title", "").lower()
                summary_lower = getattr(entry, "summary", "").lower()
                
                # More flexible keyword matching with synonyms and related terms
                matching_kw = None
                for kw in keywords:
                    kw_lower = kw.lower()
                    
                    # Define keyword synonyms and related terms
                    synonyms = {
                        'AI': ['ai', 'artificial intelligence', 'machine learning', 'ml', 'neural network', 'deep learning', 'llm', 'gpt', 'chatbot', 'automation'],
                        'strategy': ['strategy', 'strategic', 'planning', 'roadmap', 'vision', 'transformation', 'consulting', 'advisory'],
                        'consulting': ['consulting', 'consultant', 'advisory', 'adviser', 'expertise', 'mckinsey', 'bcg', 'bain', 'deloitte', 'pwc'],
                        'business development': ['business development', 'growth', 'expansion', 'scaling', 'revenue', 'market', 'business'],
                        'CRM': ['crm', 'customer relationship', 'salesforce', 'hubspot', 'customer management'],
                        'leadership': ['leadership', 'executive', 'ceo', 'cfo', 'cto', 'director', 'management'],
                        'innovation': ['innovation', 'innovative', 'breakthrough', 'disruptive', 'cutting-edge'],
                        'automation': ['automation', 'automated', 'robotic', 'workflow', 'process'],
                        'data analytics': ['data analytics', 'analytics', 'data science', 'big data', 'insights', 'metrics'],
                        'cloud computing': ['cloud', 'aws', 'azure', 'gcp', 'saas', 'infrastructure'],
                        'cybersecurity': ['cybersecurity', 'security', 'cyber', 'breach', 'hack', 'privacy'],
                        'fintech': ['fintech', 'financial technology', 'payments', 'banking', 'fintech'],
                        'martech': ['martech', 'marketing technology', 'adtech', 'marketing automation'],
                        'saas': ['saas', 'software as a service', 'subscription', 'platform'],
                        'digital transformation': ['digital transformation', 'digital', 'transformation', 'digitization', 'digitalization'],
                        'sustainability': ['sustainability', 'sustainable', 'esg', 'environmental', 'green', 'climate'],
                        'regulation': ['regulation', 'regulatory', 'compliance', 'policy', 'governance']
                    }
                    
                    # Check if keyword or its synonyms match
                    terms_to_check = [kw_lower] + synonyms.get(kw_lower, [])
                    for term in terms_to_check:
                        if len(term) <= 3:  # Short terms use word boundaries
                            if re.search(r'\b' + re.escape(term) + r'\b', title_lower) or re.search(r'\b' + re.escape(term) + r'\b', summary_lower):
                                matching_kw = kw
                                break
                        else:  # Longer terms use substring matching
                            if term in title_lower or term in summary_lower:
                                matching_kw = kw
                                break
                    if matching_kw:
                        break
                
                if not matching_kw:
                    continue

                url = resolve_url(getattr(entry, "link", ""))
                canon = canonicalize(url)
                if canon in seen_urls:
                    continue
                seen_urls.add(canon)

                hostname = (urlparse(url).hostname or "").lower()
                
                # Filter by trusted domains - require high-authority sources
                if not any(hostname == d or hostname.endswith("." + d) for d in TRUSTED_DOMAINS):
                    continue
                
                articles.append(Article(
                    keyword=matching_kw,
                    title=getattr(entry, "title", "(no title)").strip(),
                    link=url,
                    published=published,
                    hostname=hostname,
                    source_hint=source,
                    summary=getattr(entry, "summary", "")
                ))
                if len(articles) >= max_articles:
                    break
                    
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                time.sleep(5)  # Wait 5 seconds on rate limit
                continue
            continue
        except Exception:
            continue

    print(f"[fetch_articles] Found {len(articles)} relevant articles")
    return articles[:max_articles]

# ---------------- Extract + Summarize ----------------
SYSTEM_PROMPT = """
You are an editor writing a concise newsletter for management consultants.
Return JSON only with fields: title, summary (2 sentences, 30-45 words), why_it_matters (1 sentence <=20 words), suggested_action (<=10 words), tags (list up to 3).
"""
EXAMPLE_JSON = {
    "title": "Google fined €2.95bn by EU for adtech dominance",
    "summary": "The EU fined Google €2.95bn for favouring its adtech products over rivals, intensifying antitrust scrutiny in digital advertising.",
    "why_it_matters": "Raises compliance and vendor risk for clients using Google ad stacks.",
    "suggested_action": "Audit adstack vendor dependency",
    "tags": ["Regulation","MarTech","Europe"]
}

def extract_text(url: str, max_chars: int = 3500) -> str:
    try:
        # Add small delay to avoid overwhelming servers
        time.sleep(0.5)
        r = requests.get(url, timeout=10, headers=HEADERS)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        paragraphs = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]
        return " ".join(paragraphs)[:max_chars]
    except Exception:
        return ""

def call_structured_summary(article_text: str, url: str):
    try:
        prompt = f"Example JSON:\n{json.dumps(EXAMPLE_JSON)}\n\nArticle URL: {url}\n\nArticle text:\n{article_text}\n\nReturn only JSON."
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role":"user","content":prompt}],
            max_tokens=400,
            temperature=0.0
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```", 2)[-1].strip()
        return json.loads(raw)
    except Exception:
        return None

def enrich_with_summaries(articles: List[Article], force_refresh: bool = False):
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)
    except Exception:
        cache = {}
    changed = False

    for art in articles:
        canon = canonicalize(art.link)
        if canon in cache and not force_refresh:
            data = cache[canon]
            art.title = data.get("title", art.title)
            art.summary = data.get("summary")
            art.why_it_matters = data.get("why_it_matters")
            art.suggested_action = data.get("suggested_action")
            art.tags = data.get("tags", [])
            continue

        text = extract_text(art.link)
        if not text:
            art.summary = art.summary or "No content available."
            continue

        structured = call_structured_summary(text, art.link)
        if structured:
            art.title = structured.get("title") or art.title
            art.summary = structured.get("summary")
            art.why_it_matters = structured.get("why_it_matters")
            art.suggested_action = structured.get("suggested_action")
            art.tags = structured.get("tags", [])
            cache[canon] = {
                "title": art.title,
                "summary": art.summary,
                "why_it_matters": art.why_it_matters,
                "suggested_action": art.suggested_action,
                "tags": art.tags,
                "fetched_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
            changed = True
        else:
            art.summary = text[:300] + ("…" if len(text) > 300 else "")

    if changed:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)

# ---------------- Scoring + TLDR ----------------
def compute_score(a: Article, keywords: List[str]) -> float:
    if a.published:
        days = (datetime.datetime.now(datetime.timezone.utc) - a.published).days
        recency = max(0.0, 1.0 - days / 30.0)
    else:
        recency = 0.5
    kw_rel = 1.0 if any(k.lower() in a.title.lower() for k in keywords) else 0.6
    return recency * kw_rel

def score_and_trim(articles: List[Article], keywords: List[str], limit: int = MAX_SUMMARIES_PER_RUN) -> List[Article]:
    for art in articles:
        art.score = compute_score(art, keywords)
    return sorted(articles, key=lambda x: x.score, reverse=True)[:limit]

def generate_tldr(top_articles: List[Article]):
    if not top_articles:
        return [], "Key developments for clients."
    concat = "\n\n".join([f"{i+1}. {a.title} — {a.summary}" for i,a in enumerate(top_articles[:6])])
    prompt = (
        "Produce JSON with: bullets (3 short bullets 10-18 words each), why_it_matters (one sentence <=20 words).\n\n"
        f"Input:\n{concat}\n\nReturn only JSON."
    )
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            max_tokens=200,
            temperature=0.0
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```", 2)[-1].strip()
        data = json.loads(raw)
        return data.get("bullets", []), data.get("why_it_matters", "")
    except Exception as e:
        return [a.title for a in top_articles[:3]], "Key developments for clients."

# ---------------- Synthesis Functions ----------------
def convert_markdown_to_html(text: str) -> str:
    """Convert basic markdown formatting to HTML"""
    # Remove any markdown code blocks first
    text = re.sub(r'```html\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    
    # Convert markdown headers to HTML
    text = re.sub(r'^### (.*?)$', r'<h3 style="color:#333; border-bottom:2px solid #007bff; padding-bottom:5px; margin-top:20px;">\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.*?)$', r'<h2 style="color:#333; margin-top:25px; margin-bottom:15px;">\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^# (.*?)$', r'<h1 style="color:#333; margin-top:30px; margin-bottom:20px;">\1</h1>', text, flags=re.MULTILINE)
    
    # Convert bold text
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    
    # Convert bullet points
    text = re.sub(r'^- (.*?)$', r'<li style="margin-bottom:8px;">\1</li>', text, flags=re.MULTILINE)
    
    # Wrap consecutive list items in ul tags
    text = re.sub(r'(<li style="margin-bottom:8px;">.*?</li>(?:\s*<li style="margin-bottom:8px;">.*?</li>)*)', 
                  r'<ul style="margin:10px 0; padding-left:20px;">\1</ul>', text, flags=re.DOTALL)
    
    # Convert line breaks to HTML breaks
    text = text.replace('\n', '<br>')
    
    return text

def synthesize_insights_with_articles(agent_insights: str, articles: List[Article]) -> str:
    """Create a comprehensive synthesis of agent insights and supporting articles"""
    if not agent_insights or not articles:
        return "Unable to synthesize insights with current data."
    
    try:
        # Get current date for context
        current_date = datetime.date.today().strftime("%B %Y")
        
        # Create a comprehensive synthesis prompt
        articles_text = "\n\n".join([
            f"Article {i+1}: {a.title}\nSummary: {a.summary}\nWhy it matters: {a.why_it_matters}\nSuggested action: {a.suggested_action}\nSource: {a.hostname}\nDate: {a.published.strftime('%Y-%m-%d') if a.published else 'Unknown'}"
            for i, a in enumerate(articles[:8])
        ])
        
        synthesis_prompt = f"""
        As a Differ consultant, create a comprehensive monthly market brief for {current_date} that synthesizes strategic insights with supporting evidence from recent news articles.
        
        AGENT INSIGHTS:
        {agent_insights}
        
        SUPPORTING ARTICLES:
        {articles_text}
        
        Create a SINGLE, COHERENT report that:
        1. Starts with a compelling executive summary (2-3 sentences)
        2. Organizes insights into 3-4 key themes with concrete evidence
        3. For each theme, provides:
           - Strategic context from the agent insights
           - Supporting evidence from relevant articles
           - Specific action items for clients
        4. Concludes with immediate next steps
        
        Write in a professional, consultative tone. Avoid generic buzzwords. Focus on concrete developments and actionable insights.
        Use specific company names, dates, and outcomes when available.
        
        IMPORTANT: Write in plain text with simple formatting. Do NOT use HTML tags, markdown code blocks, or any special formatting. Just write clear, well-structured content that can be easily converted to HTML later.
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a senior consultant at Differ writing a comprehensive market brief. Be specific, actionable, and evidence-based. Write in plain text with clear structure."},
                {"role": "user", "content": synthesis_prompt}
            ],
            max_tokens=1500,
            temperature=0.3
        )
        
        content = response.choices[0].message.content.strip()
        
        # Convert any remaining markdown to HTML
        return convert_markdown_to_html(content)
        
    except Exception as e:
        print(f"Error synthesizing insights: {e}")
        return "Unable to synthesize insights with current data."

# ---------------- Build HTML ----------------
def build_html(articles: List[Article], subject: str, keywords: List[str], agent_insights: str = None) -> str:
    prioritized = score_and_trim(articles, keywords)
    
    # Create comprehensive synthesis
    if agent_insights and prioritized:
        synthesized_content = synthesize_insights_with_articles(agent_insights, prioritized)
    else:
        synthesized_content = "Unable to generate comprehensive analysis with current data."

    parts = [f"<h1>Differ — Monthly Market Brief: {subject}</h1>",
             f"<div style='color:#666; margin-bottom:20px;'>{datetime.date.today().strftime('%B %Y')}</div>"]
    
    # Add comprehensive synthesis
    parts.append("<div style='background:#ffffff; padding:20px; margin:20px 0; border:1px solid #e0e0e0; border-radius:8px;'>")
    parts.append("<div style='font-family: Arial, sans-serif; line-height:1.6; color:#333;'>")
    parts.append(synthesized_content)
    parts.append("</div>")
    parts.append("</div>")

    # Add supporting evidence section
    if prioritized:
        parts.append("<h2>📚 Supporting Evidence</h2>")
        parts.append("<div style='background:#f8f9fa; padding:15px; border-radius:6px;'>")
        parts.append("<p style='margin:0 0 15px 0; color:#666; font-size:14px;'>Key articles that support the analysis above:</p>")
        
        for i, a in enumerate(prioritized[:6], 1):
            pub = a.published.strftime("%Y-%m-%d") if a.published else "Unknown"
            parts.append(f"<div style='margin-bottom:15px; padding:10px; background:white; border-radius:4px; border-left:3px solid #007bff;'>")
            parts.append(f"<div style='font-weight:600; margin-bottom:5px;'>")
            parts.append(f"<a href='{a.link}' style='color:#007bff; text-decoration:none;'>{a.title}</a>")
            parts.append(f"</div>")
            parts.append(f"<div style='color:#555; font-size:14px; margin-bottom:8px;'>{a.summary}</div>")
            parts.append(f"<div style='color:#888; font-size:12px;'>")
            parts.append(f"<strong>Source:</strong> {a.hostname} • <strong>Date:</strong> {pub} • <strong>Action:</strong> {a.suggested_action}")
            parts.append(f"</div></div>")
        
        parts.append("</div>")

    parts.append("<div style='font-size:12px;color:#777; margin-top:30px; padding-top:15px; border-top:1px solid #eee;'>")
    parts.append("<strong>Methodology:</strong> Comprehensive market analysis powered by Differ's AI agent, synthesized with curated RSS sources and GPT analysis.<br>")
    parts.append("<strong>Contact:</strong> marknadskommitten@differ.se | <strong>Differ:</strong> info@differ.se | +46 8 519 510 00")
    parts.append("</div>")
    
    return "\n".join(parts)

def write_output_html(html: str, subject: str) -> str:
    from pathlib import Path
    base_dir = Path(__file__).resolve().parent
    out_dir = base_dir / "Output"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_subject = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in subject)[:60]
    filename = f"{timestamp}_{safe_subject or 'Monthly_Digest'}.html"
    out_path = out_dir / filename
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return str(out_path)

# ---------------- MAIN ----------------
def main(subject: str, keywords: List[str], agent_insights: str = None):
    print(f"[main] Generating monthly brief with {len(keywords)} keywords")
    print(f"[main] Keywords: {keywords[:10]}...")  # Show first 10 keywords
    
    article_candidates = fetch_articles(keywords)
    if not article_candidates:
        print("No articles found - aborting")
        return

    print(f"[main] Found {len(article_candidates)} articles, enriching with summaries...")
    enrich_with_summaries(article_candidates)
    
    print(f"[main] Building HTML report...")
    html = build_html(article_candidates, subject, keywords, agent_insights)
    
    subject_line = f"Differ — Monthly Market Brief: {subject} ({datetime.date.today().strftime('%B %Y')})"
    output_path = write_output_html(html, subject_line)
    print(f"Digest generated successfully: {output_path}")
    return output_path

# ---------------- RUN ----------------
if __name__ == "__main__":
    print("🤖 Querying Differ agent for market insights...")
    agent_insights = query_differ_agent_for_insights()
    
    if agent_insights:
        print("✅ Differ agent insights received")
        print("📝 Extracting keywords from agent insights...")
        keywords = extract_keywords_from_agent_insights(agent_insights)
    else:
        print("⚠️  Agent query failed, using fallback keywords")
        keywords = [
            "Artificial Intelligence", "AI", "Digital Transformation", "Strategy",
            "Sustainability", "MarTech", "Customer Experience", "Innovation",
            "Cybersecurity", "Cloud Computing", "Data Analytics", "Automation"
        ]
    
    print(f"🎯 Using {len(keywords)} keywords: {keywords[:8]}...")
    
    subject = "Strategic market developments"
    main(subject, keywords, agent_insights)

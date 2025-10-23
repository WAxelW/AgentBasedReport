#!/usr/bin/env python3
"""
Differ Monthly Newsletter Generator (Python 3)

- Fetches articles from curated RSS sources filtered by keywords/themes
- Deduplicates URLs & canonicalizes links
- Summarizes with GPT in structured JSON format
- Builds HTML digest
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
        "https://feeds.hbr.org/HarvardBusiness",
        "https://feeds.feedburner.com/forbes/",
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
                        'strategy': ['strategy', 'strategic', 'planning', 'roadmap', 'vision', 'transformation'],
                        'consulting': ['consulting', 'consultant', 'advisory', 'adviser', 'expertise'],
                        'business development': ['business development', 'growth', 'expansion', 'scaling', 'revenue'],
                        'CRM': ['crm', 'customer relationship', 'salesforce', 'hubspot', 'customer management'],
                        'innovation': ['innovation', 'innovative', 'breakthrough', 'disruptive', 'cutting-edge'],
                        'automation': ['automation', 'automated', 'robotic', 'workflow', 'process'],
                        'data analytics': ['data analytics', 'analytics', 'data science', 'big data', 'insights', 'metrics'],
                        'cloud computing': ['cloud', 'aws', 'azure', 'gcp', 'saas', 'infrastructure'],
                        'cybersecurity': ['cybersecurity', 'security', 'cyber', 'breach', 'hack', 'privacy'],
                        'fintech': ['fintech', 'financial technology', 'payments', 'banking', 'fintech'],
                        'martech': ['martech', 'marketing technology', 'adtech', 'marketing automation'],
                        'saas': ['saas', 'software as a service', 'subscription', 'platform']
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

# ---------------- Build HTML ----------------
def build_html(articles: List[Article], subject: str, keywords: List[str]) -> str:
    prioritized = score_and_trim(articles, keywords)
    top3 = prioritized[:3]
    bullets, why = generate_tldr(prioritized[:6])

    grouped = {}
    for a in prioritized:
        grouped.setdefault(a.keyword, []).append(a)

    parts = [f"<h1>Differ — Monthly Brief: {subject}</h1>",
             f"<div style='color:#666'>{datetime.date.today().strftime('%B %Y')}</div>",
             "<h2>TL;DR</h2><ul>"]
    for b in bullets:
        parts.append(f"<li>{b}</li>")
    parts.append("</ul>")
    parts.append(f"<div><strong>Why it matters:</strong> {why}</div>")

    parts.append("<h2>Top 3</h2>")
    for a in top3:
        pub = a.published.strftime("%Y-%m-%d") if a.published else ""
        parts.append(f"<div style='border:1px solid #eee; padding:8px; margin:8px 0;'>"
                     f"<a href='{a.link}' style='font-weight:600'>{a.title}</a>"
                     f"<div>{a.summary}</div>"
                     f"<div style='color:#888; font-size:12px'>{a.hostname} • {pub} • Why: {a.why_it_matters} • <em>{a.suggested_action}</em></div>"
                     f"</div>")

    parts.append("<h2>By theme</h2>")
    for kw, arr in grouped.items():
        parts.append(f"<h3>{kw}</h3><ul>")
        for a in arr[:MAX_PER_THEME]:
            pub = a.published.strftime("%Y-%m-%d") if a.published else ""
            tags = ", ".join(a.tags or [])
            parts.append(f"<li><a href='{a.link}'>{a.title}</a> — {a.summary}"
                         f"<div style='color:#888; font-size:12px'>Source: {a.hostname} • {pub} • {tags}</div></li>")
        parts.append("</ul>")

    parts.append("<div style='font-size:12px;color:#777'>Method: curated RSS sources, summarized with GPT. Reply marknadskommitten@differ.se</div>")
    return "\n".join(parts)

def write_output_html(html: str, subject: str) -> str:
    import os
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
def main(subject: str, keywords: List[str]):
    article_candidates = fetch_articles(keywords)
    if not article_candidates:
        print("No articles found - aborting")
        return

    enrich_with_summaries(article_candidates)
    html = build_html(article_candidates, subject, keywords)
    subject_line = f"Differ — Monthly Brief: {subject} ({datetime.date.today().strftime('%B %Y')})"
    output_path = write_output_html(html, subject_line)
    print(f"Digest generated successfully: {output_path}")

# ---------------- RUN ----------------
if __name__ == "__main__":
    subject = "Management consulting adjacent topics"
    keywords = [
        "Artificial Intelligence", "AI", "Machine Learning", "Digital Transformation",
        "Strategy", "Consulting", "Business Development", "CRM", "Customer Relationship",
        "Leadership", "Innovation", "Automation", "Data Analytics",
        "Cloud Computing", "Cybersecurity", "Fintech", "MarTech", "SaaS"
    ]
    main(subject, keywords)

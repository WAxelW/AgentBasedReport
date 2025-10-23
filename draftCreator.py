import requests
from openai import OpenAI
import mailchimp_marketing as MailchimpMarketing
from mailchimp_marketing.api_client import ApiClientError
from datetime import datetime, timedelta

# --- CONFIG ---
MAILCHIMP_API_KEY = "914174703889684b6f268d80d12f2d91-us21"
MAILCHIMP_SERVER_PREFIX = "us21"  # e.g., "us21" from your API key
LIST_ID = "a38116c1cf"
TEMPLATE_ID = 10574664  # Mailchimp template you designed (must be int)
OPENAI_API_KEY = "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"

openai_client = OpenAI(api_key=OPENAI_API_KEY)
mc_client = MailchimpMarketing.Client()
mc_client.set_config({"api_key": MAILCHIMP_API_KEY, "server": MAILCHIMP_SERVER_PREFIX})
BASE_URL = f"https://{MAILCHIMP_SERVER_PREFIX}.api.mailchimp.com/3.0"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; DifferNewsDigest/1.0)"}

# --- STEP 1: Fetch articles (placeholder, replace with your scraping logic) ---
def fetch_articles():
    """
    Fetch articles from RSS feeds based on keywords/themes.
    Returns a list of relevant articles from trusted sources.
    """
    import feedparser
    import time
    from urllib.parse import quote
    
    # Keywords for tech, strategy, business development, CRM, and management
    keywords = [
        "artificial intelligence", "AI", "machine learning", "digital transformation",
        "strategy", "consulting", "business development", "CRM", "customer relationship",
        "management", "leadership", "innovation", "automation", "data analytics",
        "cloud computing", "cybersecurity", "fintech", "martech", "saas"
    ]
    
    # Trusted, high-authority RSS sources
    rss_sources = [
        # Tech & Business
        "https://feeds.feedburner.com/techcrunch/",
        "https://feeds.feedburner.com/venturebeat/SZYF",
        "https://feeds.feedburner.com/arstechnica/",
        "https://feeds.feedburner.com/wired/",
        "https://feeds.feedburner.com/theverge/",
        "https://feeds.feedburner.com/engadget/",
        
        # Business & Strategy
        "https://feeds.hbr.org/HarvardBusiness",
        "https://feeds.feedburner.com/mckinsey/",
        "https://feeds.feedburner.com/bcg/",
        "https://feeds.feedburner.com/bain/",
        "https://feeds.feedburner.com/deloitte/",
        "https://feeds.feedburner.com/pwc/",
        
        # European sources
        "https://feeds.bbci.co.uk/news/technology/rss.xml",
        "https://feeds.feedburner.com/techworld/",
        "https://www.di.se/rss/",
        
        # CRM & Business Software
        "https://feeds.feedburner.com/salesforce/",
        "https://feeds.feedburner.com/hubspot/",
        "https://feeds.feedburner.com/zendesk/",
    ]
    
    articles = []
    cutoff_date = datetime.now() - timedelta(days=30)
    
    for source in rss_sources:
        try:
            print(f"Fetching from: {source}")
            resp = requests.get(source, timeout=10, headers=HEADERS)
            resp.raise_for_status()
            feed = feedparser.parse(resp.content)
            
            for entry in feed.entries[:10]:  # Limit per source
                # Check if article has a publish date and is recent
                published = None
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    published = datetime.fromtimestamp(time.mktime(entry.published_parsed))
                    if published < cutoff_date:
                        continue
                
                # Check if article matches any keywords
                title_lower = entry.title.lower()
                summary_lower = (entry.summary if hasattr(entry, "summary") else "").lower()
                
                matching_keyword = None
                for kw in keywords:
                    if kw.lower() in title_lower or kw.lower() in summary_lower:
                        matching_keyword = kw
                        break
                
                if matching_keyword:
                    articles.append({
                        "title": entry.title,
                        "url": entry.link,
                        "summary": entry.summary if hasattr(entry, "summary") else "",
                        "date": published.strftime("%Y-%m-%d") if published else "Unknown",
                        "keyword": matching_keyword,
                        "source": source
                    })
                    
        except Exception as e:
            print(f"Error fetching from {source}: {e}")
            continue
    
    # Remove duplicates based on URL
    seen_urls = set()
    unique_articles = []
    for article in articles:
        if article["url"] not in seen_urls:
            seen_urls.add(article["url"])
            unique_articles.append(article)
    
    print(f"Found {len(unique_articles)} relevant articles")
    return unique_articles[:20]  # Return top 20 articles


# --- STEP 2: Summarize into newsletter content ---
def generate_newsletter_content(articles, subject):
    summaries = "\n\n".join([f"- {a['title']} ({a['date']})\n{a['summary']}\nRead more: {a['url']}" for a in articles])
    
    prompt = f"""
    You are preparing a professional monthly newsletter for management consultants.
    Subject: {subject}
    Summarize the following articles into a concise, engaging draft with:
    - A short intro
    - A TL;DR section
    - Top 3 highlights
    - Themed summaries
    Keep it professional, concise, and relevant for Differ Strategy colleagues.

    Articles:
    {summaries}
    """

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )

    return response.choices[0].message.content.strip()

# --- STEP 3: Create campaign draft in Mailchimp ---
def create_campaign(subject, content):
    try:
        campaign = mc_client.campaigns.create({
            "type": "regular",
            "recipients": {"list_id": LIST_ID},
            "settings": {
                "subject_line": subject,
                "title": f"Differ Monthly Brief — {datetime.now().strftime('%B %Y')}",
                "from_name": "Marknadskommittén",
                "reply_to": "marknadskommitten@differ.se",
                # no template; we'll send raw HTML content
            }
        })
        campaign_id = campaign["id"]

        # Build simple HTML body
        simple_html = (
            f"<h1>{subject}</h1>"
            f"<div style='font-family:system-ui,Segoe UI,Arial,sans-serif;white-space:pre-wrap;'>"
            f"{content}"
            f"</div>"
        )
        mc_client.campaigns.set_content(campaign_id, {"html": simple_html})

        return campaign_id
    except ApiClientError as e:
        status = getattr(e, 'status_code', None)
        raw = getattr(e, 'text', None)
        if raw is None:
            raw = str(e)
        elif not isinstance(raw, str):
            try:
                import json as _json
                raw = _json.dumps(raw)
            except Exception:
                raw = str(raw)
        print(f"Mailchimp API error ({status}): {raw[:500]}")
        raise

# --- STEP 4: Main workflow ---
if __name__ == "__main__":
    subject = "Artificial Intelligence & Martech — Monthly Brief"
    articles = fetch_articles()
    newsletter_content = generate_newsletter_content(articles, subject)

    campaign_id = create_campaign(subject, newsletter_content)

    campaign_url = f"https://admin.mailchimp.com/campaigns/show/?id={campaign_id}"
    print("✅ Draft campaign created!")
    print(f"Review & send here: {campaign_url}")

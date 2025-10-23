import feedparser
import datetime
import time
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from urllib.parse import quote_plus, urlparse
import mailchimp_marketing as MailchimpMarketing
from mailchimp_marketing.api_client import ApiClientError


# ---------------- CONFIG ----------------
MAILCHIMP_API_KEY = "914174703889684b6f268d80d12f2d91-us21"
MAILCHIMP_SERVER = "us21"
LIST_ID = "a38116c1cf"
SEGMENT_NAME = "Differ internt DL ALL"
OPENAI_API_KEY = "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"

# OpenAI client (separate from Mailchimp client)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Max articles per keyword per month
MAX_ARTICLES = 10
# ----------------------------------------


# ---------- SETUP ----------
mc_client = MailchimpMarketing.Client()
mc_client.set_config({"api_key": MAILCHIMP_API_KEY, "server": MAILCHIMP_SERVER})


# ---------- STEP 1: Fetch news ----------
def fetch_news(keywords, max_articles=5):
    articles = []
    cutoff_date = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=30)

    # Multiple reliable RSS sources (Sweden/EU/US/Asia)
    rss_sources = [
        # Sweden
        "https://feeds.feedburner.com/techworld/",
        "https://www.di.se/rss/",
        "https://www.svd.se/rss.xml",

        # Europe / UK
        "https://feeds.bbci.co.uk/news/technology/rss.xml",
        "https://feeds.feedburner.com/techcrunch/",
        "https://feeds.feedburner.com/venturebeat/SZYF",
        "https://feeds.feedburner.com/oreilly/radar",
        "https://feeds.feedburner.com/arstechnica/",

        # USA
        "https://feeds.feedburner.com/wired/",
        "https://feeds.feedburner.com/theverge/",
        "https://feeds.feedburner.com/engadget/",
        "https://feeds.feedburner.com/mashable/",

        # Asia
        "https://feeds.feedburner.com/techinasia/",
    ]

    headers = {"User-Agent": "Mozilla/5.0 (compatible; NewsDigestBot/1.0)"}

    # Whitelist of trustworthy domains
    allowed_domains = {
        "bbc.co.uk", "bbc.com",
        "di.se", "svd.se",
        "techcrunch.com", "venturebeat.com",
        "wired.com", "theverge.com",
        "engadget.com", "mashable.com",
        "techinasia.com", "arstechnica.com",
    }

    # Track duplicates and overall cap
    seen_links = set()
    total_count = 0

    for source in rss_sources:
        print(f"Fetching: {source}")
        try:
            response = requests.get(source, timeout=10, headers=headers)
            response.raise_for_status()

            feed = feedparser.parse(response.content)
            print(f"Feed entries: {len(feed.entries)}")

            per_source_count = 0
            for entry in feed.entries:
                if total_count >= max_articles:
                    break
                # Published date filtering
                published = None
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    published = datetime.datetime.fromtimestamp(
                        time.mktime(entry.published_parsed), tz=datetime.timezone.utc
                    )
                    if published < cutoff_date:
                        continue

                # Keyword match in title or summary
                title_lower = getattr(entry, "title", "").lower()
                summary_lower = getattr(entry, "summary", "").lower()
                matched_kw = None
                for kw in keywords:
                    if kw.lower() in title_lower or kw.lower() in summary_lower:
                        matched_kw = kw
                        break

                if not matched_kw:
                    continue

                # Trustworthy domain filter
                link = getattr(entry, "link", "")
                hostname = urlparse(link).hostname or ""
                hostname = hostname.lower()
                is_trusted = any(hostname == d or hostname.endswith("." + d) for d in allowed_domains)
                if not is_trusted:
                    continue

                # Deduplicate by link
                if link in seen_links:
                    continue

                articles.append({
                    "keyword": matched_kw,
                    "title": getattr(entry, "title", "(no title)"),
                    "link": link,
                    "published": published,
                    "summary": getattr(entry, "summary", None)
                })

                per_source_count += 1
                total_count += 1
                if per_source_count >= max_articles:
                    break

        except Exception as e:
            print(f"Error fetching {source}: {e}")

    return articles


# ---------- STEP 2: Scrape + summarize ----------
def scrape_and_summarize(article_url, max_chars=1500):
    try:
        html = requests.get(article_url, timeout=10).text
        soup = BeautifulSoup(html, "html.parser")
        paragraphs = [p.get_text() for p in soup.find_all("p")]
        text = " ".join(paragraphs)[:max_chars]

        if not text:
            return "No content found."

        prompt = f"Summarize this news article in 2–3 sentences:\n\n{text}"
        resp = openai_client.chat.completions.create(model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=120)
        return resp.choices[0].message.content.strip()

    except Exception as e:
        return f"Error summarizing: {e}"


# ---------- STEP 3: Build HTML ----------
def build_html_report(articles):
    html = "<h1>Monthly News Digest</h1>"
    grouped = {}

    for art in articles:
        grouped.setdefault(art["keyword"], []).append(art)

    for kw, arts in grouped.items():
        html += f"<h2>{kw}</h2><ul>"
        for art in arts:
            pub = art["published"].strftime("%Y-%m-%d") if art["published"] else ""
            html += f"<li><a href='{art['link']}'><b>{art['title']}</b></a> ({pub})<br>"
            html += f"<p>{art['summary']}</p></li>"
        html += "</ul>"
    return html


# ---------- STEP 4: Find segment ID ----------
def get_segment_id_by_name(list_id, segment_name):
    try:
        response = mc_client.lists.list_segments(list_id)
        for seg in response["segments"]:
            if seg["name"] == segment_name:
                return seg["id"]
    except ApiClientError as error:
        print("Error fetching segments:", error.text)
    return None


# ---------- STEP 5: Send via Mailchimp ----------
def send_newsletter(html_content, subject="Monthly News Digest"):
    segment_id = get_segment_id_by_name(LIST_ID, SEGMENT_NAME)
    if not segment_id:
        print(f"Segment '{SEGMENT_NAME}' not found.")
        return

    try:
        # Create campaign
        campaign = mc_client.campaigns.create({
            "type": "regular",
            "recipients": {
                "list_id": LIST_ID,
                "segment_opts": {"saved_segment_id": segment_id}
            },
            "settings": {
                "subject_line": subject,
                "from_name": "Marknadskommittén",
                "reply_to": "marknadskommitten@differ.se"
            }
        })

        # Set content
        campaign_id = campaign["id"]
        mc_client.campaigns.set_content(campaign_id, {"html": html_content})

        # Send immediately
        mc_client.campaigns.send(campaign_id)
        print(f"✅ Campaign sent successfully to segment '{SEGMENT_NAME}'.")

    except ApiClientError as error:
        print("Error sending campaign:", error.text)


# ---------- MAIN ----------
if __name__ == "__main__":
    keywords = ["AI", "artificial intelligence", "martech", "automation"]  # Change keywords here
    articles = fetch_news(keywords, MAX_ARTICLES)

    for art in articles:
        art["summary"] = scrape_and_summarize(art["link"])

    report_html = build_html_report(articles)
    send_newsletter(report_html)

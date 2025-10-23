import feedparser
import datetime
import time
import requests
from bs4 import BeautifulSoup
import openai
from urllib.parse import quote

# Configure OpenAI (replace with your key)

# ---------- STEP 1: Fetch news ----------
def fetch_news(keywords, max_articles=5):
    articles = []
    cutoff_date = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=30)

    # Use multiple reliable RSS sources (Sweden, Europe, USA, Asia priority order)
    rss_sources = [
        # Sweden
        "https://feeds.feedburner.com/techworld/",
        "https://www.di.se/rss/",
        "https://www.svd.se/rss.xml",

        # Europe
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

    for source in rss_sources:
        print(f"Fetching: {source}")
        try:
            # Use requests to fetch the RSS content first
            response = requests.get(source, timeout=10)
            response.raise_for_status()

            # Parse with feedparser
            feed = feedparser.parse(response.content)
            print(f"Feed entries: {len(feed.entries)}")

            for entry in feed.entries[:max_articles]:  # Limit per source
                # Check if article has a publish date
                if hasattr(entry, "published_parsed"):
                    published = datetime.datetime.fromtimestamp(time.mktime(entry.published_parsed))
                    # Make timezone-aware for comparison
                    published = published.replace(tzinfo=datetime.timezone.utc)
                    if published < cutoff_date:
                        continue  # too old, skip

                # Check if article matches any of our keywords
                title_lower = entry.title.lower()
                summary_lower = (entry.summary if hasattr(entry, "summary") else "").lower()

                matching_keyword = None
                for kw in keywords:
                    if kw.lower() in title_lower or kw.lower() in summary_lower:
                        matching_keyword = kw
                        break

                if matching_keyword:
                    articles.append({
                        "keyword": matching_keyword,
                        "title": entry.title,
                        "link": entry.link,
                        "summary": entry.summary if hasattr(entry, "summary") else None,
                        "published": published if hasattr(entry, "published_parsed") else None
                    })

        except Exception as e:
            print(f"Error fetching {source}: {e}")

    return articles

# ---------- STEP 2: Extract + summarize ----------
def scrape_and_summarize(article_url, max_chars=1000):
    """
    Scrapes article text and asks GPT to summarize it.
    """
    try:
        html = requests.get(article_url, timeout=10).text
        soup = BeautifulSoup(html, "html.parser")

        # crude text extraction
        paragraphs = [p.get_text() for p in soup.find_all("p")]
        text = " ".join(paragraphs)[:max_chars]

        if not text:
            return "No content found."

        # summarize with OpenAI
        prompt = f"Summarize this news article in 2-3 sentences:\n\n{text}"
        client = openai.OpenAI(api_key=openai.api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",  # lightweight + cheap model
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120
        )
        return resp.choices[0].message.content.strip()

    except Exception as e:
        return f"Error summarizing: {e}"

# ---------- STEP 3: Format HTML digest ----------
def build_html_report(articles):
    """
    Turns articles into HTML string for Mailchimp.
    """
    html = "<h1>Monthly News Digest</h1>\n<ul>"
    for art in articles:
        html += f"<li><a href='{art['link']}'><b>{art['title']}</b></a><br>"
        html += f"<p>{art['summary']}</p></li>"
    html += "</ul>"
    return html

# ---------- MAIN ----------
if __name__ == "__main__":
    # 1. Keywords are variable – just change this list
    keywords = ["AI", "artificial intelligence", "climate", "startup", "tech"]

    # 2. Collect articles
    raw_articles = fetch_news(keywords)
    print(f"Found {len(raw_articles)} articles")

    # 3. Summarize each
    for art in raw_articles:
        art["summary"] = scrape_and_summarize(art["link"])

    # 4. Build digest
    report_html = build_html_report(raw_articles)

    # 5. Save HTML to file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"news_digest_{timestamp}.txt"

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(report_html)

    print(f"HTML digest saved to: {filename}")
    print(f"Preview (first 500 chars):")
    print(report_html[:500])

    # 6. Now you can push report_html into Mailchimp with campaigns.set_content()

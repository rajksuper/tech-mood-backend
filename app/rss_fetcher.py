import feedparser
from bs4 import BeautifulSoup
import requests
from datetime import datetime
import random
import re
from concurrent.futures import ThreadPoolExecutor


# -----------------------------------------
# CLEANERS
# -----------------------------------------

def clean_html(text):
    if not text:
        return ""
    # remove tags
    text = re.sub(r"<.*?>", "", text)
    # decode entities
    html_entities = {
        "&nbsp;": " ", "&amp;": "&", "&quot;": '"', "&#39;": "'",
        "&rsquo;": "'", "&lsquo;": "'", "&ndash;": "-", "&mdash;": "-"
    }
    for k, v in html_entities.items():
        text = text.replace(k, v)
    return text.strip()


def truncate(text, limit=350):
    """Trim very long summaries."""
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + "..."


# -----------------------------------------
# IMAGE EXTRACTORS
# -----------------------------------------

def extract_image(entry):
    """
    Extracts image from multiple RSS formats:
    - media:content
    - enclosure
    - embedded HTML
    - OpenGraph fallback
    """
    # 1. media:content
    if "media_content" in entry:
        for media in entry.media_content:
            if "url" in media:
                return media["url"]

    # 2. enclosure
    if "links" in entry:
        for link in entry.links:
            if link.get("rel") == "enclosure" and "image" in link.get("type", ""):
                return link.get("href")

    # 3. content HTML <img>
    if "content" in entry:
        for c in entry.content:
            soup = BeautifulSoup(c.value, "html.parser")
            img = soup.find("img")
            if img and img.get("src"):
                return img["src"]

    # 4. summary HTML <img>
    if "summary" in entry:
        soup = BeautifulSoup(entry.summary, "html.parser")
        img = soup.find("img")
        if img and img.get("src"):
            return img["src"]

    # 5. fetch OG:image from page (commented out for speed)
    # if "link" in entry:
    #     try:
    #         html = requests.get(entry.link, timeout=2).text
    #         soup = BeautifulSoup(html, "html.parser")
    #         og = soup.find("meta", property="og:image")
    #         if og and og.get("content"):
    #             return og["content"]
    #     except:
    #         pass

    return None  # no image found


# -----------------------------------------
# RSS FEED SOURCES
# -----------------------------------------

RSS_FEEDS = [
    # --- Hacker News / Dev Community ---
    "https://news.ycombinator.com/rss",
    "https://dev.to/feed",
    "https://stackoverflow.blog/feed/",
    "https://www.reddit.com/r/technology/.rss",
    
    # --- Pure Tech News ---
    "https://www.theverge.com/rss/index.xml",
    "https://feeds.arstechnica.com/arstechnica/index",
    "https://www.wired.com/feed/rss",
    "https://www.wired.com/feed/category/business/latest/rss",
    "https://www.techmeme.com/feed.xml",

    # --- AI / ML / Data ---
    "https://venturebeat.com/category/ai/feed/",
    "https://techcrunch.com/tag/artificial-intelligence/feed/",
    "https://ai.googleblog.com/feeds/posts/default",
    "https://openai.com/blog/rss/",
    "https://www.technologyreview.com/feed/",
    "https://www.anthropic.com/news/rss",
    "https://deepmind.google/blog/rss.xml",
    "https://bair.berkeley.edu/blog/feed.xml",
    "https://www.marktechpost.com/feed/",
    "https://syncedreview.com/feed/",
    "https://thegradient.pub/rss/",
    "https://distill.pub/rss.xml",
    "https://lilianweng.github.io/index.xml",
    "https://www.aiweirdness.com/rss/",

    # --- Startups / VC / Funding ---
    "https://techcrunch.com/startups/feed/",
    "https://techcrunch.com/feed/",
    "https://news.crunchbase.com/feed/",
    "https://sifted.eu/feed/",
    "https://www.ycombinator.com/blog/rss/",
    "https://a16z.com/feed/",
    "https://www.sequoiacap.com/feed/",
    "https://www.nfx.com/feed",
    "https://www.strictlyvc.com/feed/",
    "https://pitchbook.com/news/feed",
    "https://www.cbinsights.com/research/feed/",
    "https://www.eu-startups.com/feed/",
    "https://techfundingnews.com/feed/",
    "https://www.finsmes.com/feed",

    # --- Cybersecurity ---
    "https://krebsonsecurity.com/feed/",
    "https://feeds.feedburner.com/TheHackersNews",
    "https://www.bleepingcomputer.com/feed/",

    # --- Big Tech ---
    "https://www.macrumors.com/feed/",
    "https://blogs.microsoft.com/feed/",
    "https://blogs.nvidia.com/feed/",
    "https://electrek.co/feed/",

    # --- Business + Markets ---
    "https://www.reuters.com/business/feed/",
    "https://www.cnbc.com/id/10000664/device/rss/rss.html",
    "https://www.cnbc.com/id/19854910/device/rss/rss.html",
    "https://www.marketwatch.com/rss/marketpulse",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=XLK&region=US&lang=en-US",
    "https://seekingalpha.com/feed/sector/technology",
    "https://stratechery.com/feed/",
    "https://www.theinformation.com/feed",
    "https://www.bloomberg.com/technology/feed/",
    "https://www.ft.com/technology?format=rss",
    "https://feeds.feedburner.com/businessinsider",

    # --- Crypto / Web3 ---
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed",
    "https://bitcoinmagazine.com/.rss/full/",
    "https://thedefiant.io/feed",
    "https://www.theblock.co/rss.xml",
    "https://cryptoslate.com/feed/",
    "https://cryptopotato.com/feed/",
    
    # --- Stock Market / Tech Stocks ---
    "https://www.investors.com/feed/",
    "https://www.fool.com/feeds/index.aspx",
    "https://stocknews.com/feed/",
    "https://www.benzinga.com/feed",
    "https://finance.yahoo.com/rss/topstories",
    "https://www.investing.com/rss/news.rss",
]


# -----------------------------------------
# SINGLE FEED FETCHER
# -----------------------------------------

def fetch_single_feed(url):
    """Fetch a single RSS feed and return articles"""
    try:
        # Add timeout to prevent hanging on slow feeds
        import urllib.request
        urllib.request.urlopen(url, timeout=10)  # Quick check if URL is reachable
        
        feed = feedparser.parse(url)
        articles = []
        skipped = 0
        
        # Check if feed has entries
        if not feed.entries:
            print(f"⚠ No entries found: {url}")
            return []

        for entry in feed.entries:
            title_raw = entry.get("title", "")
            summary_raw = entry.get("summary", "") or entry.get("description", "")
            link = entry.get("link", "")
            
            # Skip if missing required fields
            if not title_raw or not link:
                skipped += 1
                continue
            
            # Clean the data
            title_clean = clean_html(title_raw).strip()
            summary_clean = truncate(clean_html(summary_raw)).strip()
            
            # Skip if title is too short (likely junk)
            if len(title_clean) < 10:
                skipped += 1
                continue

            article = {
                "title": title_clean,
                "summary": summary_clean,
                "source": feed.feed.get("title", "Unknown"),
                "source_url": link,
                "image_url": extract_image(entry),
                "published_at": entry.get("published", "") or entry.get("updated", "") or "",
            }

            articles.append(article)

        print(f"✓ Fetched {len(articles)} from {feed.feed.get('title', url)} (skipped {skipped})")
        return articles

    except Exception as e:
        print(f"✗ Feed error: {url} - {e}")
        return []


# -----------------------------------------
# MAIN FETCH FUNCTION (PARALLEL)
# -----------------------------------------

def fetch_articles():
    """Fetch all RSS feeds in parallel"""
    all_articles = []
    failed_feeds = []

    print(f"Fetching {len(RSS_FEEDS)} RSS feeds in parallel...")

    # Fetch feeds in parallel (10 workers at a time)
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(fetch_single_feed, RSS_FEEDS))
        
        for i, articles in enumerate(results):
            if articles:
                all_articles.extend(articles)
            else:
                failed_feeds.append(RSS_FEEDS[i])

    print(f"\n===== FETCH SUMMARY =====")
    print(f"Total articles fetched: {len(all_articles)}")
    print(f"Successful feeds: {len(RSS_FEEDS) - len(failed_feeds)}/{len(RSS_FEEDS)}")
    if failed_feeds:
        print(f"Failed feeds ({len(failed_feeds)}):")
        for feed in failed_feeds[:5]:  # Show first 5
            print(f"  - {feed}")
        if len(failed_feeds) > 5:
            print(f"  ... and {len(failed_feeds) - 5} more")
    print(f"=========================\n")

    # Mix all sources so they aren't grouped
    random.shuffle(all_articles)

    return all_articles
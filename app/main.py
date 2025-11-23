from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.rss_fetcher import fetch_articles
from app.sentiment import analyze_sentiment
from app.supabase_client import insert_article, supabase
from datetime import datetime, timedelta
import random
from collections import Counter
import re

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# RSS Feed to Category Mapping (6 core categories)
FEED_CATEGORIES = {
    # AI & ML
    "news.ycombinator.com": "AI & ML",
    "dev.to": "AI & ML",
    "venturebeat.com": "AI & ML",
    "techcrunch.com/tag/artificial-intelligence": "AI & ML",
    "ai.googleblog.com": "AI & ML",
    "openai.com": "AI & ML",
    "anthropic.com": "AI & ML",
    "huggingface.co": "AI & ML",
    "deepmind.google": "AI & ML",
    "bair.berkeley.edu": "AI & ML",
    "marktechpost.com": "AI & ML",
    "syncedreview.com": "AI & ML",
    "thegradient.pub": "AI & ML",
    "distill.pub": "AI & ML",
    "lilianweng.github.io": "AI & ML",
    "aiweirdness.com": "AI & ML",
    
    # Startups & VC
    "techcrunch.com/startups": "Startups & VC",
    "news.crunchbase.com": "Startups & VC",
    "sifted.eu": "Startups & VC",
    "ycombinator.com": "Startups & VC",
    "a16z.com": "Startups & VC",
    "sequoiacap.com": "Startups & VC",
    "nfx.com": "Startups & VC",
    "strictlyvc.com": "Startups & VC",
    "pitchbook.com": "Startups & VC",
    "cbinsights.com": "Startups & VC",
    "eu-startups.com": "Startups & VC",
    "techfundingnews.com": "Startups & VC",
    "finsmes.com": "Startups & VC",
    
    # Cybersecurity
    "krebsonsecurity.com": "Cybersecurity",
    "thehackernews.com": "Cybersecurity",
    "bleepingcomputer.com": "Cybersecurity",
    
    # Big Tech (includes hardware, chips, cloud, space)
    "macrumors.com": "Big Tech",
    "blogs.microsoft.com": "Big Tech",
    "blogs.nvidia.com": "Big Tech",
    "electrek.co": "Big Tech",
    
    # Markets & Finance (includes crypto)
    "reuters.com": "Markets & Finance",
    "cnbc.com": "Markets & Finance",
    "marketwatch.com": "Markets & Finance",
    "finance.yahoo.com": "Markets & Finance",
    "seekingalpha.com": "Markets & Finance",
    "stratechery.com": "Markets & Finance",
    "theinformation.com": "Markets & Finance",
    "bloomberg.com": "Markets & Finance",
    "ft.com": "Markets & Finance",
    "coindesk.com": "Markets & Finance",
    "cointelegraph.com": "Markets & Finance",
}


def get_category(source_url, title="", summary=""):
    """Determine category based on source URL and content keywords"""
    
    # First check exact feed mapping
    for domain, category in FEED_CATEGORIES.items():
        if domain in source_url:
            return category
    
    # Keyword-based categorization for unmapped feeds
    text = (title + " " + summary).lower()
    
    # Skip lifestyle/shopping content - goes to General Tech
    if any(word in text for word in ["black friday", "gift guide", "best deals", "cyber monday", "shopping", "sale"]):
        return "General Tech"
    
    # Cybersecurity
    if any(word in text for word in ["hack", "breach", "malware", "ransomware", "vulnerability", "security", "cyberattack"]):
        return "Cybersecurity"
    
    # AI & ML
    if any(word in text for word in ["ai", "artificial intelligence", "machine learning", "llm", "chatgpt", "openai", "neural", "model"]):
        return "AI & ML"
    
    # Startups & VC
    if any(word in text for word in ["funding", "raises", "series a", "series b", "venture capital", "startup", "acquisition", "ipo"]):
        return "Startups & VC"
    
    # Big Tech (includes hardware, chips, cloud, space)
    if any(word in text for word in ["apple", "google", "microsoft", "amazon", "meta", "tesla", "nvidia", "chip", "semiconductor", "aws", "azure", "cloud", "spacex", "satellite", "rocket"]):
        return "Big Tech"
    
    # Markets & Finance (includes crypto)
    if any(word in text for word in ["stock", "market", "trading", "investor", "finance", "economic", "crypto", "bitcoin", "ethereum", "blockchain"]):
        return "Markets & Finance"
    
    return "General Tech"


@app.get("/")
def home():
    return {"status": "backend running"}


@app.get("/run")
def run_pipeline():
    articles = fetch_articles()
    saved = []

    random.shuffle(articles)
    articles.sort(
        key=lambda x: (x["published_at"] or "1970", random.random()),
        reverse=True
    )

    # OPTIMIZATION: Batch check for existing URLs (in chunks of 100)
    print(f"Checking for duplicates in batches...")
    all_urls = [item["source_url"] for item in articles]
    
    existing_urls = set()
    batch_size = 100
    
    # Check URLs in batches to avoid URL too long error
    for i in range(0, len(all_urls), batch_size):
        batch = all_urls[i:i + batch_size]
        existing_result = (
            supabase.table("articles")
            .select("source_url")
            .in_("source_url", batch)
            .execute()
        )
        existing_urls.update(row["source_url"] for row in existing_result.data)
    
    print(f"Found {len(existing_urls)} existing articles")

    # Process only new articles
    for item in articles:
        # Skip if already exists
        if item["source_url"] in existing_urls:
            continue
        
        combined_text = f"{item['title']} {item['summary']}"
        label, score, raw = analyze_sentiment(combined_text)

        record = {
            "title": item["title"],
            "summary": item["summary"],
            "source": item["source"],
            "source_url": item["source_url"],
            "sentiment_label": label,
            "sentiment_score": raw['score'],
            "raw_model_output": raw,
            "created_at": datetime.utcnow().isoformat(),
            "image_url": item.get("image_url"),
            "published_at": item.get("published_at"),
            "category": get_category(item["source_url"], item["title"], item["summary"]),
        }

        insert_article(record)
        saved.append(record)
        print(f"✓ Inserted: {item['title'][:50]}...")

    print(f"Done! Inserted {len(saved)} new articles")
    return {"inserted": len(saved), "skipped": len(articles) - len(saved)}


# Get all categories
@app.get("/categories")
def get_categories():
    result = supabase.table("articles").select("category").execute()
    categories = list(set(row["category"] for row in result.data if row.get("category")))
    return {"categories": sorted(categories)}


# Get trending topics/keywords
@app.get("/trending")
def get_trending():
    # Get articles from last 24 hours
    yesterday = (datetime.utcnow() - timedelta(days=1)).isoformat()
    
    recent = (
        supabase.table("articles")
        .select("title")
        .gte("created_at", yesterday)
        .execute()
    )
    
    # Extract keywords from titles
    keywords = []
    
    # MASSIVE stop words list
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "from", 
        "is", "are", "was", "were", "be", "been", "has", "have", "had", "will", "can", "could", "may", 
        "might", "about", "as", "its", "it", "this", "that", "these", "those", "than", "then", "there", 
        "their", "they", "them", "we", "us", "our", "your", "you", "my", "me", "he", "him", "his", "she", 
        "her", "hers", "all", "any", "some", "few", "more", "most", "other", "such", "no", "not", "only", 
        "own", "same", "so", "up", "out", "just", "now", "how", "what", "when", "where", "who", "why", 
        "which", "get", "got", "make", "made", "take", "new", "first", "last", "long", "good", "bad", 
        "old", "great", "little", "big", "high", "low", "off", "over", "under", "after", "before", "says",
        "said", "vs", "into", "onto", "via", "per", "like", "goes", "going", "go", "does", "do", "did",
        "one", "two", "three", "use", "using", "uses", "used", "way", "back", "down", "see", "seen",
        "top", "best", "better", "want", "wants", "wanted", "need", "needs", "needed", "also", "even",
        "well", "much", "very", "too", "still", "come", "comes", "day", "days", "year", "years", "time",
        "week", "weeks", "month", "months", "around", "through", "during", "since", "while", "both",
        "each", "every", "between", "against", "within", "without", "being", "become", "becomes",
        "should", "would", "must", "let", "lets", "find", "finds", "found", "look", "looks", "looking",
        "try", "tries", "trying", "tried", "think", "thinks", "thought", "know", "knows", "knew", "known"
    }
    
    # Tech-related terms to PRIORITIZE (whitelist)
    tech_terms = {
        "openai", "anthropic", "google", "microsoft", "apple", "meta", "amazon", "nvidia", "tesla",
        "spacex", "uber", "airbnb", "stripe", "shopify", "salesforce", "oracle", "ibm", "intel",
        "amd", "qualcomm", "samsung", "sony", "huawei", "alibaba", "tencent", "baidu", "bytedance",
        "chatgpt", "gemini", "claude", "grok", "copilot", "midjourney", "dalle", "stable",
        "bitcoin", "ethereum", "crypto", "blockchain", "web3", "nft", "defi", "ai", "machine",
        "learning", "llm", "model", "neural", "algorithm", "data", "cloud", "quantum", "robotics",
        "autonomous", "electric", "vehicle", "startup", "funding", "series", "vc", "investment",
        "acquisition", "ipo", "layoffs", "hiring", "ceo", "cybersecurity", "breach", "hack", "privacy",
        "iphone", "android", "ios", "windows", "linux", "app", "software", "hardware", "chip",
        "semiconductor", "processor", "gpu", "api", "saas", "platform", "launch", "release",
        "update", "upgrade", "feature", "product", "service", "technology", "tech", "digital"
    }
    
    for article in recent.data:
        title = article["title"].lower()
        # Extract words (3+ chars, alphanumeric)
        words = re.findall(r'\b[a-z]{3,}\b', title)
        
        # Keep words that are either tech terms OR not in stop words
        for word in words:
            if word in tech_terms:
                keywords.append(word)  # Always include tech terms
            elif word not in stop_words and len(word) >= 4:  # Other words must be 4+ chars
                keywords.append(word)
    
    # Count frequency
    counter = Counter(keywords)
    
    # Prioritize tech terms in ranking
    ranked = []
    for word, count in counter.most_common(50):
        if word in tech_terms:
            count = count * 2  # Boost tech terms
        ranked.append((word, count))
    
    # Re-sort by boosted count
    ranked.sort(key=lambda x: x[1], reverse=True)
    
    # Filter: only show keywords that appear 3+ times (original count)
    top_keywords = [(k, counter[k]) for k, c in ranked[:20] if counter[k] >= 3]
    
    return {"trending": [{"keyword": k.capitalize(), "count": c} for k, c in top_keywords[:10]]}


# Homepage - Returns 12 with images + 12 without images (with optional category filter)
@app.get("/articles")
def get_articles(category: str = None):
    # Build base query
    query_with = supabase.table("articles").select("*").not_.is_("image_url", "null").neq("image_url", "")
    query_without = supabase.table("articles").select("*").or_("image_url.is.null,image_url.eq.")
    
    # Add category filter if provided
    if category:
        query_with = query_with.eq("category", category)
        query_without = query_without.eq("category", category)
    
    # Execute queries
    with_images = query_with.order("id", desc=True).limit(12).execute()
    without_images = query_without.order("id", desc=True).limit(12).execute()
    
    return {
        "with_images": with_images.data,
        "without_images": without_images.data
    }


# Pagination - Returns next batch of 12 with images + 12 without images (with optional category filter)
@app.get("/articles/page/{page_num}")
def get_articles_paginated(page_num: int, category: str = None):
    offset = page_num * 12
    
    # Build base query
    query_with = supabase.table("articles").select("*").not_.is_("image_url", "null").neq("image_url", "")
    query_without = supabase.table("articles").select("*").or_("image_url.is.null,image_url.eq.")
    
    # Add category filter if provided
    if category:
        query_with = query_with.eq("category", category)
        query_without = query_without.eq("category", category)
    
    # Execute queries
    with_images = query_with.order("id", desc=True).range(offset, offset + 11).execute()
    without_images = query_without.order("id", desc=True).range(offset, offset + 11).execute()
    
    return {
        "with_images": with_images.data,
        "without_images": without_images.data
    }
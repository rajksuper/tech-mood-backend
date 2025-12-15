from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.rss_fetcher import fetch_articles
from app.sentiment import analyze_sentiment
from app.supabase_client import insert_article, supabase
from datetime import datetime, timedelta
import random
from collections import Counter
import re
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load .env file for local development
load_dotenv()

app = FastAPI()

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
    
    # New Crypto feeds
    "decrypt.co": "Markets & Finance",
    "bitcoinmagazine.com": "Markets & Finance",
    "thedefiant.io": "Markets & Finance",
    "theblock.co": "Markets & Finance",
    "cryptoslate.com": "Markets & Finance",
    "cryptopotato.com": "Markets & Finance",
    
    # New Stock Market feeds
    "investors.com": "Markets & Finance",
    "fool.com": "Markets & Finance",
    "stocknews.com": "Markets & Finance",
    "benzinga.com": "Markets & Finance",
    "investing.com": "Markets & Finance",
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


# Daily Summary - For Twitter/Social Media posts
@app.get("/daily-summary")
def daily_summary(hours: int = 24, top: int = 3):
    """Generate daily summary for social media posting"""
    
    # Get time threshold
    time_threshold = datetime.utcnow() - timedelta(hours=hours)
    
    # Get sentiment counts
    result = (
        supabase.table("articles")
        .select("sentiment_label")
        .gte("published_at", time_threshold.isoformat())
        .execute()
    )
    
    # Count sentiments (handle both upper and lowercase)
    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0, "mixed": 0}
    for article in result.data:
        label = article.get("sentiment_label", "neutral").lower()
        if label in sentiment_counts:
            sentiment_counts[label] += 1
    
    # Get latest articles with images first, then without
    latest_with_images = (
        supabase.table("articles")
        .select("title, sentiment_label, image_url")
        .gte("published_at", time_threshold.isoformat())
        .neq("image_url", None)
        .neq("image_url", "")
        .order("published_at", desc=True)
        .limit(top)
        .execute()
    )
    
    latest_articles = []
    featured_image = None
    
    for article in latest_with_images.data:
        title = article["title"][:60] + "..." if len(article["title"]) > 60 else article["title"]
        latest_articles.append({
            "title": title,
            "sentiment": article["sentiment_label"],
            "image_url": article.get("image_url")
        })
        # Use first image as featured
        if not featured_image and article.get("image_url"):
            featured_image = article["image_url"]
    
    # If not enough articles with images, get more without
    if len(latest_articles) < top:
        remaining = top - len(latest_articles)
        latest_no_images = (
            supabase.table("articles")
            .select("title, sentiment_label")
            .gte("published_at", time_threshold.isoformat())
            .or_("image_url.is.null,image_url.eq.")
            .order("published_at", desc=True)
            .limit(remaining)
            .execute()
        )
        for article in latest_no_images.data:
            title = article["title"][:60] + "..." if len(article["title"]) > 60 else article["title"]
            latest_articles.append({
                "title": title,
                "sentiment": article["sentiment_label"],
                "image_url": None
            })
    
    # Generate tweet text
    today = datetime.utcnow().strftime("%b %d")
    
    # Build tweet - compact sentiment line
    p = sentiment_counts["positive"]
    n = sentiment_counts["negative"]
    u = sentiment_counts["neutral"]
    m = sentiment_counts["mixed"]
    
    tweet = f"🟢{p} 🔴{n} 🟡{u} 🟣{m}\n\n"
    
    for article in latest_articles[:3]:
        tweet += f"• {article['title']}\n"
    
    tweet += "\ntechsentiments.com"
    
    return {
        "sentiment_counts": sentiment_counts,
        "total_articles": sum(sentiment_counts.values()),
        "latest_articles": latest_articles,
        "featured_image": featured_image,
        "tweet_text": tweet,
        "tweet_length": len(tweet)
    }


# Autocomplete - suggest keywords as user types
@app.get("/autocomplete")
def autocomplete(q: str):
    """Return keyword suggestions based on article titles"""
    if not q or len(q) < 2:
        return {"suggestions": []}
    
    search_term = q.lower().strip()
    
    # Search for titles containing the search term
    result = (
        supabase.table("articles")
        .select("title")
        .ilike("title", f"%{search_term}%")
        .order("published_at", desc=True)
        .limit(100)
        .execute()
    )
    
    # Extract unique keywords from titles that match
    keywords = {}
    stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "has", "have", "had", "do", "does", "did", "will", "would", "could", "should", "it", "its", "this", "that", "these", "those", "as", "from", "into", "about", "how", "what", "when", "where", "who", "why", "which", "new", "says", "said"}
    
    for article in result.data:
        title = article["title"].lower()
        # Find words that start with the search term
        words = re.findall(r'\b([a-z]+)\b', title)
        for word in words:
            if word.startswith(search_term) and word not in stop_words and len(word) >= 3:
                keywords[word] = keywords.get(word, 0) + 1
    
    # Sort by frequency and return top 5
    sorted_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)
    suggestions = [k for k, v in sorted_keywords[:5]]
    
    return {"suggestions": suggestions}


# Get total article count (for pagination)
@app.get("/articles/count")
def get_article_count(category: str = None):
    query = supabase.table("articles").select("id", count="exact")
    
    if category:
        query = query.eq("category", category)
    
    result = query.execute()
    return {"count": result.count}


# Homepage - Returns articles sorted by published_at (newest first)
@app.get("/articles")
def get_articles(category: str = None, limit: int = 50):
    # Build base query
    query = supabase.table("articles").select("*")

    # Add category filter if provided
    if category:
        query = query.eq("category", category)

    # Execute query - SORT BY published_at DESC (newest first)
    result = query.order("published_at", desc=True).limit(limit).execute()

    return {
        "articles": result.data
    }


# Articles WITH images - paginated
@app.get("/articles/images")
def get_articles_with_images(page: int = 0, limit: int = 12, category: str = None, source: str = None):
    offset = page * limit
    
    query = supabase.table("articles").select("*", count="exact")
    
    if category:
        query = query.eq("category", category)
    
    if source:
        query = query.ilike("source_url", f"%{source}%")
    
    # Filter for articles WITH images
    query = query.neq("image_url", None).neq("image_url", "")
    
    result = query.order("published_at", desc=True).range(offset, offset + limit - 1).execute()
    
    return {
        "articles": result.data,
        "count": result.count,
        "page": page,
        "has_more": (offset + limit) < (result.count or 0)
    }


# Articles WITHOUT images - paginated
@app.get("/articles/text")
def get_articles_without_images(page: int = 0, limit: int = 12, category: str = None, source: str = None):
    offset = page * limit
    
    query = supabase.table("articles").select("*", count="exact")
    
    if category:
        query = query.eq("category", category)
    
    if source:
        query = query.ilike("source_url", f"%{source}%")
    
    # Filter for articles WITHOUT images (null or empty)
    query = query.or_("image_url.is.null,image_url.eq.")
    
    result = query.order("published_at", desc=True).range(offset, offset + limit - 1).execute()
    
    return {
        "articles": result.data,
        "count": result.count,
        "page": page,
        "has_more": (offset + limit) < (result.count or 0)
    }


# Pagination - Returns next batch of articles sorted by published_at
@app.get("/articles/page/{page_num}")
def get_articles_paginated(page_num: int, category: str = None, limit: int = 50):
    offset = page_num * limit

    # Build base query
    query = supabase.table("articles").select("*")

    # Add category filter if provided
    if category:
        query = query.eq("category", category)

    # Execute query - SORT BY published_at DESC (newest first)
    result = query.order("published_at", desc=True).range(offset, offset + limit - 1).execute()

    return {
        "articles": result.data
    }


# Helper function to fix query using OpenAI
def ai_fix_query(query):
    """Use OpenAI to interpret unclear queries, fix typos, extract single keyword"""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "system",
                "content": """You fix search queries for a tech news database.
Your job is to output ONE clean search keyword (the most important word).

Rules:
1. Fix typos: bitcoins→bitcoin, nviidia→nvidia, etherium→ethereum, aplle→apple
2. Extract the MOST IMPORTANT keyword from the query (just 1 word)
3. Convert tickers: TSLA→tesla, AAPL→apple, NVDA→nvidia, BTC→bitcoin, ETH→ethereum
4. Output ONLY 1 lowercase word, nothing else
5. No punctuation, no explanation, just the keyword
6. Prioritize: company names > crypto names > tech terms > generic words

Examples:
bitcoins price → bitcoin
tesla stock → tesla
nviidia earnings → nvidia
TSLA stock news → tesla
what's happening with bitcoin today → bitcoin
latest on etherium price → ethereum
apple stocks falling → apple
show me nvidia gpu news → nvidia
how is tesla doing → tesla
whats up with openai → openai
bitcoin and ethereum → bitcoin
microsoft earnings report → microsoft
google stock price → google"""
            }, {
                "role": "user", 
                "content": query
            }],
            max_tokens=10,
            temperature=0
        )
        result = response.choices[0].message.content.strip().lower()
        # Remove any quotes or extra characters
        result = result.replace('"', '').replace("'", "").strip()
        # Take only first word if multiple returned
        result = result.split()[0] if result.split() else result
        print(f"AI processed: '{query}' → '{result}'")
        return result
    except Exception as e:
        print(f"OpenAI error: {e}")
        # Fallback: just return first word lowercase
        words = query.lower().strip().split()
        return words[0] if words else query.lower().strip()


# Helper function to get word variations (singular/plural)
def get_word_variations(word):
    """Return list of word variations for better matching"""
    variations = [word]
    
    # Common singular/plural patterns
    if word.endswith('s') and len(word) > 3:
        variations.append(word[:-1])  # stocks -> stock
    else:
        variations.append(word + 's')  # stock -> stocks
    
    return list(set(variations))


# Helper function to search database
def search_database(search_term, offset, limit):
    """Search articles in database with word variations"""
    # Get variations of the search term
    variations = get_word_variations(search_term)
    
    # Build OR conditions for all variations
    conditions = []
    for var in variations:
        conditions.append(f"title.ilike.%{var}%")
        conditions.append(f"summary.ilike.%{var}%")
    
    result = (
        supabase.table("articles")
        .select("*", count="exact")
        .or_(",".join(conditions))
        .order("published_at", desc=True)
        .range(offset, offset + limit - 1)
        .execute()
    )
    
    return result


# Search articles by keyword (searches title and summary)
@app.get("/search")
def search_articles(q: str, page: int = 0, limit: int = 50):
    """
    Smart search with AI-powered query processing.
    Always uses GPT-4o-mini to:
    - Fix typos (nviidia → nvidia)
    - Extract keywords from long queries
    - Handle stock tickers (TSLA → tesla)
    """
    if not q or len(q.strip()) < 2:
        return {"articles": [], "query": q, "count": 0, "ai_corrected": False}
    
    original_query = q.strip()
    offset = page * limit
    
    # Always use AI to process query (only on first page to save on pagination)
    if page == 0:
        print(f"Processing query with AI: {original_query}")
        corrected_term = ai_fix_query(original_query)
        print(f"AI result: {original_query} → {corrected_term}")
    else:
        # For pagination, use the query as-is (already corrected on page 0)
        corrected_term = original_query.lower()
    
    ai_corrected = corrected_term.lower() != original_query.lower()
    
    # Search database with AI-processed query
    result = search_database(corrected_term, offset, limit)
    
    return {
        "articles": result.data,
        "query": original_query,
        "corrected_query": corrected_term if ai_corrected else None,
        "count": result.count,
        "page": page,
        "has_more": (offset + limit) < (result.count or 0),
        "ai_corrected": ai_corrected
    }


# ===========================================
# SUBSCRIBER ENDPOINTS
# ===========================================

class SubscribeRequest(BaseModel):
    email: str


@app.post("/subscribe")
def subscribe(request: SubscribeRequest):
    """
    Subscribe endpoint - saves email to Supabase
    """
    email = request.email.strip().lower()
    
    # Basic validation
    if not email or "@" not in email:
        return {"success": False, "message": "Invalid email address"}
    
    try:
        # Check if already subscribed
        existing = supabase.table("subscribers").select("id").eq("email", email).execute()
        
        if existing.data:
            return {"success": True, "message": "You're already subscribed!"}
        
        # Insert new subscriber
        supabase.table("subscribers").insert({
            "email": email,
            "created_at": datetime.utcnow().isoformat(),
            "source": "website_footer"
        }).execute()
        
        return {"success": True, "message": "Thanks for subscribing!"}
    
    except Exception as e:
        print(f"Subscribe error: {e}")
        return {"success": False, "message": "Something went wrong. Please try again."}


@app.get("/subscribers")
def get_subscribers():
    """Get all subscribers (admin endpoint)"""
    result = supabase.table("subscribers").select("*").order("created_at", desc=True).execute()
    return {"subscribers": result.data, "count": len(result.data)}
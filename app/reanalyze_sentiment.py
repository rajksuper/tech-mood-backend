"""
Re-analyze sentiment for all existing articles in Supabase.
Run this after updating sentiment.py to apply new logic.

Usage: python reanalyze_sentiment.py
"""

from supabase import create_client
from dotenv import load_dotenv
import os
import time

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise Exception("ENV variables not loaded. Check .env file.")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# Import the updated sentiment analyzer
from sentiment import analyze_sentiment


def reanalyze_all_articles(batch_size=100):
    """
    Fetch all articles and re-run sentiment analysis with updated logic.
    """
    print("=" * 50)
    print("SENTIMENT RE-ANALYSIS SCRIPT")
    print("=" * 50)
    
    # Get total count
    count_result = supabase.table("articles").select("id", count="exact").execute()
    total_articles = count_result.count
    print(f"\nTotal articles to process: {total_articles}")
    
    # Track changes
    stats = {
        "total": 0,
        "changed": 0,
        "positive": 0,
        "negative": 0,
        "neutral": 0,
        "mixed": 0
    }
    
    offset = 0
    
    while offset < total_articles:
        # Fetch batch of articles
        result = supabase.table("articles") \
            .select("id, title, summary, sentiment_label") \
            .range(offset, offset + batch_size - 1) \
            .execute()
        
        articles = result.data
        
        if not articles:
            break
        
        print(f"\nProcessing batch {offset // batch_size + 1} ({offset + 1}-{offset + len(articles)} of {total_articles})...")
        
        for article in articles:
            stats["total"] += 1
            
            # Combine title and summary for analysis
            text = f"{article['title']}. {article.get('summary', '')}"
            
            # Get new sentiment
            new_label, new_score, _ = analyze_sentiment(text)
            old_label = article.get("sentiment_label", "").lower()
            
            # Update if changed
            if new_label != old_label:
                stats["changed"] += 1
                
                # Update in database
                supabase.table("articles") \
                    .update({
                        "sentiment_label": new_label,
                        "sentiment_score": new_score
                    }) \
                    .eq("id", article["id"]) \
                    .execute()
                
                print(f"  ↻ Changed: '{article['title'][:50]}...' | {old_label} → {new_label}")
            
            # Track new distribution
            stats[new_label] = stats.get(new_label, 0) + 1
        
        offset += batch_size
        
        # Small delay to avoid rate limiting
        time.sleep(0.5)
    
    # Print summary
    print("\n" + "=" * 50)
    print("RE-ANALYSIS COMPLETE")
    print("=" * 50)
    print(f"\nTotal processed: {stats['total']}")
    print(f"Labels changed:  {stats['changed']} ({stats['changed']*100//max(stats['total'],1)}%)")
    print(f"\nNew distribution:")
    print(f"  ✓ Positive: {stats.get('positive', 0)}")
    print(f"  ✗ Negative: {stats.get('negative', 0)}")
    print(f"  ○ Neutral:  {stats.get('neutral', 0)}")
    print(f"  ◐ Mixed:    {stats.get('mixed', 0)}")
    print("=" * 50)


if __name__ == "__main__":
    reanalyze_all_articles()
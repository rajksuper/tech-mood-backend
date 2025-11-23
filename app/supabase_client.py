from supabase import create_client
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise Exception("ENV variables not loaded. Check .env file.")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

def insert_article(article):
    return supabase.table("articles").insert(article).execute()

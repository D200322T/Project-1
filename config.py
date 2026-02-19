"""
Central configuration for the Beauty Product Tracker.

Defines the products tracked, their stock tickers, search keywords,
and subreddits/RSS feeds used for mention scraping.
"""

# ─── Products ────────────────────────────────────────────────────────────────
# Ordered list: first entry (L'Oréal) appears first in the dashboard.
PRODUCTS = [
    {
        "name": "L'Oréal",
        "ticker": "LRLCY",        # L'Oréal ADR on OTC markets
        "keywords": ["loreal", "l'oreal", "l'oréal", "loréal", "lorealparis"],
        "brand_names": ["L'Oréal Paris", "Maybelline", "Garnier", "Lancôme", "NYX", "CeraVe", "La Roche-Posay"],
    },
    {
        "name": "Estée Lauder",
        "ticker": "EL",
        "keywords": ["estee lauder", "estée lauder", "esteelauder"],
        "brand_names": ["Clinique", "MAC", "Bobbi Brown", "Jo Malone", "Aveda", "Origins"],
    },
    {
        "name": "e.l.f. Beauty",
        "ticker": "ELF",
        "keywords": ["elf beauty", "e.l.f.", "elfcosmetics", "elf cosmetics"],
        "brand_names": ["e.l.f.", "Well People", "Keys Soulcare"],
    },
    {
        "name": "Ulta Beauty",
        "ticker": "ULTA",
        "keywords": ["ulta beauty", "ulta", "ultabeauty"],
        "brand_names": ["Ulta Beauty"],
    },
    {
        "name": "Coty",
        "ticker": "COTY",
        "keywords": ["coty inc", "coty beauty"],
        "brand_names": ["CoverGirl", "Rimmel", "Sally Hansen", "OPI", "Burberry Beauty"],
    },
    {
        "name": "Revlon",
        "ticker": "REV",
        "keywords": ["revlon"],
        "brand_names": ["Revlon", "Elizabeth Arden", "American Crew"],
    },
    {
        "name": "Beiersdorf",
        "ticker": "BDRFY",
        "keywords": ["beiersdorf"],
        "brand_names": ["NIVEA", "Eucerin", "La Prairie", "Hansaplast"],
    },
]

# ─── Scraping Sources ─────────────────────────────────────────────────────────

# Reddit subreddits to scan for beauty product mentions
REDDIT_SUBREDDITS = [
    "MakeupAddiction",
    "SkincareAddiction",
    "beauty",
    "BeautyGuruChatter",
    "drugstorebeauty",
    "FrugalFemaleFashion",
    "AsianBeauty",
    "Sephora",
    "Ulta",
    "investingforbeginners",
    "stocks",
    "investing",
]

# How many posts to pull per subreddit per scrape run (Reddit API limit: 100)
REDDIT_POST_LIMIT = 100

# RSS feeds for beauty/finance news
RSS_FEEDS = [
    "https://www.allure.com/feed/rss",
    "https://www.byrdie.com/rss",
    "https://www.instyle.com/rss",
    "https://www.harpersbazaar.com/rss",
    "https://rss.app/feeds/tQRWWbzGBkiKHDQH.xml",  # BeautyMatter
]

# ─── Data / Cache ─────────────────────────────────────────────────────────────

DATA_DIR = "data"
CACHE_FILE = "data/mentions_cache.json"
STOCK_CACHE_FILE = "data/stock_cache.json"

# Number of calendar days of historical data to load
HISTORY_DAYS = 90

# ─── Regression ───────────────────────────────────────────────────────────────

# Lag offsets (in trading days) to test: does today's mentions predict tomorrow's move?
LAG_DAYS = [0, 1, 2, 3]

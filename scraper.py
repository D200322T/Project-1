"""
scraper.py — Mention scraping from Reddit and RSS news feeds.

Counts how many times each beauty product (and its associated brands/keywords)
appears across posts/comments on Reddit and articles from RSS feeds.

Results are aggregated per product per day and stored in a JSON cache so that
the dashboard can load quickly without hitting APIs on every render.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import feedparser
import requests

import config

# ─── Optional imports (graceful degradation if creds not provided) ─────────────
try:
    import praw

    PRAW_AVAILABLE = True
except ImportError:
    PRAW_AVAILABLE = False


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _load_cache(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def _save_cache(path: str, data: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _build_pattern(product: dict) -> re.Pattern:
    """Build a case-insensitive regex that matches any keyword or brand name."""
    terms = product["keywords"] + product["brand_names"]
    escaped = [re.escape(t) for t in terms]
    return re.compile("|".join(escaped), re.IGNORECASE)


def _date_str(ts: float) -> str:
    """Convert a UTC timestamp to YYYY-MM-DD string."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def _today() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")


def _date_range(days: int) -> list[str]:
    today = datetime.now(tz=timezone.utc).date()
    return [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]


# ─── Reddit scraper ────────────────────────────────────────────────────────────

def _get_reddit_client() -> Optional["praw.Reddit"]:
    if not PRAW_AVAILABLE:
        return None
    client_id = os.getenv("REDDIT_CLIENT_ID", "")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET", "")
    user_agent = os.getenv("REDDIT_USER_AGENT", "BeautyProductTracker/1.0")
    if not client_id or not client_secret:
        return None
    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )


def scrape_reddit(days: int = config.HISTORY_DAYS) -> dict[str, dict[str, int]]:
    """
    Scrape Reddit for product mentions.

    Returns:
        { product_name: { "YYYY-MM-DD": mention_count, ... }, ... }
    """
    reddit = _get_reddit_client()
    if reddit is None:
        print("[scraper] Reddit credentials not found — skipping Reddit scraping.")
        return {}

    patterns = {p["name"]: _build_pattern(p) for p in config.PRODUCTS}
    cutoff = time.time() - days * 86400

    # counts[product][date] = int
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for sub_name in config.REDDIT_SUBREDDITS:
        try:
            sub = reddit.subreddit(sub_name)
            for post in sub.new(limit=config.REDDIT_POST_LIMIT):
                if post.created_utc < cutoff:
                    continue
                date = _date_str(post.created_utc)
                text = f"{post.title} {post.selftext}"
                for product in config.PRODUCTS:
                    name = product["name"]
                    hits = len(patterns[name].findall(text))
                    if hits:
                        counts[name][date] += hits

                # Also scan top-level comments
                post.comments.replace_more(limit=0)
                for comment in post.comments.list():
                    if comment.created_utc < cutoff:
                        continue
                    c_date = _date_str(comment.created_utc)
                    for product in config.PRODUCTS:
                        name = product["name"]
                        hits = len(patterns[name].findall(comment.body))
                        if hits:
                            counts[name][c_date] += hits

            time.sleep(0.5)  # be polite to the API
        except Exception as exc:
            print(f"[scraper] Reddit error on r/{sub_name}: {exc}")

    return {k: dict(v) for k, v in counts.items()}


# ─── RSS / News scraper ────────────────────────────────────────────────────────

def scrape_rss(days: int = config.HISTORY_DAYS) -> dict[str, dict[str, int]]:
    """
    Scrape RSS feeds for product mentions.

    Returns:
        { product_name: { "YYYY-MM-DD": mention_count, ... }, ... }
    """
    patterns = {p["name"]: _build_pattern(p) for p in config.PRODUCTS}
    cutoff_dt = datetime.now(tz=timezone.utc) - timedelta(days=days)

    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for feed_url in config.RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries:
                # Parse published date
                published = None
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                    published = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
                else:
                    published = datetime.now(tz=timezone.utc)

                if published < cutoff_dt:
                    continue

                date = published.strftime("%Y-%m-%d")
                text = f"{entry.get('title', '')} {entry.get('summary', '')}"

                for product in config.PRODUCTS:
                    name = product["name"]
                    hits = len(patterns[name].findall(text))
                    if hits:
                        counts[name][date] += hits

        except Exception as exc:
            print(f"[scraper] RSS error on {feed_url}: {exc}")

    return {k: dict(v) for k, v in counts.items()}


# ─── NewsAPI scraper ──────────────────────────────────────────────────────────

def scrape_newsapi(days: int = config.HISTORY_DAYS) -> dict[str, dict[str, int]]:
    """
    Scrape NewsAPI (newsapi.org) for product mentions.
    Requires NEWS_API_KEY environment variable.

    Returns:
        { product_name: { "YYYY-MM-DD": mention_count, ... }, ... }
    """
    api_key = os.getenv("NEWS_API_KEY", "")
    if not api_key:
        print("[scraper] NEWS_API_KEY not set — skipping NewsAPI scraping.")
        return {}

    from_date = (datetime.now(tz=timezone.utc) - timedelta(days=min(days, 30))).strftime("%Y-%m-%d")
    patterns = {p["name"]: _build_pattern(p) for p in config.PRODUCTS}
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for product in config.PRODUCTS:
        name = product["name"]
        query = " OR ".join(f'"{kw}"' for kw in product["keywords"][:3])
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": query,
                "from": from_date,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 100,
                "apiKey": api_key,
            }
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            articles = resp.json().get("articles", [])
            for article in articles:
                published_at = article.get("publishedAt", "")
                date = published_at[:10] if published_at else _today()
                text = f"{article.get('title', '')} {article.get('description', '')}"
                hits = len(patterns[name].findall(text))
                if hits:
                    counts[name][date] += hits
            time.sleep(0.25)
        except Exception as exc:
            print(f"[scraper] NewsAPI error for {name}: {exc}")

    return {k: dict(v) for k, v in counts.items()}


# ─── Aggregation ──────────────────────────────────────────────────────────────

def _merge_counts(
    *sources: dict[str, dict[str, int]]
) -> dict[str, dict[str, int]]:
    """Merge multiple source count dicts, summing overlapping dates."""
    merged: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for source in sources:
        for product, dates in source.items():
            for date, count in dates.items():
                merged[product][date] += count
    return {k: dict(v) for k, v in merged.items()}


def run_full_scrape(days: int = config.HISTORY_DAYS) -> dict[str, dict[str, int]]:
    """
    Run all scrapers, merge results, cache to disk, and return combined counts.

    Structure returned:
        {
            "L'Oréal": {"2024-01-15": 42, "2024-01-16": 17, ...},
            "Estée Lauder": {...},
            ...
        }
    """
    print("[scraper] Starting scrape run...")

    reddit_counts = scrape_reddit(days)
    rss_counts = scrape_rss(days)
    news_counts = scrape_newsapi(days)

    combined = _merge_counts(reddit_counts, rss_counts, news_counts)

    # Ensure every product has an entry (even if zero) for each date in range
    date_range = _date_range(days)
    for product in config.PRODUCTS:
        name = product["name"]
        if name not in combined:
            combined[name] = {}
        for date in date_range:
            combined[name].setdefault(date, 0)

    _save_cache(config.CACHE_FILE, combined)
    print(f"[scraper] Done. Cache written to {config.CACHE_FILE}")
    return combined


def load_cached_mentions() -> dict[str, dict[str, int]]:
    """Load previously scraped mention counts from disk cache."""
    return _load_cache(config.CACHE_FILE)

# Beauty Product Social Media & Stock Tracker

A Streamlit dashboard that scrapes social media and news for beauty product mentions, then runs OLS regression to explore the correlation between mention volume and daily stock price changes.

**L'Oréal is shown first**, followed by Estée Lauder, e.l.f. Beauty, Ulta Beauty, Coty, Revlon, and Beiersdorf.

---

## Features

- **Mention counting** across Reddit, RSS news feeds, and NewsAPI
- **Daily stock price charts** via Yahoo Finance (yfinance)
- **OLS regression** — mentions on day *t* vs. stock return on day *t+lag*
- **Lag analysis** — test whether mentions today predict returns tomorrow (lag=1, 2, 3)
- **Regression summary table** — R², Pearson r, p-value, significance flag for every product
- **Leaderboard** and **cross-product trend** charts

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API credentials

```bash
cp .env.example .env
```

Edit `.env` and fill in:

| Variable | Required | Where to get it |
|---|---|---|
| `REDDIT_CLIENT_ID` | Yes (for Reddit) | https://www.reddit.com/prefs/apps |
| `REDDIT_CLIENT_SECRET` | Yes (for Reddit) | same as above |
| `REDDIT_USER_AGENT` | No | default: `BeautyProductTracker/1.0` |
| `NEWS_API_KEY` | No | https://newsapi.org |

The app works without any API keys — RSS feeds are always scraped, and Yahoo Finance requires no key. Reddit and NewsAPI enhance coverage.

### 3. Run the dashboard

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

---

## Architecture

```
app.py          — Streamlit dashboard (UI + orchestration)
config.py       — Products, tickers, keywords, subreddits, RSS feeds
scraper.py      — Reddit (PRAW), RSS (feedparser), NewsAPI mention scrapers
stock_data.py   — Yahoo Finance price history (yfinance)
regression.py   — OLS regression + summary statistics (scipy)
data/           — JSON caches (mentions_cache.json, stock_cache.json)
```

### Data Flow

```
[Reddit API]   ──┐
[RSS Feeds]    ──┼──► scraper.py ──► data/mentions_cache.json ──┐
[NewsAPI]      ──┘                                               │
                                                                 ▼
[Yahoo Finance] ─────────────────► stock_data.py ──────────► regression.py
                                                                 │
                                                                 ▼
                                                              app.py (Streamlit)
```

### Regression Details

For each product and each lag offset (0–3 trading days) the app fits:

```
stock_return(t + lag) = β₀ + β₁ × mentions(t) + ε
```

and reports R², Pearson r, and the p-value for β₁. Results where **p < 0.05** are flagged as statistically significant (though correlation does not imply causation).

---

## Tracked Products & Tickers

| Product | Ticker | Key Brands |
|---|---|---|
| L'Oréal | LRLCY | L'Oréal Paris, Maybelline, Garnier, Lancôme, CeraVe |
| Estée Lauder | EL | Clinique, MAC, Bobbi Brown, Jo Malone |
| e.l.f. Beauty | ELF | e.l.f., Well People, Keys Soulcare |
| Ulta Beauty | ULTA | Ulta Beauty |
| Coty | COTY | CoverGirl, Rimmel, Sally Hansen, OPI |
| Revlon | REV | Revlon, Elizabeth Arden |
| Beiersdorf | BDRFY | NIVEA, Eucerin, La Prairie |

---

## Extending the App

- **Add a product**: append an entry to `PRODUCTS` in `config.py`
- **Add a subreddit**: extend `REDDIT_SUBREDDITS` in `config.py`
- **Add an RSS feed**: extend `RSS_FEEDS` in `config.py`
- **Change regression lags**: modify `LAG_DAYS` in `config.py`

"""
stock_data.py — Fetches historical daily stock prices via yfinance.

For each tracked product, downloads OHLCV data and computes:
  - daily_return: percentage change in closing price day-over-day
  - log_return: natural-log return (better for regression normality)

Results are cached to disk so repeated dashboard loads do not re-hit Yahoo Finance.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

import config


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


# ─── Fetch ────────────────────────────────────────────────────────────────────

def fetch_stock_history(
    ticker: str,
    days: int = config.HISTORY_DAYS,
) -> pd.DataFrame:
    """
    Download daily price history for *ticker* covering the past *days* calendar days.

    Returns a DataFrame with columns:
        date (str YYYY-MM-DD), open, high, low, close, volume,
        daily_return (%), log_return
    """
    end = datetime.now(tz=timezone.utc)
    # Pad by 10 extra days to account for weekends/holidays at the boundary
    start = end - timedelta(days=days + 10)

    try:
        tk = yf.Ticker(ticker)
        df = tk.history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=True,
        )
    except Exception as exc:
        print(f"[stock_data] yfinance error for {ticker}: {exc}")
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    df = df.reset_index()
    # Normalise column name (yfinance sometimes returns 'Datetime' instead of 'Date')
    date_col = "Date" if "Date" in df.columns else "Datetime"
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                             "Close": "close", "Volume": "volume"})
    df = df[["date", "open", "high", "low", "close", "volume"]].copy()

    df = df.sort_values("date").reset_index(drop=True)
    df["daily_return"] = df["close"].pct_change() * 100          # percentage
    import math as _math
    df["log_return"] = (df["close"] / df["close"].shift(1)).apply(
        lambda x: 0.0 if pd.isna(x) or x <= 0 else _math.log(float(x))
    )

    # Keep only the requested window
    cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    df = df[df["date"] >= cutoff].reset_index(drop=True)

    return df


# ─── Batch fetch with cache ───────────────────────────────────────────────────

def fetch_all_stocks(days: int = config.HISTORY_DAYS) -> dict[str, pd.DataFrame]:
    """
    Fetch price history for all products defined in config.PRODUCTS.

    Returns:
        { product_name: DataFrame, ... }
    """
    result: dict[str, pd.DataFrame] = {}
    cached = _load_cache(config.STOCK_CACHE_FILE)
    to_save: dict[str, list] = {}

    for product in config.PRODUCTS:
        name = product["name"]
        ticker = product["ticker"]
        print(f"[stock_data] Fetching {ticker} ({name})...")
        df = fetch_stock_history(ticker, days)
        if df.empty and name in cached:
            print(f"[stock_data]   Using cached data for {name}.")
            df = pd.DataFrame(cached[name])
        result[name] = df
        if not df.empty:
            to_save[name] = df.to_dict(orient="records")

    _save_cache(config.STOCK_CACHE_FILE, to_save)
    return result


def load_cached_stocks() -> dict[str, pd.DataFrame]:
    """Load previously fetched stock data from disk cache."""
    cached = _load_cache(config.STOCK_CACHE_FILE)
    return {name: pd.DataFrame(records) for name, records in cached.items()}


# ─── Convenience ──────────────────────────────────────────────────────────────

def get_latest_price(ticker: str) -> dict:
    """Return the most recent close price and daily return for a ticker."""
    try:
        tk = yf.Ticker(ticker)
        info = tk.fast_info
        price = getattr(info, "last_price", None)
        prev_close = getattr(info, "previous_close", None)
        if price and prev_close and prev_close != 0:
            change_pct = (price - prev_close) / prev_close * 100
        else:
            change_pct = None
        return {"price": price, "change_pct": change_pct}
    except Exception:
        return {"price": None, "change_pct": None}

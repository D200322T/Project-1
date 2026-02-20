"""
app.py — Streamlit dashboard for the Beauty Product Social-Media & Stock Tracker.

Run with:
    streamlit run app.py

Features:
  - Live mention counts per product per platform (Reddit, news, RSS)
  - Historical mention trend charts
  - Daily stock price + return charts
  - OLS regression: mentions vs. stock return (with lag selector)
  - Regression summary table across all products
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import config
import regression as reg
import scraper
import stock_data as sd

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Beauty Product Tracker",
    page_icon="💄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Helpers ──────────────────────────────────────────────────────────────────

PRODUCT_NAMES = [p["name"] for p in config.PRODUCTS]
TICKER_MAP = {p["name"]: p["ticker"] for p in config.PRODUCTS}


@st.cache_data(ttl=3600, show_spinner=False)
def load_mentions(force_refresh: bool = False) -> dict:
    """Load mention data from cache or trigger a fresh scrape."""
    cached = scraper.load_cached_mentions()
    if not cached or force_refresh:
        with st.spinner("Scraping social media & news for product mentions..."):
            return scraper.run_full_scrape()
    return cached


@st.cache_data(ttl=1800, show_spinner=False)
def load_stocks(force_refresh: bool = False) -> dict:
    """Load stock data from cache or fetch from Yahoo Finance."""
    if force_refresh:
        with st.spinner("Fetching stock price history from Yahoo Finance..."):
            raw = sd.fetch_all_stocks()
    else:
        raw = sd.load_cached_stocks()
        if not raw:
            with st.spinner("Fetching stock price history from Yahoo Finance..."):
                raw = sd.fetch_all_stocks()
    return {name: df for name, df in raw.items()}


def total_mentions(mention_data: dict, product_name: str) -> int:
    counts = mention_data.get(product_name, {})
    return sum(counts.values())


def mentions_series(mention_data: dict, product_name: str) -> pd.DataFrame:
    counts = mention_data.get(product_name, {})
    if not counts:
        return pd.DataFrame(columns=["date", "mentions"])
    df = pd.DataFrame(list(counts.items()), columns=["date", "mentions"])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def format_change(pct: float | None) -> str:
    if pct is None:
        return "N/A"
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.2f}%"


def sig_badge(is_sig: bool) -> str:
    return "✅ Yes" if is_sig else "❌ No"


def normalize_series(s: pd.Series, method: str) -> pd.Series:
    if method == "None" or s.empty:
        return s
    if method == "Z-score":
        std = s.std()
        return (s - s.mean()) / std if std > 0 else s - s.mean()
    if method == "Min-Max":
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn) if mx > mn else pd.Series(np.zeros(len(s)), index=s.index)
    return s


# ─── Regression narrative helpers ─────────────────────────────────────────────

REGRESSION_CONTEXT: dict[str, str] = {
    "L'Oréal": (
        "L'Oréal is the world's largest beauty conglomerate, with revenue spread across "
        "mass-market (Maybelline, NYX), luxury (Lancôme, Kiehl's), and dermatology segments. "
        "Its Paris-listed stock (OR.PA) moves primarily on earnings surprises, euro/dollar "
        "dynamics, and macro consumer-spending sentiment rather than day-to-day social chatter. "
        "Brand mentions spike around product launches, sustainability announcements, and M&A "
        "speculation, but these rarely translate to same-day or next-day price action. A weak "
        "or insignificant regression is the expected baseline here. If any lag shows "
        "significance, consider whether a major PR event coincided with an analyst upgrade "
        "in that window — correlation without a clear mechanism is likely coincidental."
    ),
    "Estée Lauder": (
        "Estée Lauder (EL) has been one of the most volatile major beauty stocks since 2022 "
        "due to heavy exposure to China's duty-free travel-retail channel. Mention spikes are "
        "often driven by sub-brand launches (MAC, La Mer, Clinique) rather than corporate "
        "news, diluting any clean signal. Crucially, when EL mentions surge on negative news "
        "(China slowdown coverage, inventory destocking), returns tend to fall — this can flip "
        "the correlation sign or inflate residual variance and make the regression noisy. "
        "A near-zero or negative R² at short lags is plausible. Any positive lag-2 or lag-3 "
        "signal might reflect beauty-press coverage flowing into short-term retail interest "
        "before analysts react."
    ),
    "e.l.f. Beauty": (
        "e.l.f. Beauty (ELF) is arguably the most socially-driven stock in this dataset — its "
        "entire growth story was built on TikTok virality and Gen Z engagement, and its "
        "valuation multiple reflects that narrative premium. A positive, potentially significant "
        "lag-1 correlation would be consistent with social buzz leading consumer purchases, "
        "which then surprise analysts and move the stock. However, the scraper captures Reddit "
        "and news mentions; if the primary channel driving demand is TikTok (not scraped), the "
        "signal will be muted. If significance appears at lag = 0, consider whether large "
        "Reddit threads coincide with stock run-up days as a consequence of momentum — not "
        "necessarily as a cause."
    ),
    "Revlon": (
        "Revlon emerged from Chapter 11 bankruptcy in 2023 and its equity is extremely thinly "
        "traded with erratic, low-liquidity price swings. A large proportion of its mentions "
        "are driven by bankruptcy and restructuring coverage rather than genuine consumer demand "
        "signals. Expect near-zero R² and high p-values across all lags — the noise-to-signal "
        "ratio is very high, and any apparent correlation in a small sample is likely spurious. "
        "A single outlier day (e.g. a restructuring announcement causing a large mention spike "
        "and a sharp price move) can single-handedly dominate the regression slope, producing "
        "a misleadingly 'clean' line that disappears if that observation is removed."
    ),
    "Maybelline": (
        "Maybelline is a wholly-owned L'Oréal subsidiary with no independent stock listing; "
        "mentions feed indirectly into L'Oréal's revenue line. Running this regression against "
        "L'Oréal's ticker means the mention signal is diluted by all other L'Oréal businesses "
        "(luxury, dermatology, professional products). Mention spikes tend to follow influencer "
        "collaborations and hero-product moments (Sky High mascara, Fit Me franchise). Any weak "
        "positive lag-1 effect, if present, would suggest that Maybelline buzz contributes to "
        "L'Oréal's broader brand sentiment — but isolating that contribution is not possible "
        "without product-level revenue data."
    ),
}


def render_regression_breakdown(
    name: str, r: "reg.RegressionResult", all_lag_results: list
) -> None:
    """Render a comprehensive methodology + results breakdown for one product."""
    st.markdown("#### Regression Results Breakdown")

    max_lag = max((lr.lag_days for lr in all_lag_results), default=r.lag_days)
    best = reg.best_lag_result(all_lag_results)
    best_lag_note = (
        f"Across all computed lags (0–{max_lag}), the **best-performing lag is "
        f"{best.lag_days} day(s)** (R² = {best.r_squared:.4f})."
        if best else ""
    )

    r2_word = (
        "negligible (<1%)" if r.r_squared < 0.01
        else "very weak (1–5%)" if r.r_squared < 0.05
        else "weak (5–20%)" if r.r_squared < 0.20
        else "moderate (20–50%)"
    )
    pearson_word = (
        "negligible" if abs(r.pearson_r) < 0.1
        else "weak" if abs(r.pearson_r) < 0.3
        else "moderate" if abs(r.pearson_r) < 0.5
        else "strong"
    )
    direction_sign = "increase" if r.slope >= 0 else "decrease"

    with st.expander("What do these numbers mean?", expanded=True):
        st.markdown(f"""
**How lag regression works**

A "lag" of **{r.lag_days} trading day(s)** means we test: *"Do mention counts on day T predict \
the stock return on day T + {r.lag_days}?"*

- **Lag = 0** — same-day co-movement. Do mentions and returns rise together on the same day?
- **Lag = 1** — leading indicator test. Does today's social buzz predict *tomorrow's* return?
- **Higher lags** — tests whether sentiment leads price moves by several trading days.

{best_lag_note}

---

**Current result at lag = {r.lag_days}**

| Metric | Value | What it means |
|---|---|---|
| **R²** | {r.r_squared:.4f} | Mention volume explains ~{r.r_squared * 100:.1f}% of return variance — {r2_word} explanatory power. |
| **Pearson r** | {r.pearson_r:.4f} | {pearson_word.capitalize()} {"positive" if r.pearson_r >= 0 else "negative"} linear association between mention count and {r.lag_days}-day-ahead return. |
| **p-value** | {r.p_value:.4f} | {"Significant — the slope is unlikely to be zero by chance (p < 0.05)." if r.is_significant else "Not significant — we cannot reject the null hypothesis of no relationship (p ≥ 0.05)."} |
| **Slope** | {r.slope:.6f} | Each additional mention is associated with a {abs(r.slope):.6f}% {direction_sign} in the {r.lag_days}-day-ahead return. |
| **Intercept** | {r.intercept:.4f} | Predicted return when mentions = 0 is {r.intercept:.4f}%. |
| **n (obs.)** | {r.n_observations} | Paired trading-day observations used to fit the model. |
""")

    if r.is_significant:
        st.success(
            f"**Significant result:** At lag {r.lag_days}, {name} shows a statistically "
            f"significant {r.direction} correlation between mention volume and stock return "
            f"(p = {r.p_value:.4f}). R² = {r.r_squared:.4f} — social mentions explain "
            f"~{r.r_squared * 100:.1f}% of return variance. Correlation does not imply causation."
        )
    else:
        st.info(
            f"**No significant relationship:** At lag {r.lag_days}, there is no statistically "
            f"significant linear relationship between {name} mention volume and stock return "
            f"(p = {r.p_value:.4f}). Social media activity alone is not a reliable predictor "
            f"of price movements at this lag."
        )


def render_regression_suggestions(name: str) -> None:
    """Render brand-specific contextual suggestions about the regression results."""
    context = REGRESSION_CONTEXT.get(name)
    if not context:
        return
    st.markdown("#### Regression Suggestions")
    st.warning(f"**Contextual factors that may be driving these numbers for {name}:**\n\n{context}")


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("💄 Beauty Tracker")
    st.markdown("---")

    st.subheader("Data Controls")
    if st.button("Refresh Mentions (scrape now)", use_container_width=True):
        st.cache_data.clear()
        load_mentions(force_refresh=True)
        st.success("Mentions refreshed!")

    if st.button("Refresh Stock Prices", use_container_width=True):
        st.cache_data.clear()
        load_stocks(force_refresh=True)
        st.success("Stock data refreshed!")

    st.markdown("---")
    st.subheader("Filters")
    selected_products = st.multiselect(
        "Products to display",
        options=PRODUCT_NAMES,
        default=PRODUCT_NAMES,
    )
    history_days = st.slider(
        "History window (days)",
        min_value=7,
        max_value=config.HISTORY_DAYS,
        value=30,
        step=1,
    )
    selected_lags = st.multiselect(
        "Lag days to compute",
        options=list(range(8)),
        default=config.LAG_DAYS,
        help="Which lag offsets (trading days) to include in regression analysis.",
    )
    if not selected_lags:
        selected_lags = config.LAG_DAYS

    selected_lag = st.selectbox(
        "Display lag (days)",
        options=sorted(selected_lags),
        index=min(1, len(selected_lags) - 1),
        help="Lag=0: same-day correlation. Lag=1: do today's mentions predict tomorrow's return?",
    )

    st.markdown("---")
    st.subheader("Normalization")
    normalization = st.selectbox(
        "Mention normalization",
        options=["None", "Z-score", "Min-Max"],
        index=0,
        help=(
            "None: raw counts. "
            "Z-score: subtract mean, divide by std. "
            "Min-Max: scale to [0, 1]. "
            "Applied to mention charts for fair cross-product comparison."
        ),
    )

    st.markdown("---")
    st.caption("Data sources: Reddit · RSS feeds · NewsAPI · Yahoo Finance")
    st.caption(f"Last run: {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")


# ─── Load data ────────────────────────────────────────────────────────────────

mention_data = load_mentions()
stock_dfs = load_stocks()
all_reg_results = reg.run_all_regressions(mention_data, stock_dfs, lags=selected_lags)

# ─── Header ───────────────────────────────────────────────────────────────────

st.title("Beauty Product Social Media & Stock Tracker")
st.markdown(
    "Track how many times beauty brands are mentioned across Reddit and news outlets, "
    "and see whether mention volume correlates with daily stock price movements."
)

# ─── Top KPI row (L'Oréal first, matching config order) ──────────────────────

st.markdown("## Mention Overview")
cols = st.columns(len(selected_products))
for i, name in enumerate(p for p in PRODUCT_NAMES if p in selected_products):
    total = total_mentions(mention_data, name)
    ticker = TICKER_MAP[name]
    live = sd.get_latest_price(ticker)
    price_str = f"${live['price']:.2f}" if live["price"] else "—"
    change_str = format_change(live["change_pct"])
    delta_color = "normal"
    with cols[i]:
        st.metric(
            label=name,
            value=f"{total:,} mentions",
            delta=f"{price_str} ({change_str})",
            delta_color=delta_color,
        )

# ─── Per-product deep-dive ────────────────────────────────────────────────────

st.markdown("---")
st.markdown("## Per-Product Deep Dive")

for product_cfg in config.PRODUCTS:
    name = product_cfg["name"]
    if name not in selected_products:
        continue

    ticker = product_cfg["ticker"]
    stock_df = stock_dfs.get(name, pd.DataFrame())
    m_series = mentions_series(mention_data, name)
    lag_results = all_reg_results.get(name, [])
    best = reg.best_lag_result(lag_results)

    with st.expander(f"### {name}  ({ticker})", expanded=(name == "L'Oréal")):

        col_left, col_right = st.columns(2)

        # ── Mentions over time ──
        with col_left:
            st.subheader("Mention Count Over Time")
            if not m_series.empty:
                cutoff = pd.Timestamp.now() - pd.Timedelta(days=history_days)
                m_filtered = m_series[m_series["date"] >= cutoff].copy()
                m_filtered["mentions"] = normalize_series(m_filtered["mentions"], normalization)
                y_label = f"Mentions ({normalization})" if normalization != "None" else "Mentions"
                fig_mentions = px.bar(
                    m_filtered,
                    x="date",
                    y="mentions",
                    labels={"date": "Date", "mentions": y_label},
                    color_discrete_sequence=["#c96b9b"],
                )
                fig_mentions.update_layout(
                    margin=dict(l=0, r=0, t=10, b=0),
                    height=280,
                    xaxis_title="",
                    yaxis_title=y_label,
                )
                st.plotly_chart(fig_mentions, use_container_width=True)
                raw_total = mentions_series(mention_data, name)
                raw_total = raw_total[raw_total["date"] >= cutoff]
                st.caption(
                    f"Total mentions in window: **{int(raw_total['mentions'].sum()):,}**"
                )
            else:
                st.info("No mention data available. Run a scrape first.")

        # ── Stock price ──
        with col_right:
            st.subheader("Stock Price & Daily Return")
            if not stock_df.empty and "close" in stock_df.columns:
                stock_df["date"] = pd.to_datetime(stock_df["date"])
                cutoff = pd.Timestamp.now() - pd.Timedelta(days=history_days)
                sf = stock_df[stock_df["date"] >= cutoff].copy()

                fig_stock = go.Figure()
                fig_stock.add_trace(
                    go.Scatter(
                        x=sf["date"],
                        y=sf["close"],
                        mode="lines",
                        name="Close Price",
                        line=dict(color="#4a90d9", width=2),
                    )
                )
                fig_stock.update_layout(
                    margin=dict(l=0, r=0, t=10, b=0),
                    height=280,
                    xaxis_title="",
                    yaxis_title="Price (USD)",
                    legend=dict(x=0, y=1),
                )
                st.plotly_chart(fig_stock, use_container_width=True)

                latest_close = sf["close"].iloc[-1] if not sf.empty else None
                latest_ret = sf["daily_return"].iloc[-1] if not sf.empty else None
                if latest_close:
                    st.caption(
                        f"Latest close: **${latest_close:.2f}** | "
                        f"Last return: **{format_change(latest_ret)}**"
                    )
            else:
                st.info("No stock data. Ensure yfinance can reach Yahoo Finance.")

        # ── Regression scatter for selected lag ──
        st.subheader(f"Regression: Mentions → Stock Return (lag = {selected_lag} day(s))")

        lag_match = next((r for r in lag_results if r.lag_days == selected_lag), None)
        if lag_match and lag_match.n_observations >= 5:
            r = lag_match
            fig_reg = go.Figure()

            # Scatter
            fig_reg.add_trace(
                go.Scatter(
                    x=r.x_values,
                    y=r.y_values,
                    mode="markers",
                    text=r.dates,
                    marker=dict(color="#c96b9b", size=7, opacity=0.7),
                    name="Observations",
                    hovertemplate="Date: %{text}<br>Mentions: %{x}<br>Return: %{y:.2f}%<extra></extra>",
                )
            )

            # Regression line
            x_arr = np.array(r.x_values)
            y_hat = r.intercept + r.slope * x_arr
            fig_reg.add_trace(
                go.Scatter(
                    x=x_arr,
                    y=y_hat,
                    mode="lines",
                    name=f"OLS fit (R²={r.r_squared:.3f})",
                    line=dict(color="#4a90d9", width=2, dash="dash"),
                )
            )

            fig_reg.update_layout(
                height=320,
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="Mention Count",
                yaxis_title="Daily Stock Return (%)",
                legend=dict(x=0, y=1),
            )
            st.plotly_chart(fig_reg, use_container_width=True)

            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            metric_col1.metric("R²", f"{r.r_squared:.4f}")
            metric_col2.metric("Pearson r", f"{r.pearson_r:.4f}")
            metric_col3.metric("p-value", f"{r.p_value:.4f}")
            metric_col4.metric("Statistically Significant", sig_badge(r.is_significant))
            st.caption(
                f"Slope: {r.slope:.6f} | Intercept: {r.intercept:.4f} | "
                f"n = {r.n_observations} trading-day observations | "
                f"Direction: {r.direction}"
            )
            render_regression_breakdown(name, r, lag_results)
        else:
            st.info(
                "Not enough overlapping data points to run regression at this lag. "
                "Try refreshing data or selecting a different lag."
            )
        render_regression_suggestions(name)


# ─── Summary regression table ─────────────────────────────────────────────────

st.markdown("---")
st.markdown("## Regression Summary — All Products")
st.markdown(
    "Best-lag result per product (the lag with the highest R²). "
    "A significant result suggests that mention volume *correlates* with stock return — "
    "not necessarily that it *causes* it."
)

summary_df = reg.summary_table(all_reg_results)
# Filter to selected products only
summary_df = summary_df[summary_df["Product"].isin(selected_products)].copy()

# Force string conversion — .apply avoids any dtype inference issues
summary_df["Significant (p<0.05)"] = summary_df["Significant (p<0.05)"].apply(
    lambda x: "Yes" if x else "No"
)

# Explicitly declare every column type so Streamlit's Arrow renderer
# cannot auto-detect booleans and re-apply its own green checkbox styling
st.dataframe(
    summary_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Product": st.column_config.TextColumn("Product"),
        "Best Lag (days)": st.column_config.TextColumn("Best Lag (days)"),
        "R²": st.column_config.NumberColumn("R²", format="%.4f"),
        "Pearson r": st.column_config.NumberColumn("Pearson r", format="%.4f"),
        "p-value": st.column_config.NumberColumn("p-value", format="%.4f"),
        "Significant (p<0.05)": st.column_config.TextColumn("Significant (p<0.05)"),
        "Direction": st.column_config.TextColumn("Direction"),
        "Slope": st.column_config.NumberColumn("Slope", format="%.6f"),
        "Observations": st.column_config.NumberColumn("Observations"),
    },
)


# ─── Mention leaderboard ──────────────────────────────────────────────────────

st.markdown("---")
st.markdown("## Mention Leaderboard")

leaderboard_rows = []
for name in PRODUCT_NAMES:
    if name not in selected_products:
        continue
    total = total_mentions(mention_data, name)
    leaderboard_rows.append({"Product": name, "Total Mentions": total})

lb_df = pd.DataFrame(leaderboard_rows).sort_values("Total Mentions", ascending=False)

fig_lb = px.bar(
    lb_df,
    x="Product",
    y="Total Mentions",
    color="Total Mentions",
    color_continuous_scale="RdPu",
    labels={"Total Mentions": "Mentions"},
    text="Total Mentions",
)
fig_lb.update_traces(texttemplate="%{text:,}", textposition="outside")
fig_lb.update_layout(
    height=380,
    margin=dict(l=0, r=0, t=10, b=0),
    coloraxis_showscale=False,
    xaxis_title="",
)
st.plotly_chart(fig_lb, use_container_width=True)


# ─── Cross-product mention trend ──────────────────────────────────────────────

st.markdown("---")
st.markdown("## Cross-Product Mention Trends")

trend_frames = []
for name in PRODUCT_NAMES:
    if name not in selected_products:
        continue
    m = mentions_series(mention_data, name)
    if not m.empty:
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=history_days)
        m = m[m["date"] >= cutoff].copy()
        m["mentions"] = normalize_series(m["mentions"], normalization)
        m["product"] = name
        trend_frames.append(m)

trend_y_label = f"Mentions ({normalization})" if normalization != "None" else "Mentions"
if trend_frames:
    trend_df = pd.concat(trend_frames, ignore_index=True)
    fig_trend = px.line(
        trend_df,
        x="date",
        y="mentions",
        color="product",
        labels={"date": "Date", "mentions": trend_y_label, "product": "Product"},
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig_trend.update_layout(
        height=360,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="",
        legend_title="Product",
    )
    st.plotly_chart(fig_trend, use_container_width=True)
else:
    st.info("No mention trend data available.")


# ─── Footer ───────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    "Beauty Product Tracker | Data: Reddit (PRAW), RSS/News feeds, Yahoo Finance | "
    "Regression: OLS via scipy.stats.linregress | "
    "This is not financial advice."
)

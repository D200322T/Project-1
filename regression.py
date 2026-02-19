"""
regression.py — Regression analysis between social-media mention counts and stock returns.

For each product we build a joint daily DataFrame of:
    mentions_t  — total cross-platform mention count on day t
    return_t+lag — daily stock return (%) on day t + lag

Then we fit OLS (Ordinary Least Squares) regression and report:
    - intercept / slope
    - R² (coefficient of determination)
    - p-value for the slope coefficient
    - Pearson correlation coefficient
    - Best lag (the lag with highest |R²|)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

import config


@dataclass
class RegressionResult:
    product_name: str
    lag_days: int                  # 0 = same-day, 1 = mentions predict next day's return
    n_observations: int
    slope: float
    intercept: float
    r_squared: float
    pearson_r: float
    p_value: float
    std_err: float
    is_significant: bool           # p < 0.05
    direction: str                 # "positive" | "negative" | "none"
    # Raw arrays (for plotting)
    x_values: list[float] = field(default_factory=list)
    y_values: list[float] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)


def _build_joint_df(
    mention_counts: dict[str, int],
    stock_df: pd.DataFrame,
    lag: int,
) -> pd.DataFrame:
    """
    Create a DataFrame aligning mention counts on day t with stock return on day t+lag.

    Args:
        mention_counts: { "YYYY-MM-DD": count }
        stock_df: DataFrame with columns [date, daily_return, ...]
        lag: number of trading days to shift the return forward

    Returns:
        DataFrame with columns [date, mentions, stock_return]
    """
    if stock_df.empty or not mention_counts:
        return pd.DataFrame()

    mention_s = pd.Series(mention_counts, name="mentions")
    mention_s.index = pd.to_datetime(mention_s.index)
    mention_df = mention_s.reset_index()
    mention_df.columns = ["date", "mentions"]

    stock = stock_df[["date", "daily_return"]].copy()
    stock["date"] = pd.to_datetime(stock["date"])
    stock = stock.dropna(subset=["daily_return"])

    # Shift: align mention on trading day t with return on trading day t+lag
    stock["date_shifted"] = stock["date"] - pd.to_timedelta(lag, unit="D") * (-1)
    # Simpler: keep stock as-is, shift mentions back by lag days
    # i.e. stock return on date D corresponds to mentions on date D-lag
    stock = stock.rename(columns={"daily_return": "stock_return"})
    stock["mention_date"] = stock["date"] - pd.to_timedelta(lag, unit="D")

    merged = stock.merge(
        mention_df.rename(columns={"date": "mention_date"}),
        on="mention_date",
        how="inner",
    )

    merged = merged.dropna(subset=["mentions", "stock_return"])
    merged = merged[merged["mentions"] >= 0]
    merged["date_str"] = merged["date"].dt.strftime("%Y-%m-%d")
    return merged[["date_str", "mentions", "stock_return"]].reset_index(drop=True)


def run_regression(
    product_name: str,
    mention_counts: dict[str, int],
    stock_df: pd.DataFrame,
    lag: int = 0,
) -> Optional[RegressionResult]:
    """
    Fit OLS regression: stock_return ~ mentions (with given lag).

    Returns None if there are fewer than 5 usable observations.
    """
    df = _build_joint_df(mention_counts, stock_df, lag)
    if df.empty or len(df) < 5:
        return None

    x = df["mentions"].values.astype(float)
    y = df["stock_return"].values.astype(float)

    # Remove infinite values
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    dates = df["date_str"].values[mask]

    if len(x) < 5:
        return None

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r_squared = r_value ** 2

    if abs(r_value) < 1e-10:
        direction = "none"
    elif r_value > 0:
        direction = "positive"
    else:
        direction = "negative"

    return RegressionResult(
        product_name=product_name,
        lag_days=lag,
        n_observations=len(x),
        slope=float(slope),
        intercept=float(intercept),
        r_squared=float(r_squared),
        pearson_r=float(r_value),
        p_value=float(p_value),
        std_err=float(std_err),
        is_significant=p_value < 0.05,
        direction=direction,
        x_values=x.tolist(),
        y_values=y.tolist(),
        dates=dates.tolist(),
    )


def run_all_regressions(
    mention_data: dict[str, dict[str, int]],
    stock_data: dict[str, pd.DataFrame],
    lags: list[int] = config.LAG_DAYS,
) -> dict[str, list[RegressionResult]]:
    """
    Run regressions for every product × lag combination.

    Returns:
        { product_name: [RegressionResult(lag=0), RegressionResult(lag=1), ...] }
    """
    results: dict[str, list[RegressionResult]] = {}

    for product in config.PRODUCTS:
        name = product["name"]
        mentions = mention_data.get(name, {})
        stock_df = stock_data.get(name, pd.DataFrame())

        lag_results = []
        for lag in lags:
            result = run_regression(name, mentions, stock_df, lag)
            if result is not None:
                lag_results.append(result)

        results[name] = lag_results

    return results


def best_lag_result(lag_results: list[RegressionResult]) -> Optional[RegressionResult]:
    """Return the RegressionResult with the highest absolute R² across lags."""
    if not lag_results:
        return None
    return max(lag_results, key=lambda r: r.r_squared)


def summary_table(
    all_results: dict[str, list[RegressionResult]]
) -> pd.DataFrame:
    """
    Build a summary DataFrame (one row per product) using the best-lag result.

    Columns: product, best_lag, r_squared, pearson_r, p_value, significant,
             direction, slope, n_obs
    """
    rows = []
    for name, lag_results in all_results.items():
        best = best_lag_result(lag_results)
        if best is None:
            rows.append({
                "Product": name,
                "Best Lag (days)": "—",
                "R²": None,
                "Pearson r": None,
                "p-value": None,
                "Significant (p<0.05)": False,
                "Direction": "—",
                "Slope": None,
                "Observations": 0,
            })
        else:
            rows.append({
                "Product": name,
                "Best Lag (days)": best.lag_days,
                "R²": round(best.r_squared, 4),
                "Pearson r": round(best.pearson_r, 4),
                "p-value": round(best.p_value, 4),
                "Significant (p<0.05)": best.is_significant,
                "Direction": best.direction,
                "Slope": round(best.slope, 6),
                "Observations": best.n_observations,
            })

    # Put L'Oréal first to match the product list ordering in config
    product_order = [p["name"] for p in config.PRODUCTS]
    df = pd.DataFrame(rows)
    df["_order"] = df["Product"].apply(
        lambda x: product_order.index(x) if x in product_order else 999
    )
    return df.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)

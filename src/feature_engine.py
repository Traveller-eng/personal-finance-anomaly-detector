"""
feature_engine.py — Feature Engineering Pipeline
==================================================
PURPOSE (Interview Talking Point):
    Raw transaction data (date, amount, category, merchant) isn't enough for
    anomaly detection. We need to engineer features that capture BEHAVIORAL CONTEXT:
    - Is this amount unusual FOR THIS CATEGORY?
    - Is the user spending more than THEIR RECENT AVERAGE?
    - Is this transaction happening at an UNUSUAL TIME?

    These engineered features transform the problem from "find big numbers" to
    "find behavioral deviations" — which is what makes this system intelligent.

WHY THESE FEATURES:
    The Isolation Forest works by randomly splitting features to isolate points.
    If all we give it is raw amount, it only finds large transactions.
    But with z-scores, rolling averages, and category ratios, it can find
    MULTI-DIMENSIONAL anomalies: a normal-sized transaction at an unusual time
    in an unusual category is still an anomaly.
"""

import pandas as pd
import numpy as np


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master feature engineering pipeline.

    Takes cleaned data and adds ML-ready features.
    Returns DataFrame with all original columns plus engineered features.

    Interview Explanation:
        Each feature captures a different "dimension" of spending behavior:
        - amount_zscore: "How unusual is this amount within its category?"
        - rolling averages: "What's the user's recent spending trend?"
        - category ratios: "How does this compare to their baseline?"
        - frequency features: "How active has the user been recently?"
    """
    df = df.copy()

    # ─── 1. Category-level Z-Score ──────────────────────────────────────
    # "How many standard deviations is this amount from the category mean?"
    # Z-score > 2: unusual, > 3: very unusual
    df = _add_category_zscore(df)

    # ─── 2. Rolling Averages ────────────────────────────────────────────
    # "What's the user's spend trend over the last 7 and 30 days?"
    df = _add_rolling_averages(df)

    # ─── 3. Category Daily Ratio ────────────────────────────────────────
    # "How does this transaction compare to average daily spend in this category?"
    df = _add_category_ratio(df)

    # ─── 4. Transaction Frequency ───────────────────────────────────────
    # "How many transactions in the last 7 days?"
    df = _add_frequency_features(df)

    # ─── 5. Days Since Last Transaction ─────────────────────────────────
    # "How long since the user's last transaction? Long gaps may indicate unusual behavior"
    df = _add_days_since_last(df)

    # ─── 6. Amount Log Transform ────────────────────────────────────────
    # "Log-transform reduces the effect of extreme values on the model"
    df["amount_log"] = np.log1p(df["amount"])

    # ─── 7. Category Encoding ──────────────────────────────────────────
    # "Convert categorical data to numeric for the ML model"
    df = _add_category_encoding(df)

    # Fill any NaN from rolling calculations with 0
    feature_cols = get_feature_columns()
    df[feature_cols] = df[feature_cols].fillna(0)

    return df


def _add_category_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate z-score of amount within each category.

    Interview Deep-Dive:
        Z-score = (value - mean) / std_dev
        A z-score of 3 means the value is 3 standard deviations from the mean.
        We compute this PER CATEGORY because ₹5000 is normal for shopping
        but anomalous for food. This is category-relative normalization.
    """
    category_stats = df.groupby("category")["amount"].agg(["mean", "std"]).reset_index()
    category_stats.columns = ["category", "cat_mean", "cat_std"]
    category_stats["cat_std"] = category_stats["cat_std"].replace(0, 1)  # Avoid division by zero

    df = df.merge(category_stats, on="category", how="left")
    df["amount_zscore"] = (df["amount"] - df["cat_mean"]) / df["cat_std"]
    df = df.drop(columns=["cat_mean", "cat_std"])

    return df


def _add_rolling_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 7-day and 30-day rolling average spending.

    Interview Deep-Dive:
        Rolling averages capture TREND — if someone usually spends ₹500/day
        but suddenly their 7-day average is ₹2000/day, something changed.
        We sort by date first to ensure the rolling window is chronological.
    """
    # Compute daily total spend, then rolling average
    daily_spend = df.groupby("date")["amount"].sum().reset_index()
    daily_spend.columns = ["date", "daily_total"]
    daily_spend = daily_spend.sort_values("date")

    daily_spend["rolling_7d_avg"] = (
        daily_spend["daily_total"].rolling(window=7, min_periods=1).mean()
    )
    daily_spend["rolling_30d_avg"] = (
        daily_spend["daily_total"].rolling(window=30, min_periods=1).mean()
    )

    df = df.merge(daily_spend[["date", "rolling_7d_avg", "rolling_30d_avg"]], on="date", how="left")

    return df


def _add_category_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ratio of transaction amount to average daily spend in that category.

    Interview Deep-Dive:
        If the average daily food spend is ₹400, and today's food transaction
        is ₹2000, the ratio is 5.0 — meaning 5× the usual daily food spend.
        This normalizes across categories with very different spending levels.
    """
    # Average daily spend per category
    date_range_days = (df["date"].max() - df["date"].min()).days + 1
    category_daily_avg = (
        df.groupby("category")["amount"].sum() / date_range_days
    ).reset_index()
    category_daily_avg.columns = ["category", "cat_daily_avg"]
    category_daily_avg["cat_daily_avg"] = category_daily_avg["cat_daily_avg"].replace(0, 1)

    df = df.merge(category_daily_avg, on="category", how="left")
    df["category_daily_ratio"] = df["amount"] / df["cat_daily_avg"]
    df = df.drop(columns=["cat_daily_avg"])

    return df


def _add_frequency_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count transactions in the last 7 days for each date.

    Interview Deep-Dive:
        Transaction frequency matters — a user who normally makes 3 txns/day
        suddenly making 10 is suspicious, even if individual amounts are normal.
        This is COUNT-BASED anomaly detection capability.
    """
    daily_counts = df.groupby("date").size().reset_index(name="daily_txn_count")
    daily_counts = daily_counts.sort_values("date")
    daily_counts["txn_frequency_7d"] = (
        daily_counts["daily_txn_count"].rolling(window=7, min_periods=1).sum()
    )

    df = df.merge(daily_counts[["date", "txn_frequency_7d"]], on="date", how="left")

    return df


def _add_days_since_last(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate days since the previous transaction.

    Interview Deep-Dive:
        Unusual gaps in spending can indicate behavioral changes:
        - A user who transacts daily going silent for a week, then making
          a large purchase, could be a compromised account.
    """
    unique_dates = df["date"].drop_duplicates().sort_values()
    date_df = pd.DataFrame({"date": unique_dates})
    date_df["days_since_last_txn"] = date_df["date"].diff().dt.days.fillna(0)

    df = df.merge(date_df, on="date", how="left")

    return df


def _add_category_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label-encode categories for the ML model.

    Interview Deep-Dive:
        ML models need numeric inputs. Label encoding assigns each category
        an integer. We use this instead of one-hot encoding because Isolation
        Forest handles ordinal features well, and one-hot would increase
        dimensionality unnecessarily.
    """
    categories_sorted = sorted(df["category"].unique())
    cat_to_int = {cat: i for i, cat in enumerate(categories_sorted)}
    df["category_encoded"] = df["category"].map(cat_to_int)

    return df


def get_feature_columns() -> list[str]:
    """
    Return the list of feature columns used by the anomaly detector.
    Centralizing this ensures consistency between training and prediction.
    """
    return [
        "amount",
        "amount_log",
        "amount_zscore",
        "category_encoded",
        "day_of_week",
        "is_weekend",
        "rolling_7d_avg",
        "rolling_30d_avg",
        "category_daily_ratio",
        "txn_frequency_7d",
        "days_since_last_txn",
    ]

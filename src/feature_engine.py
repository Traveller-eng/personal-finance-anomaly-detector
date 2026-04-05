"""
feature_engine.py — Behavioral feature engineering
=================================================
Builds the context features that turn raw transactions into a behavior-aware
signal set for anomaly detection and insight generation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Master feature engineering pipeline."""
    if df is None:
        return pd.DataFrame(columns=get_feature_columns())

    if len(df) == 0:
        empty = df.copy()
        for column in get_feature_columns():
            if column not in empty.columns:
                empty[column] = []
        return empty

    df = df.copy().sort_values("date").reset_index(drop=True)
    if "merchant_normalized" not in df.columns:
        df["merchant_normalized"] = df["merchant"].astype(str).str.lower().str.strip()
    if "entity_type" not in df.columns:
        df["entity_type"] = "unknown"
    df = _add_daily_context(df)
    df = _add_category_history(df)
    df = _add_merchant_history(df)
    df = _add_flow_history(df)
    df = _add_days_since_last(df)
    df = _add_encodings(df)

    df["amount_log"] = np.log1p(df["amount"])
    df["weekend_flag"] = df.get("is_weekend", 0).astype(float)
    df["category_daily_ratio"] = df["amount"] / df["category_mean_prior"].replace(0, np.nan)
    df["category_daily_ratio"] = df["category_daily_ratio"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    numeric_columns = get_feature_columns()
    df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def _add_daily_context(df: pd.DataFrame) -> pd.DataFrame:
    """Add trailing daily-spend and daily-activity context."""
    daily = (
        df.groupby("date")
        .agg(daily_total=("amount", "sum"), daily_txn_count=("amount", "size"))
        .sort_index()
    )

    full_index = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_index, fill_value=0.0)
    daily.index.name = "date"
    daily = daily.reset_index()

    daily["rolling_7d_avg"] = daily["daily_total"].rolling(window=7, min_periods=1).mean()
    daily["rolling_mean_30d"] = daily["daily_total"].rolling(window=30, min_periods=1).mean()
    daily["rolling_30d_avg"] = daily["rolling_mean_30d"]
    daily["rolling_std_30d"] = daily["daily_total"].rolling(window=30, min_periods=2).std().fillna(0.0)
    daily["txn_frequency_7d"] = daily["daily_txn_count"].rolling(window=7, min_periods=1).sum()
    daily["transaction_frequency"] = daily["daily_txn_count"].rolling(window=30, min_periods=1).sum()
    daily["velocity_tx_per_day"] = daily["transaction_frequency"] / 30.0

    return df.merge(
        daily[
            [
                "date",
                "rolling_7d_avg",
                "rolling_mean_30d",
                "rolling_30d_avg",
                "rolling_std_30d",
                "txn_frequency_7d",
                "transaction_frequency",
                "velocity_tx_per_day",
            ]
        ],
        on="date",
        how="left",
    )


def _add_category_history(df: pd.DataFrame) -> pd.DataFrame:
    """Add category share and prior category spend baselines."""

    def enrich_group(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("date").copy()
        prior_amounts = group["amount"].shift(1)
        group["category_mean_prior"] = prior_amounts.expanding().mean()
        group["category_std_prior"] = prior_amounts.expanding().std().fillna(0.0)
        return group

    df = df.groupby("category", group_keys=False).apply(enrich_group)
    df = df.sort_values("date").reset_index(drop=True)

    df["amount_zscore"] = (
        (df["amount"] - df["category_mean_prior"]) / df["category_std_prior"].replace(0, np.nan)
    )
    df["amount_zscore"] = df["amount_zscore"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df["cumulative_amount"] = df["amount"].cumsum()
    df["category_cumulative_amount"] = df.groupby("category")["amount"].cumsum()
    df["category_share"] = df["category_cumulative_amount"] / df["cumulative_amount"].replace(0, np.nan)
    df["category_share"] = df["category_share"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def _add_merchant_history(df: pd.DataFrame) -> pd.DataFrame:
    """Add merchant familiarity features."""
    df["merchant_frequency"] = df.groupby("merchant_normalized").cumcount() + 1
    df["new_merchant_flag"] = (df["merchant_frequency"] == 1).astype(float)
    return df


def _add_flow_history(df: pd.DataFrame) -> pd.DataFrame:
    """Add cashflow- and transfer-aware features."""
    df["credit_flag"] = (df["type"] == "credit").astype(float)
    df["transfer_flag"] = df.get("is_transfer", False).astype(float)
    return df


def _add_days_since_last(df: pd.DataFrame) -> pd.DataFrame:
    """Add time since the previous transaction in days."""
    df["days_since_last_txn"] = (
        df["date"].diff().dt.total_seconds().div(86400).clip(lower=0).fillna(0.0)
    )
    return df


def _add_encodings(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical fields for the signal engine."""
    categories_sorted = sorted(df["category"].astype(str).unique())
    entity_types_sorted = sorted(df["entity_type"].astype(str).unique())

    cat_to_int = {category: index for index, category in enumerate(categories_sorted)}
    entity_to_int = {entity: index for index, entity in enumerate(entity_types_sorted)}

    df["category_encoded"] = df["category"].map(cat_to_int).astype(float)
    df["entity_type_encoded"] = df["entity_type"].map(entity_to_int).astype(float)
    return df


def get_feature_columns() -> list[str]:
    """Return the numeric features used by the signal engine."""
    return [
        "amount",
        "amount_log",
        "amount_zscore",
        "category_encoded",
        "entity_type_encoded",
        "day_of_week",
        "weekend_flag",
        "rolling_mean_30d",
        "rolling_std_30d",
        "transaction_frequency",
        "merchant_frequency",
        "category_share",
        "velocity_tx_per_day",
        "new_merchant_flag",
        "days_since_last_txn",
        "credit_flag",
        "transfer_flag",
    ]

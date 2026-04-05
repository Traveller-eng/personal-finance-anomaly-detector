"""
preprocessor.py — Canonical transaction cleaning and validation
==============================================================
Transforms raw transaction data into a strict, behavior-aware schema that is
safe for downstream classification, profiling, detection, and UI rendering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.entity_resolution import resolve_transaction_entities
from src.transaction_schema import (
    CATEGORY_PLACEHOLDERS,
    TRANSFER_CATEGORIES,
    build_empty_transaction_frame,
    ensure_schema_columns,
    normalize_confidence,
    normalize_source,
)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master cleaning pipeline.

    The output always conforms to the strict transaction schema, even when the
    input is sparse or malformed.
    """
    if df is None:
        return build_empty_transaction_frame()

    if len(df) == 0:
        return build_empty_transaction_frame(list(df.columns))

    df = df.copy()
    df = _ensure_minimum_columns(df)
    df = _parse_dates(df)
    df = _clean_amounts(df)
    df = _clean_types(df)
    df = _normalize_categories(df)
    df = _clean_merchants(df)
    df = _handle_missing(df)
    if len(df) == 0:
        return ensure_schema_columns(build_empty_transaction_frame(list(df.columns)))

    parse_confidence = float(df["parse_confidence"].iloc[0]) if "parse_confidence" in df.columns else 1.0
    df = _initialize_schema(df, parse_confidence=parse_confidence)
    df = _validate_transactions(df)
    if len(df) == 0:
        return ensure_schema_columns(build_empty_transaction_frame(list(df.columns)), parse_confidence=parse_confidence)
    df = _synchronize_transfer_metadata(df)
    df = _add_temporal_features(df)
    df = df.sort_values("date").reset_index(drop=True)
    df = df.drop_duplicates()
    df = ensure_schema_columns(df, parse_confidence=parse_confidence)
    return df


def _ensure_minimum_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add the minimum columns required to clean the data safely."""
    df = df.copy()
    defaults = {
        "date": pd.NaT,
        "amount": np.nan,
        "type": "debit",
        "merchant": "Unknown",
        "category": "uncategorized",
    }
    for column, default in defaults.items():
        if column not in df.columns:
            df[column] = default
    if "parse_confidence" not in df.columns:
        df["parse_confidence"] = 1.0
    return df


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the date column to datetime and drop unparseable rows."""
    if "date" not in df.columns:
        df["date"] = pd.NaT

    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)

    null_dates = df["date"].isnull().sum()
    if null_dates > 0:
        print(f"  [Preprocessor] dropped {null_dates} rows with unparseable dates")
        df = df.dropna(subset=["date"])

    return df


def _clean_amounts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the amount column into a positive float.

    Signed values from source files are converted to absolute values because
    transaction direction is represented in the separate `type` column.
    """
    if "amount" not in df.columns:
        df["amount"] = np.nan

    if df["amount"].dtype == object:
        cleaned = (
            df["amount"]
            .astype(str)
            .str.replace(r"[₹$€£,\s]", "", regex=True)
            .str.strip()
        )
        cleaned = cleaned.apply(
            lambda value: f"-{value[1:-1]}"
            if isinstance(value, str) and value.startswith("(") and value.endswith(")")
            else value
        )
        df["amount"] = cleaned

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").abs()

    invalid = df["amount"].isnull() | (df["amount"] <= 0)
    if invalid.sum() > 0:
        print(f"  [Preprocessor] dropped {invalid.sum()} rows with invalid amounts")
        df = df[~invalid]

    return df


def _clean_types(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize transaction direction into `debit` or `credit`."""
    mapping = {
        "paid": "debit",
        "payment": "debit",
        "withdrawal": "debit",
        "purchase": "debit",
        "dr": "debit",
        "expense": "debit",
        "received": "credit",
        "deposit": "credit",
        "salary": "credit",
        "refund": "credit",
        "cr": "credit",
        "income": "credit",
    }

    df["type"] = df["type"].fillna("debit").astype(str).str.lower().str.strip()
    df["type"] = df["type"].replace(mapping)
    df.loc[~df["type"].isin({"debit", "credit"}), "type"] = "debit"
    return df


def _normalize_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize category labels while preserving placeholders for later classification."""
    df["category"] = df["category"].where(df["category"].notna(), other="uncategorized")
    df["category"] = df["category"].astype(str).str.strip().str.lower()

    category_map = {
        "groceries": "food",
        "dining": "food",
        "restaurants": "food",
        "taxi": "transport",
        "cab": "transport",
        "travel": "transport",
        "utilities": "bills",
        "subscription": "entertainment",
        "subscriptions": "entertainment",
        "medical": "health",
        "medicine": "health",
        "pharmacy": "health",
        "course": "education",
        "courses": "education",
        "housing": "rent",
        "salary": "income_transfer",
        "transfer": "personal_transfer",
    }
    df["category"] = df["category"].replace(category_map)
    df.loc[df["category"].isin({"", "nan", "none"}), "category"] = "uncategorized"

    return df


def _clean_merchants(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize merchant strings without destroying user-provided formatting."""
    df["merchant"] = df["merchant"].astype(str).str.strip()
    df.loc[df["merchant"].isin({"", "nan", "none"}), "merchant"] = "Unknown"
    return df


def _handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill remaining soft-missing values."""
    df["category"] = df["category"].fillna("uncategorized")
    df["merchant"] = df["merchant"].fillna("Unknown")
    df["type"] = df["type"].fillna("debit")
    return df


def _initialize_schema(df: pd.DataFrame, parse_confidence: float = 1.0) -> pd.DataFrame:
    """Add canonical schema fields and baseline metadata."""
    df = ensure_schema_columns(df, parse_confidence=parse_confidence)
    df = resolve_transaction_entities(df)

    provided_category = ~df["category"].isin(CATEGORY_PLACEHOLDERS)
    df.loc[provided_category, "source"] = df.loc[provided_category, "source"].apply(lambda value: normalize_source(value, "user"))
    df.loc[~provided_category, "source"] = df.loc[~provided_category, "source"].apply(lambda value: normalize_source(value, "rule"))

    df.loc[provided_category, "category_confidence"] = df.loc[provided_category, "category_confidence"].apply(
        lambda value: normalize_confidence(value, "manual")
    )
    df.loc[~provided_category, "category_confidence"] = df.loc[~provided_category, "category_confidence"].apply(
        lambda value: normalize_confidence(value, "low")
    )

    return df


def _validate_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Strict validation layer: keep only rows with a valid date and positive amount."""
    if len(df) == 0:
        return build_empty_transaction_frame(list(df.columns))

    validated = df.dropna(subset=["date", "amount"])
    validated = validated[validated["amount"] > 0]
    return validated


def _synchronize_transfer_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Keep transfer categories and transfer flags aligned."""
    df = df.copy()
    transfer_mask = df["category"].isin(TRANSFER_CATEGORIES) | df["is_transfer"]
    df.loc[transfer_mask, "is_transfer"] = True

    df.loc[transfer_mask & (df["type"] == "credit"), "category"] = "income_transfer"
    df.loc[transfer_mask & (df["type"] != "credit"), "category"] = "personal_transfer"
    return df


def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based derived columns shared by the downstream pipeline."""
    if len(df) == 0:
        return df

    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_name"] = df["date"].dt.day_name()
    df["month"] = df["date"].dt.month
    df["week_number"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["day_of_month"] = df["date"].dt.day
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    return df


def get_cleaning_report(original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> dict:
    """Generate a summary report comparing original vs cleaned data."""
    original_rows = len(original_df) if original_df is not None else 0
    cleaned_rows = len(cleaned_df)
    drop_rate = 0.0 if original_rows == 0 else (1 - cleaned_rows / original_rows) * 100

    return {
        "original_rows": original_rows,
        "cleaned_rows": cleaned_rows,
        "rows_dropped": original_rows - cleaned_rows,
        "drop_rate": f"{drop_rate:.1f}%",
        "columns_added": [col for col in cleaned_df.columns if original_df is None or col not in original_df.columns],
        "null_remaining": int(cleaned_df.isnull().sum().sum()),
        "schema_columns": list(cleaned_df.columns),
    }

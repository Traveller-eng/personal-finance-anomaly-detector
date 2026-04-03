"""
preprocessor.py — Data Cleaning & Transformation
==================================================
PURPOSE:
    Transforms raw transaction data into a clean, ML-ready format.
    This is where we handle the "real-world messiness" — missing values,
    inconsistent formats, type coercion, and derived temporal features.

KEY DESIGN DECISIONS:
    1. We NEVER modify the original DataFrame — all operations return new copies.
       This makes debugging easier (you can inspect intermediate states).
    2. Temporal features (day_of_week, is_weekend, etc.) are added here because
       they're universal — every downstream module needs them.
    3. Category normalization uses lowercase + strip to handle "Food", "food ", "FOOD".
"""

import pandas as pd
import numpy as np


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master cleaning pipeline. Runs all cleaning steps in order.



    Parameters:
        df: Raw DataFrame from data_loader

    Returns:
        Cleaned DataFrame ready for feature engineering
    """
    df = df.copy()

    # Step 1: Parse dates
    df = _parse_dates(df)

    # Step 2: Clean amounts
    df = _clean_amounts(df)

    # Step 3: Normalize categories
    df = _normalize_categories(df)

    # Step 4: Clean merchants
    df = _clean_merchants(df)

    # Step 5: Handle missing values
    df = _handle_missing(df)

    # Step 6: Add temporal features
    df = _add_temporal_features(df)

    # Step 7: Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    # Step 8: Remove exact duplicates
    df = df.drop_duplicates()

    return df


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert date column to datetime.
    Handles multiple formats: YYYY-MM-DD, DD/MM/YYYY, MM-DD-YYYY, etc.
    """
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Drop rows where date couldn't be parsed
    null_dates = df["date"].isnull().sum()
    if null_dates > 0:
        print(f"  ⚠️  Dropped {null_dates} rows with unparseable dates")
        df = df.dropna(subset=["date"])

    return df


def _clean_amounts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the amount column:
    - Remove currency symbols (₹, $, €, etc.)
    - Remove commas in numbers (1,200 → 1200)
    - Convert to float
    - Remove zero/negative amounts
    """
    if df["amount"].dtype == object:
        # Remove currency symbols and commas
        df["amount"] = (
            df["amount"]
            .astype(str)
            .str.replace(r"[₹$€£,]", "", regex=True)
            .str.strip()
        )

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    # Drop invalid amounts
    invalid = df["amount"].isnull() | (df["amount"] <= 0)
    if invalid.sum() > 0:
        print(f"  ⚠️  Dropped {invalid.sum()} rows with invalid amounts (null/zero/negative)")
        df = df[~invalid]

    return df


def _normalize_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize category names:
    - Lowercase
    - Strip whitespace
    - Map common variations
    """
    # Replace None/NaN BEFORE astype(str) so they don't become the literal string 'none' or 'nan'
    df["category"] = df["category"].where(df["category"].notna(), other="uncategorized")
    df["category"] = df["category"].astype(str).str.strip().str.lower()

    # Map common category variations
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
    }
    df["category"] = df["category"].replace(category_map)

    return df


def _clean_merchants(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize merchant names."""
    df["merchant"] = df["merchant"].astype(str).str.strip().str.title()
    return df


def _handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle remaining missing values.

    Strategy:
        - date: Already handled (dropped in _parse_dates)
        - amount: Already handled (dropped in _clean_amounts)
        - category: Fill with 'uncategorized' (don't lose the transaction)
        - merchant: Fill with 'Unknown' (merchant is informational, not critical)
    """
    df["category"] = df["category"].fillna("uncategorized")
    df["merchant"] = df["merchant"].fillna("Unknown")

    return df


def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based derived columns.


    """
    df["day_of_week"] = df["date"].dt.dayofweek          # 0=Monday, 6=Sunday
    df["day_name"] = df["date"].dt.day_name()              # "Monday", "Tuesday", ...
    df["month"] = df["date"].dt.month                      # 1-12
    df["week_number"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)  # 1 if weekend
    df["day_of_month"] = df["date"].dt.day
    df["is_month_end"] = (df["date"].dt.is_month_end).astype(int)
    df["is_month_start"] = (df["date"].dt.is_month_start).astype(int)

    return df


def get_cleaning_report(original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> dict:
    """
    Generate a report comparing original vs cleaned data.
    Useful for the dashboard to show data quality.
    """
    return {
        "original_rows": len(original_df),
        "cleaned_rows": len(cleaned_df),
        "rows_dropped": len(original_df) - len(cleaned_df),
        "drop_rate": f"{(1 - len(cleaned_df) / len(original_df)) * 100:.1f}%",
        "columns_added": [
            col for col in cleaned_df.columns if col not in original_df.columns
        ],
        "null_remaining": cleaned_df.isnull().sum().sum(),
    }

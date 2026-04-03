"""
data_loader.py — Transaction Data Ingestion
=============================================
PURPOSE (Interview Talking Point):
    Handles the first step of any ML pipeline: getting data in reliably.
    This module loads CSV files with flexible column mapping and validates
    that required fields exist with correct types.

WHY A SEPARATE MODULE:
    Separation of concerns — the loader only cares about "can I read this file?"
    It doesn't care about cleaning or feature engineering. This makes the pipeline
    modular and each piece independently testable.
"""

import pandas as pd
import os


# ─── Required columns and their expected types ────────────────────────────────
REQUIRED_COLUMNS = ["date", "amount", "category", "merchant"]

COLUMN_ALIASES = {
    # Common alternative names → our standard names
    "transaction_date": "date",
    "txn_date": "date",
    "Date": "date",
    "transaction_amount": "amount",
    "amt": "amount",
    "Amount": "amount",
    "Category": "category",
    "cat": "category",
    "type": "category",
    "Merchant": "merchant",
    "merchant_name": "merchant",
    "vendor": "merchant",
    "description": "merchant",
}


def load_csv(filepath: str) -> pd.DataFrame:
    """
    Load a CSV file and standardize column names.

    Interview Explanation:
        Real-world CSVs come in many formats — different column names, encodings,
        delimiters. This function handles common variations so the rest of the
        pipeline can assume clean, standardized column names.

    Parameters:
        filepath: Path to the CSV file

    Returns:
        DataFrame with standardized column names

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns are missing
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Transaction file not found: {filepath}")

    # Try UTF-8 first, fall back to Latin-1 for special characters
    try:
        df = pd.read_csv(filepath, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding="latin-1")

    # Strip whitespace from column names
    df.columns = df.columns.str.strip().str.lower()

    # Apply column aliases
    rename_map = {}
    for col in df.columns:
        if col in COLUMN_ALIASES:
            rename_map[col] = COLUMN_ALIASES[col]
    if rename_map:
        df = df.rename(columns=rename_map)

    # Validate required columns exist
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}. "
            f"Expected: {REQUIRED_COLUMNS}"
        )

    return df


def load_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize an already-loaded DataFrame (e.g., from Streamlit upload).
    Same validation logic as load_csv but works on in-memory data.
    """
    df.columns = df.columns.str.strip().str.lower()

    rename_map = {}
    for col in df.columns:
        if col in COLUMN_ALIASES:
            rename_map[col] = COLUMN_ALIASES[col]
    if rename_map:
        df = df.rename(columns=rename_map)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}. "
            f"Expected: {REQUIRED_COLUMNS}"
        )

    return df


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Return a summary dict for quick data quality checks.
    Used by the dashboard to show data overview.
    """
    return {
        "total_rows": len(df),
        "columns": list(df.columns),
        "date_range": {
            "min": str(df["date"].min()) if "date" in df.columns else None,
            "max": str(df["date"].max()) if "date" in df.columns else None,
        },
        "null_counts": df.isnull().sum().to_dict(),
        "categories": df["category"].nunique() if "category" in df.columns else 0,
        "merchants": df["merchant"].nunique() if "merchant" in df.columns else 0,
    }

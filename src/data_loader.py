"""
data_loader.py — Transaction Data Ingestion
=============================================
PURPOSE:
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


def parse_generic_csv(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        if col in COLUMN_ALIASES:
            rename_map[col] = COLUMN_ALIASES[col]
    if rename_map:
        df = df.rename(columns=rename_map)
    return df

def parse_mint_csv(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {"date": "date", "description": "merchant", "category": "category", "amount": "amount"}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    if "transaction type" in df.columns:
        df = df[df["transaction type"].astype(str).str.lower() == "debit"]
    return df

def parse_ynab_csv(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {"date": "date", "payee": "merchant", "category group/category": "category", "outflow": "amount"}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')
        df = df.dropna(subset=["amount"])
        df = df[df["amount"] > 0]
    return df

def parse_chase_csv(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {"posting date": "date", "description": "merchant", "amount": "amount"}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors='coerce')
        df = df[df["amount"] < 0]
        df["amount"] = df["amount"].abs()
    if "category" not in df.columns:
        df["category"] = "Uncategorized"
    return df

PARSERS = {
    "Generic (Default)": parse_generic_csv,
    "Mint": parse_mint_csv,
    "YNAB": parse_ynab_csv,
    "Chase": parse_chase_csv,
}

def load_csv(filepath: str, parser_type: str = "Generic (Default)") -> pd.DataFrame:
    """
    Load a CSV file and standardize column names.



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

    # Apply parser logic
    parser_func = PARSERS.get(parser_type, parse_generic_csv)
    df = parser_func(df)

    # Validate required columns exist
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}. "
            f"Expected: {REQUIRED_COLUMNS}"
        )

    return df


def load_from_dataframe(df: pd.DataFrame, parser_type: str = "Generic (Default)") -> pd.DataFrame:
    """
    Standardize an already-loaded DataFrame (e.g., from Streamlit upload).
    Same validation logic as load_csv but works on in-memory data.
    """
    df.columns = df.columns.str.strip().str.lower()

    # Apply parser logic
    parser_func = PARSERS.get(parser_type, parse_generic_csv)
    df = parser_func(df)

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

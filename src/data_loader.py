"""
data_loader.py — Universal Transaction Data Ingestion
======================================================
PURPOSE:
    Schema-adaptive CSV loader that handles ANY bank format.
    Uses fuzzy column mapping to detect date/amount/merchant/category
    columns from real-world messy headers. NEVER crashes — returns
    clean data + warnings list.

DESIGN:
    1. Fuzzy column resolution via scored alias dictionary
    2. Missing columns auto-generated with sensible defaults
    3. Currency symbols / commas stripped before numeric conversion
    4. Returns (DataFrame, List[str]) — data + warning messages
"""

import pandas as pd
import numpy as np
import os
import re
import logging

logger = logging.getLogger("pfad.data_loader")

# ─── Canonical columns the pipeline requires ─────────────────────────────────
REQUIRED_COLUMNS = ["date", "amount", "category", "merchant", "type"]

# ─── Fuzzy alias map ──────────────────────────────────────────────────────────
# Each canonical column has a list of known alternative names (lowercase).
# Matched by exact hit first, then substring containment.
COLUMN_ALIASES = {
    "date": [
        "date", "txn date", "txn_date", "transaction_date", "transaction date",
        "posting date", "posting_date", "timestamp", "trans date", "value date",
        "value_date", "created_at", "created at", "datetime",
    ],
    "amount": [
        "amount", "amount (inr)", "amount(inr)", "amount_inr", "amt",
        "transaction_amount", "transaction amount", "value", "debit",
        "debit amount", "debit_amount", "credit", "outflow", "inflow",
        "total", "sum", "price",
    ],
    "merchant": [
        "merchant", "party", "payee", "description", "merchant_name",
        "merchant name", "vendor", "beneficiary", "beneficiary name",
        "narration", "particulars", "remarks", "details", "to/from",
        "transaction details", "transaction_details", "name",
    ],
    "category": [
        "category", "cat", "category group/category", "expense type", "expense_type",
        "label", "tag", "group", "class",
    ],
    "type": [
        "type", "transaction type", "transaction_type", "txn type", "cr/dr",
        "debit/credit", "transaction_type",
    ],
}


# ─── Core: Fuzzy Column Mapper ────────────────────────────────────────────────

def auto_map_columns(df: pd.DataFrame) -> tuple[dict, list[str]]:
    """
    Automatically detect and map DataFrame columns to canonical names
    using a multi-pass fuzzy matching strategy.

    Returns:
        rename_map: dict mapping original column name → canonical name
        warnings:   list of human-readable warning strings
    """
    warnings = []
    rename_map = {}
    used_columns = set()  # Prevent double-mapping

    incoming_cols = [str(c).strip().lower() for c in df.columns]
    original_cols = list(df.columns)

    for canonical, aliases in COLUMN_ALIASES.items():
        matched_col = None

        # Pass 1: Exact match
        for i, col_lower in enumerate(incoming_cols):
            if col_lower in aliases and original_cols[i] not in used_columns:
                matched_col = original_cols[i]
                break

        # Pass 2: Substring containment ("amount (inr)" matches "amount")
        if matched_col is None:
            for i, col_lower in enumerate(incoming_cols):
                if original_cols[i] in used_columns:
                    continue
                for alias in aliases:
                    if alias in col_lower or col_lower in alias:
                        matched_col = original_cols[i]
                        break
                if matched_col:
                    break

        if matched_col:
            used_columns.add(matched_col)
            if str(matched_col).strip().lower() != canonical:
                rename_map[matched_col] = canonical
                warnings.append(
                    f"✅ Mapped '{matched_col}' → '{canonical}'"
                )
            else:
                warnings.append(f"✅ Found exact column: '{canonical}'")
        else:
            warnings.append(
                f"⚠️ Column '{canonical}' not found in data"
            )

    logger.info("Column mapping result: %s", rename_map)
    return rename_map, warnings


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Clean a raw DataFrame after column mapping:
    - Strip currency symbols (₹, $, €, £) and commas from amount
    - Convert amount to float
    - Parse date safely
    - Fill missing merchant / category with defaults

    Returns:
        cleaned DataFrame, list of warning strings
    """
    warnings = []
    df = df.copy()

    # ── Clean Amount ──────────────────────────────────────────────────────
    if "amount" in df.columns:
        df["amount"] = (
            df["amount"]
            .astype(str)
            .str.replace(r"[₹$€£,\s]", "", regex=True)
            .str.strip()
        )
        # Handle parenthesized negatives: (500) → -500
        df["amount"] = df["amount"].apply(
            lambda x: f"-{x[1:-1]}" if isinstance(x, str) and x.startswith("(") and x.endswith(")") else x
        )
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

        invalid_count = df["amount"].isna().sum()
        if invalid_count > 0:
            warnings.append(
                f"⚠️ {invalid_count} rows had unparseable amounts — dropped"
            )
            df = df.dropna(subset=["amount"])

        # Take absolute values (handle credit/debit signs)
        df["amount"] = df["amount"].abs()

        # Remove zero amounts
        zero_count = (df["amount"] == 0).sum()
        if zero_count > 0:
            warnings.append(f"⚠️ {zero_count} zero-amount rows removed")
            df = df[df["amount"] > 0]

    # ── Map Type (debit / credit) ─────────────────────────────────────────
    if "type" in df.columns:
        # Standardize "Paid"/"Received" to "debit"/"credit"
        df["type"] = df["type"].astype(str).str.lower().str.strip()
        type_mapping = {
            "paid": "debit",
            "received": "credit",
            "dr": "debit",
            "cr": "credit",
            "withdrawal": "debit",
            "deposit": "credit"
        }
        # Replace based on mapping, keep original if not in mapping
        df["type"] = df["type"].replace(type_mapping)
        # Ensure it falls back to 'debit' if not recognized (or null)
        df.loc[~df["type"].isin(["debit", "credit"]), "type"] = "debit"
    else:
        df["type"] = "debit"
        warnings.append("ℹ️ Type column missing — auto-generated as 'debit'")

    # ── Parse Date ────────────────────────────────────────────────────────
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
        null_dates = df["date"].isna().sum()
        if null_dates > 0:
            warnings.append(
                f"⚠️ {null_dates} rows had unparseable dates — dropped"
            )
            df = df.dropna(subset=["date"])

    # ── Handle Missing Category ───────────────────────────────────────────
    if "category" not in df.columns:
        df["category"] = "uncategorized"
        warnings.append(
            "ℹ️ Category column missing — auto-generated as 'uncategorized'"
        )
    else:
        empty_cats = df["category"].isna().sum()
        if empty_cats > 0:
            df["category"] = df["category"].fillna("uncategorized")
            warnings.append(
                f"ℹ️ {empty_cats} empty categories filled with 'uncategorized'"
            )

    # ── Handle Missing Merchant ───────────────────────────────────────────
    if "merchant" not in df.columns:
        df["merchant"] = "Unknown"
        warnings.append(
            "ℹ️ Merchant column missing — auto-generated as 'Unknown'"
        )
    else:
        empty_merchants = df["merchant"].isna().sum()
        if empty_merchants > 0:
            df["merchant"] = df["merchant"].fillna("Unknown")
            warnings.append(
                f"ℹ️ {empty_merchants} empty merchants filled with 'Unknown'"
            )

    # ── Handle Missing Date ───────────────────────────────────────────────
    if "date" not in df.columns:
        df["date"] = pd.Timestamp.now()
        warnings.append(
            "⚠️ Date column missing — defaulted to today's date"
        )

    # ── Handle Missing Amount ─────────────────────────────────────────────
    if "amount" not in df.columns:
        warnings.append(
            "❌ Amount column could not be detected — this is critical"
        )

    return df, warnings


# ─── Bank-specific parsers (kept for backward compatibility) ──────────────────

def parse_mint_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Mint-specific column handling."""
    if "transaction type" in [c.lower() for c in df.columns]:
        col_name = [c for c in df.columns if c.lower() == "transaction type"][0]
        df = df[df[col_name].astype(str).str.lower() == "debit"]
    return df


def parse_ynab_csv(df: pd.DataFrame) -> pd.DataFrame:
    """YNAB-specific: use outflow column, filter positives."""
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(
            df["amount"].astype(str).str.replace(r'[^\d.]', '', regex=True),
            errors='coerce'
        )
        df = df.dropna(subset=["amount"])
        df = df[df["amount"] > 0]
    return df


def parse_chase_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Chase-specific: negative amounts are debits."""
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors='coerce')
        df = df[df["amount"] < 0]
        df["amount"] = df["amount"].abs()
    return df


PARSERS = {
    "Generic (Default)": None,  # Uses universal auto-mapper
    "Mint": parse_mint_csv,
    "YNAB": parse_ynab_csv,
    "Chase": parse_chase_csv,
}


# ─── Public API ───────────────────────────────────────────────────────────────

def load_from_dataframe(
    df: pd.DataFrame,
    parser_type: str = "Generic (Default)"
) -> tuple[pd.DataFrame, list[str]]:
    """
    Standardize an already-loaded DataFrame (e.g., from Streamlit upload).
    Uses universal fuzzy column mapping. NEVER raises — returns warnings.

    Returns:
        (cleaned_dataframe, list_of_warning_strings)
    """
    all_warnings = []

    # Guard: empty DataFrame
    if df is None or len(df.columns) == 0:
        all_warnings.append("⚠️ Empty DataFrame received — no data to process")
        return pd.DataFrame(columns=["date", "amount", "category", "merchant"]), all_warnings

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # Step 1: Auto-map columns
    rename_map, map_warnings = auto_map_columns(df)
    all_warnings.extend(map_warnings)

    if rename_map:
        df = df.rename(columns=rename_map)
        logger.info("Renamed columns: %s", rename_map)

    # Step 2: Apply bank-specific parser if not Generic
    parser_func = PARSERS.get(parser_type)
    if parser_func is not None:
        df = parser_func(df)

    # Step 3: Clean data (currency, dates, missing values)
    df, clean_warnings = clean_dataframe(df)
    all_warnings.extend(clean_warnings)

    # Step 4: Ensure all required columns exist
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            all_warnings.append(
                f"❌ Critical: '{col}' column could not be resolved"
            )

    # Step 5: Keep only relevant columns + any extras
    keep_cols = [c for c in REQUIRED_COLUMNS if c in df.columns]
    extra_cols = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    df = df[keep_cols + extra_cols]

    # Log summary
    logger.info(
        "Data loaded: %d rows, columns: %s, warnings: %d",
        len(df), list(df.columns), len(all_warnings)
    )
    for w in all_warnings:
        try:
            print(f"  [DataLoader] {w}")
        except UnicodeEncodeError:
            print(f"  [DataLoader] {w.encode('ascii', 'replace').decode()}")

    return df, all_warnings


def load_csv(
    filepath: str,
    parser_type: str = "Generic (Default)"
) -> tuple[pd.DataFrame, list[str]]:
    """
    Load a CSV file and standardize column names.
    NEVER raises ValueError — returns clean data + warnings.

    Returns:
        (cleaned_dataframe, list_of_warning_strings)
    """
    warnings = []

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Transaction file not found: {filepath}")

    # Try UTF-8 first, fall back to Latin-1
    try:
        df = pd.read_csv(filepath, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding="latin-1")
        warnings.append("ℹ️ File read with Latin-1 encoding (non-UTF8 characters detected)")

    df, load_warnings = load_from_dataframe(df, parser_type=parser_type)
    warnings.extend(load_warnings)

    return df, warnings


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

"""
transaction_schema.py — Canonical transaction schema helpers
===========================================================
Centralizes the strict transaction contract used across ingestion,
classification, feature engineering, and UI layers.
"""

from __future__ import annotations

import pandas as pd


STRICT_TRANSACTION_SCHEMA = {
    "date": "datetime64[ns]",
    "amount": "float64",
    "type": "object",
    "merchant": "object",
    "merchant_normalized": "object",
    "entity_type": "object",
    "category": "object",
    "category_confidence": "object",
    "source": "object",
    "is_transfer": "bool",
}

VALID_TRANSACTION_TYPES = {"debit", "credit"}
VALID_ENTITY_TYPES = {"person", "business", "unknown"}
VALID_CATEGORY_CONFIDENCE = {"high", "low", "manual"}
VALID_SOURCES = {"rule", "ml", "user"}
CATEGORY_PLACEHOLDERS = {"", "nan", "none", "unknown", "uncategorized", "other", "others"}
TRANSFER_CATEGORIES = {"income_transfer", "personal_transfer"}


def build_empty_transaction_frame(extra_columns: list[str] | None = None) -> pd.DataFrame:
    """Return an empty DataFrame with the canonical schema columns present."""
    columns = list(STRICT_TRANSACTION_SCHEMA.keys())
    for col in extra_columns or []:
        if col not in columns:
            columns.append(col)
    return pd.DataFrame(columns=columns)


def normalize_category_value(value) -> str:
    """Normalize categories to lowercase canonical values."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "uncategorized"

    text = str(value).strip().lower()
    return text or "uncategorized"


def normalize_confidence(value, fallback: str = "low") -> str:
    """Normalize category confidence into the strict schema values."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return fallback

    text = str(value).strip().lower()
    mapping = {
        "high": "high",
        "medium": "low",
        "low": "low",
        "manual": "manual",
    }
    return mapping.get(text, fallback)


def normalize_source(value, fallback: str = "rule") -> str:
    """Normalize legacy source values into rule/ml/user."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return fallback

    text = str(value).strip().lower()
    mapping = {
        "rule": "rule",
        "default": "rule",
        "original": "user",
        "user": "user",
        "manual": "user",
        "model": "ml",
        "ml": "ml",
    }
    return mapping.get(text, fallback)


def ensure_schema_columns(df: pd.DataFrame, parse_confidence: float = 1.0) -> pd.DataFrame:
    """
    Ensure the canonical schema columns exist.

    The function is intentionally light-touch: it adds missing columns and
    normalizes a few enum-style fields without mutating unrelated columns.
    """
    df = df.copy()

    defaults = {
        "date": pd.NaT,
        "amount": 0.0,
        "type": "debit",
        "merchant": "Unknown",
        "merchant_normalized": "unknown",
        "entity_type": "unknown",
        "category": "uncategorized",
        "category_confidence": "low",
        "source": "rule",
        "is_transfer": False,
        "parse_confidence": float(parse_confidence),
    }

    for column, default_value in defaults.items():
        if column not in df.columns:
            df[column] = default_value

    df["type"] = df["type"].astype(str).str.lower().str.strip()
    df.loc[~df["type"].isin(VALID_TRANSACTION_TYPES), "type"] = "debit"

    df["entity_type"] = df["entity_type"].astype(str).str.lower().str.strip()
    df.loc[~df["entity_type"].isin(VALID_ENTITY_TYPES), "entity_type"] = "unknown"

    df["category"] = df["category"].apply(normalize_category_value)
    df["category_confidence"] = df["category_confidence"].apply(normalize_confidence)
    df["source"] = df["source"].apply(normalize_source)
    df["is_transfer"] = df["is_transfer"].fillna(False).astype(bool)
    df["parse_confidence"] = pd.to_numeric(df["parse_confidence"], errors="coerce").fillna(parse_confidence)

    if "category_source" in df.columns:
        df["category_source"] = df["source"]
    else:
        df["category_source"] = df["source"]

    return df

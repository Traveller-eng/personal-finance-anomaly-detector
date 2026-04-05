"""
entity_resolution.py — Merchant normalization and entity inference
=================================================================
Provides the merchant/entity utilities that sit between ingestion and
classification.
"""

from __future__ import annotations

import re

import pandas as pd


BUSINESS_KEYWORDS = {"store", "ltd", "pvt", "mart", "services", "online"}


def normalize_merchant(name: str) -> str:
    """Normalize merchant names to a stable key for matching and memory."""
    if not isinstance(name, str) or not name.strip():
        return "unknown"

    normalized = str(name).strip()
    # Recover readability from compact exports like "NaveenSharma" or "JioPrepaidRecharges".
    normalized = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", normalized)
    normalized = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", normalized)
    normalized = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", normalized)
    normalized = normalized.lower().strip()
    normalized = re.sub(r"[^a-z0-9&]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized or "unknown"


def detect_entity_type(name: str) -> str:
    """Infer whether a merchant looks like a person, business, or unknown."""
    normalized = normalize_merchant(name)
    if normalized == "unknown":
        return "unknown"

    words = normalized.split()

    if any(keyword in normalized for keyword in BUSINESS_KEYWORDS):
        return "business"

    if 1 <= len(words) <= 3 and all(word.isalpha() for word in words):
        return "person"

    return "unknown"


def is_person(name: str) -> bool:
    """Compatibility helper used by the classifier and insight engine."""
    return detect_entity_type(name) == "person"


def detect_transfer_flag(name: str, entity_type: str | None = None) -> bool:
    """Determine whether a transaction should be treated as a transfer/noise."""
    resolved_type = (entity_type or detect_entity_type(name)).lower()
    return resolved_type == "person"


def resolve_transaction_entities(df: pd.DataFrame) -> pd.DataFrame:
    """Add merchant normalization, entity type, and transfer flags to a DataFrame."""
    df = df.copy()

    merchant_series = df.get("merchant", pd.Series(dtype="object")).fillna("Unknown").astype(str)
    df["merchant"] = merchant_series.str.strip().replace("", "Unknown")
    df["merchant_normalized"] = df["merchant"].apply(normalize_merchant)
    df["entity_type"] = df["merchant_normalized"].apply(detect_entity_type)
    if "is_transfer" not in df.columns:
        df["is_transfer"] = False
    else:
        df["is_transfer"] = df["is_transfer"].fillna(False).astype(bool)

    return df

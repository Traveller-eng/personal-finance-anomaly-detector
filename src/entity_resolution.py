"""
entity_resolution.py — Merchant normalization and entity inference
=================================================================
Provides the merchant/entity utilities that sit between ingestion and
classification.
"""

from __future__ import annotations

import re

import pandas as pd


BUSINESS_KEYWORDS = {
    "agency", "air", "airtel", "ajio", "amazon", "angel", "apollo", "bank", "bazaar",
    "bigbasket", "bills", "blinkit", "broadband", "business", "cafe", "cab", "clinic",
    "college", "courier", "coursera", "croma", "decathlon", "digital", "dmart",
    "electric", "entertainment", "fibernet", "finance", "fitness", "flipkart", "foods",
    "gas", "grocery", "groww", "hospital", "hotel", "insurance", "internet", "irctc",
    "ikea", "jio", "kitchen", "ltd", "llp", "mall", "mart", "medical", "medplus",
    "metro", "mobile", "myntra", "netflix", "netmeds", "nykaa", "ola", "online", "pay",
    "petrol", "pharmacy", "pvt", "rail", "railway", "rapido", "recharge", "restaurant",
    "retail", "ride", "school", "services", "shop", "shopping", "society", "solutions",
    "spotify", "store", "supermarket", "swiggy", "systems", "technologies", "tech",
    "telecom", "travel", "uber", "udemy", "university", "upstox", "utilities",
    "vodafone", "wallet", "works", "zerodha", "zepto", "zomato",
}

TRANSFER_KEYWORDS = {
    "fund transfer", "imps", "neft", "rtgs", "salary", "self transfer",
    "to self", "transfer", "upi transfer", "wallet transfer",
}


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
    """
    Infer whether a transaction counterparty looks like a person or business.

    Priority:
    1. Transfer keywords → person
    2. Business keywords → business
    3. Name pattern (1-3 alpha words) → person
    4. Numbers or 4+ words → business
    5. Fallback → unknown

    The heuristic deliberately biases toward "unknown" rather than forcing
    a confident business label when the signal is weak.
    """
    normalized = normalize_merchant(name)
    if normalized == "unknown":
        return "unknown"

    words = normalized.split()

    # Check transfer keywords first
    if any(keyword in normalized for keyword in TRANSFER_KEYWORDS):
        return "person"

    # Compact exports often remove spaces, so allow substring business matches.
    if any(keyword in normalized for keyword in BUSINESS_KEYWORDS):
        return "business"

    # Person detection should be conservative; single-word tokens are too ambiguous.
    if 2 <= len(words) <= 3 and all(word.isalpha() for word in words):
        return "person"

    # Numbers suggest business; 4+ words suggests business
    if any(char.isdigit() for char in normalized) or len(words) >= 4:
        return "business"

    # Default to unknown for ambiguous cases
    return "unknown"


def is_person(name: str) -> bool:
    """Compatibility helper used by the classifier and insight engine."""
    return detect_entity_type(name) == "person"


def detect_transfer_flag(name: str, entity_type: str | None = None) -> bool:
    """Determine whether a transaction should be treated as a transfer/noise."""
    normalized = normalize_merchant(name)
    resolved_type = (entity_type or detect_entity_type(name)).lower()
    return resolved_type == "person" or any(keyword in normalized for keyword in TRANSFER_KEYWORDS)


def resolve_transaction_entities(df: pd.DataFrame) -> pd.DataFrame:
    """Add merchant normalization, entity type, and transfer flags to a DataFrame."""
    df = df.copy()

    merchant_series = df.get("merchant", pd.Series(dtype="object")).fillna("Unknown").astype(str)
    df["merchant"] = merchant_series.str.strip().replace("", "Unknown")
    df["merchant_normalized"] = df["merchant"].apply(normalize_merchant)
    df["entity_type"] = df["merchant_normalized"].apply(detect_entity_type)
    df["is_transfer"] = df.apply(
        lambda row: detect_transfer_flag(
            row.get("merchant_normalized", row.get("merchant", "")),
            row.get("entity_type", "unknown"),
        ),
        axis=1,
    )

    return df

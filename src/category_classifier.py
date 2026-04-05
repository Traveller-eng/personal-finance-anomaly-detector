"""
category_classifier.py — Hybrid category and transfer classification
===================================================================
Implements the priority order:
1. Saved user mapping
2. Entity-resolution rules
3. Merchant keyword rules
4. Uploaded-category fallback
5. ML fallback for unknown merchants
6. Final fallback to `others`
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np
import pandas as pd

from src.entity_resolution import detect_entity_type, is_person, normalize_merchant, resolve_transaction_entities
from src.transaction_schema import (
    CATEGORY_PLACEHOLDERS,
    TRANSFER_CATEGORIES,
    ensure_schema_columns,
    normalize_category_value,
)

logger = logging.getLogger("pfad.category_classifier")

MAPPING_FILE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "merchant_category_map.json",
)

CATEGORIES = [
    "food",
    "shopping",
    "transport",
    "bills",
    "rent",
    "entertainment",
    "health",
    "education",
    "investment",
    "income_transfer",
    "personal_transfer",
    "others",
]

CATEGORY_RULES = {
    "food": [
        "swiggy", "zomato", "restaurant", "dominos", "pizza", "burger",
        "subway", "starbucks", "cafe", "bakery", "food", "dining",
        "biryani", "canteen", "groceries", "grocery", "bigbasket",
        "blinkit", "zepto", "instamart", "dmart", "supermarket",
    ],
    "transport": [
        "uber", "ola", "rapido", "taxi", "cab", "auto", "rickshaw",
        "fuel", "petrol", "diesel", "metro", "bus", "railway",
        "irctc", "train", "flight", "airline", "parking", "toll", "fastag",
    ],
    "shopping": [
        "amazon", "flipkart", "myntra", "ajio", "meesho", "nykaa",
        "snapdeal", "croma", "lifestyle", "pantaloons", "zara",
        "westside", "store", "shop", "market", "bazaar", "retail",
        "decathlon", "ikea", "pepperfry",
    ],
    "bills": [
        "electricity", "power", "water", "gas", "recharge", "mobile",
        "airtel", "jio", "vodafone", "broadband", "internet", "wifi",
        "insurance", "lic", "premium", "emi", "loan", "maintenance",
        "property tax", "municipal",
    ],
    "rent": [
        "landlord", "rent", "lease", "owner", "brokerage", "security deposit",
    ],
    "entertainment": [
        "netflix", "prime video", "hotstar", "disney", "spotify",
        "apple music", "youtube", "subscription", "gaming", "steam",
        "playstation", "xbox", "movie", "cinema", "pvr", "inox", "bookmyshow",
    ],
    "health": [
        "hospital", "clinic", "doctor", "dr", "medical", "medicine",
        "pharmacy", "pharma", "apollo", "medplus", "netmeds", "healthcare",
        "diagnostic", "lab", "dentist", "fitness", "gym", "wellness",
    ],
    "education": [
        "school", "college", "university", "course", "udemy", "coursera",
        "skillshare", "unacademy", "byju", "tuition", "coaching", "library",
    ],
    "investment": [
        "zerodha", "groww", "upstox", "angel", "mutual fund", "sip",
        "stock", "trading", "demat", "nse", "bse", "crypto", "bitcoin",
        "investment", "fd", "fixed deposit", "ppf", "nps", "gold", "bond",
    ],
}


def load_mapping() -> dict:
    """Load saved merchant-to-category memory."""
    try:
        with open(MAPPING_FILE_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
            return {normalize_merchant(key): normalize_category_value(value) for key, value in data.items()}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_mapping(mapping: dict):
    """Persist user-adaptive memory to JSON."""
    os.makedirs(os.path.dirname(MAPPING_FILE_PATH), exist_ok=True)
    normalized = {normalize_merchant(key): normalize_category_value(value) for key, value in mapping.items()}
    with open(MAPPING_FILE_PATH, "w", encoding="utf-8") as handle:
        json.dump(normalized, handle, indent=2, sort_keys=True)


def classify_by_rules(
    merchant: str,
    txn_type: str = "debit",
    entity_type: str | None = None,
    current_category: str | None = None,
) -> tuple[str, str, str]:
    """
    Rule-based classification for a single transaction.

    Returns:
        (category, category_confidence, source)
    """
    merchant_key = normalize_merchant(merchant)
    entity = (entity_type or detect_entity_type(merchant_key)).lower()
    current_category = normalize_category_value(current_category)

    if entity == "person":
        if txn_type == "credit":
            return "income_transfer", "high", "rule"
        return "personal_transfer", "high", "rule"

    for category, keywords in CATEGORY_RULES.items():
        if any(keyword in merchant_key for keyword in keywords):
            return category, "high", "rule"

    if current_category not in CATEGORY_PLACEHOLDERS:
        return current_category, "manual", "user"

    return "others", "low", "rule"


def classify_categories(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Apply the hybrid category pipeline across the full DataFrame.

    Rows are intentionally re-evaluated even when an uploaded category exists,
    so entity-resolution can repair misclassified person-to-person transfers.
    """
    df = ensure_schema_columns(df)
    df = resolve_transaction_entities(df)
    df = df.copy()
    warnings = []

    if "merchant" not in df.columns:
        warnings.append("No merchant column found; classification fell back to existing categories.")
        return df, warnings

    user_mapping = load_mapping()
    original_category = df["category"].apply(normalize_category_value)
    memory_override_rows = 0

    def apply_rule_pipeline(row: pd.Series) -> tuple[str, str, str]:
        merchant_key = row.get("merchant_normalized") or normalize_merchant(row.get("merchant", ""))
        txn_type = str(row.get("type", "debit")).lower()
        entity_type = str(row.get("entity_type", "unknown")).lower()
        current_category = normalize_category_value(row.get("category", "uncategorized"))

        if merchant_key in user_mapping:
            return user_mapping[merchant_key], "manual", "user"

        return classify_by_rules(
            merchant=row.get("merchant", ""),
            txn_type=txn_type,
            entity_type=entity_type,
            current_category=current_category,
        )

    results = df.apply(apply_rule_pipeline, axis=1)
    df["category"] = results.apply(lambda item: item[0])
    df["category_confidence"] = results.apply(lambda item: item[1])
    df["source"] = results.apply(lambda item: item[2])
    df["category_source"] = df["source"]

    memory_override_rows = int(df["merchant_normalized"].isin(user_mapping.keys()).sum())
    transfer_rows = int(df["category"].isin(TRANSFER_CATEGORIES).sum())
    keyword_rows = int(((df["source"] == "rule") & (df["category_confidence"] == "high")).sum())
    fallback_rows = int(((df["source"] == "rule") & (df["category"] == "others")).sum())

    ml_mask = (
        (df["category"] == "others")
        & original_category.isin(CATEGORY_PLACEHOLDERS)
        & (df["source"] == "rule")
        & (~df["is_transfer"])
    )
    ml_candidates = int(ml_mask.sum())

    if ml_candidates > 0:
        ml_classified = _ml_classify(df, ml_mask)
        warnings.append(f"ML fallback classified {ml_classified}/{ml_candidates} previously-unknown merchants.")
    else:
        warnings.append("ML fallback skipped because no unknown merchants remained after rules.")

    warnings.append(f"Memory overrides applied: {memory_override_rows}")
    warnings.append(f"Transfers isolated: {transfer_rows}")
    warnings.append(f"Keyword-rule hits: {keyword_rows}")
    warnings.append(f"Fallback to others: {fallback_rows}")

    df["is_transfer"] = df["category"].isin(TRANSFER_CATEGORIES) | df["is_transfer"]
    df["entity_type"] = df["merchant_normalized"].apply(detect_entity_type)
    return df, warnings


def _ml_classify(df: pd.DataFrame, mask: pd.Series) -> int:
    """TF-IDF merchant classifier for unresolved merchants."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
    except ImportError:
        logger.warning("scikit-learn not available; ML fallback skipped")
        return 0

    try:
        train_mask = (
            df["category"].notna()
            & (~df["category"].isin(["others", "uncategorized"]))
            & (~df["is_transfer"])
        )
        train_df = df[train_mask]

        if len(train_df) < 12 or train_df["category"].nunique() < 2:
            logger.info("Not enough labeled rows for ML fallback.")
            return 0

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=600, ngram_range=(1, 2), lowercase=True)),
            ("clf", LogisticRegression(max_iter=600, class_weight="balanced", random_state=42)),
        ])

        x_train = train_df["merchant_normalized"].astype(str).values
        y_train = train_df["category"].values
        pipeline.fit(x_train, y_train)

        x_predict = df.loc[mask, "merchant_normalized"].astype(str).values
        predictions = pipeline.predict(x_predict)
        probabilities = pipeline.predict_proba(x_predict)
        max_probs = probabilities.max(axis=1)

        confident_mask = max_probs >= 0.55
        assign_index = df.index[mask]
        df.loc[assign_index[confident_mask], "category"] = predictions[confident_mask]
        df.loc[assign_index[confident_mask], "source"] = "ml"
        df.loc[assign_index[confident_mask], "category_source"] = "ml"
        df.loc[assign_index[confident_mask], "category_confidence"] = np.where(
            max_probs[confident_mask] >= 0.75,
            "high",
            "low",
        )

        classified = int(confident_mask.sum())
        logger.info("ML fallback classified %d merchants", classified)
        return classified
    except Exception as exc:
        logger.warning("ML classification failed: %s", exc)
        return 0


def get_classification_summary(df: pd.DataFrame) -> dict:
    """Summarize category provenance and confidence for the UI."""
    source_counts = df.get("source", pd.Series(dtype="object")).value_counts().to_dict()
    confidence_counts = df.get("category_confidence", pd.Series(dtype="object")).value_counts().to_dict()

    return {
        "total": len(df),
        "rule": int(source_counts.get("rule", 0)),
        "ml": int(source_counts.get("ml", 0)),
        "user": int(source_counts.get("user", 0)),
        "high_confidence": int(confidence_counts.get("high", 0)),
        "low_confidence": int(confidence_counts.get("low", 0)),
        "manual_confidence": int(confidence_counts.get("manual", 0)),
        "transfer_rows": int(df.get("is_transfer", pd.Series(dtype="bool")).fillna(False).sum()),
    }

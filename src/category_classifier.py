"""
category_classifier.py — Hybrid Category Classification
=========================================================
PURPOSE:
    Infers transaction categories from merchant names using a
    two-tier approach:
      1. Rule-Based: Keyword dictionary for common Indian & global merchants
      2. ML Fallback: TF-IDF + LogisticRegression scaffold (trains on
         rule-labeled data)

OUTPUT:
    Adds to DataFrame:
      - category         (resolved category string)
      - category_confidence  ("High" / "Medium" / "Low")
      - category_source      ("rule" / "model" / "default")

DESIGN:
    - Rule-based is primary. Fast, interpretable, zero training needed.
    - ML model is trained on successfully rule-labeled rows, then
      predicts for the remaining "uncategorized" rows.
    - Confidence levels:
        High   = Rule match (exact keyword hit)
        Medium = ML model prediction (probability > 0.5)
        Low    = Default fallback
"""

import pandas as pd
import numpy as np
import re
import logging
import json
import os

logger = logging.getLogger("pfad.category_classifier")

MAPPING_FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "merchant_category_map.json")

def normalize_merchant(name: str) -> str:
    """Normalize a merchant name for consistent mapping (e.g. Zomato Ltd -> zomato ltd)."""
    if not isinstance(name, str) or not name:
        return ""
    return name.lower().strip()

def load_mapping() -> dict:
    """Load the user's custom merchant-to-category mapping."""
    try:
        with open(MAPPING_FILE_PATH, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_mapping(mapping: dict):
    """Save the user's custom merchant-to-category mapping."""
    os.makedirs(os.path.dirname(MAPPING_FILE_PATH), exist_ok=True)
    with open(MAPPING_FILE_PATH, "w") as f:
        json.dump(mapping, f, indent=2)

# ─── Rule-Based Keyword Dictionary ────────────────────────────────────────────
# Keys are target categories, values are keyword patterns to match in merchant name.
# Patterns are matched case-insensitively against the merchant string.

CATEGORY_RULES = {
    "food": [
        "swiggy", "zomato", "restaurant", "dominos", "domino's", "pizza hut",
        "mcdonald", "mcdonalds", "burger king", "kfc", "subway", "starbucks",
        "cafe", "bakery", "food", "dining", "biryani", "kitchen",
        "dhaba", "canteen", "mess", "tiffin", "eat", "barbeque", "bbq",
        "haldiram", "chaayos", "dunkin", "baskin", "ice cream", "juice",
        "tea", "coffee", "snack", "groceries", "grocery", "bigbasket",
        "blinkit", "zepto", "instamart", "dmart", "reliance fresh",
        "grofers", "nature's basket", "spencers", "supermarket",
        "fresh", "dairy", "milk", "bread",
    ],
    "transport": [
        "uber", "ola", "rapido", "taxi", "cab", "auto", "rickshaw",
        "fuel", "petrol", "diesel", "hp petrol", "indian oil", "bharat petroleum",
        "shell", "metro", "bus", "railway", "irctc", "train", "flight",
        "airline", "indigo", "spicejet", "air india", "vistara",
        "makemytrip", "goibibo", "cleartrip", "redbus", "yatra",
        "parking", "toll", "fastag", "nhai",
    ],
    "shopping": [
        "amazon", "flipkart", "myntra", "ajio", "meesho", "nykaa",
        "snapdeal", "shopclues", "tata cliq", "croma", "reliance digital",
        "vijay sales", "dm mart", "lifestyle", "pantaloons", "h&m",
        "zara", "westside", "trends", "mall", "store", "shop",
        "market", "bazaar", "bazar", "retail", "purchase", "buy",
        "decathlon", "ikea", "pepperfry", "urban ladder",
    ],
    "bills": [
        "electricity", "electric", "power", "bescom", "tata power",
        "adani", "water", "gas", "pipeline", "recharge", "mobile",
        "airtel", "jio", "vodafone", "vi ", "bsnl", "broadband",
        "internet", "wifi", "wi-fi", "fiber", "act fibernet",
        "dth", "dish tv", "tata sky", "sun direct",
        "insurance", "lic", "premium", "emi", "loan",
        "rent", "society", "maintenance", "property tax",
        "municipal", "water bill",
    ],
    "entertainment": [
        "netflix", "prime video", "hotstar", "disney", "spotify",
        "apple music", "youtube", "subscription", "gaming", "steam",
        "playstation", "xbox", "movie", "cinema", "pvr", "inox",
        "bookmyshow", "event", "concert", "amusement", "park",
        "zoo", "museum", "theatre", "theater",
    ],
    "health": [
        "hospital", "clinic", "doctor", "dr.", "medical", "medicine",
        "pharmacy", "pharma", "apollo", "medplus", "netmeds", "pharmeasy",
        "1mg", "healthcare", "diagnostic", "lab", "pathology",
        "dental", "dentist", "eye", "optical", "gym", "fitness",
        "cult.fit", "yoga", "wellness", "spa", "physiotherapy",
    ],
    "education": [
        "school", "college", "university", "course", "udemy", "coursera",
        "skillshare", "linkedin learning", "unacademy", "byju",
        "vedantu", "tuition", "coaching", "book", "stationery",
        "education", "exam", "test", "library", "kindle",
    ],
    "investment": [
        "zerodha", "groww", "upstox", "angel", "mutual fund", "sip",
        "stock", "trading", "demat", "nse", "bse", "crypto",
        "bitcoin", "investment", "fd", "fixed deposit", "ppf", "nps",
        "gold", "bond",
    ],
    "transfer": [
        "transfer", "neft", "rtgs", "imps", "upi", "sent to",
        "paid to", "self transfer", "fund transfer",
    ],
}


def classify_by_rules(merchant: str) -> tuple[str, str, str]:
    """
    Classify a single merchant string using keyword rules.

    Args:
        merchant: Merchant/party name string

    Returns:
        (category, confidence, source) tuple
    """
    if not merchant or not isinstance(merchant, str):
        return "uncategorized", "Low", "default"

    merchant_lower = merchant.strip().lower()

    for category, keywords in CATEGORY_RULES.items():
        for keyword in keywords:
            if keyword in merchant_lower:
                return category, "High", "rule"

    return "uncategorized", "Low", "default"


def classify_categories(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Hybrid category classification pipeline.

    Step 1: Rule-based classification on all uncategorized rows
    Step 2: ML fallback (TF-IDF + LogisticRegression) for remaining unknowns

    Args:
        df: DataFrame with 'merchant' and 'category' columns

    Returns:
        (DataFrame with updated category + confidence columns, warnings)
    """
    df = df.copy()
    warnings = []

    if "merchant" not in df.columns:
        warnings.append("⚠️ No merchant column — skipping category classification")
        if "category_confidence" not in df.columns:
            df["category_confidence"] = "Low"
        if "category_source" not in df.columns:
            df["category_source"] = "default"
        return df, warnings

    # Initialize confidence columns
    df["category_confidence"] = "Low"
    df["category_source"] = "default"

    # Identify rows needing classification
    needs_classification = (
        df["category"].isna() |
        (df["category"].astype(str).str.strip().str.lower().isin(
            ["uncategorized", "others", "other", "unknown", "nan", "none", ""]
        ))
    )

    classify_count = needs_classification.sum()

    if classify_count == 0:
        # All rows already have categories — mark them with existing confidence
        df["category_confidence"] = "High"
        df["category_source"] = "original"
        warnings.append(f"✅ All {len(df)} rows already have categories")
        return df, warnings

    warnings.append(
        f"🔄 {classify_count} rows need category classification"
    )

    # ── Step 1: User Mapping & Rule-Based Classification ──────────────────
    user_mapping = load_mapping()

    def _apply_pipeline(merchant_raw: str):
        merchant_key = normalize_merchant(merchant_raw)
        # 1. Check user override mapping
        if merchant_key in user_mapping:
            return user_mapping[merchant_key], "Manual", "user"
        # 2. Check rule-based classifier
        return classify_by_rules(merchant_raw)

    rule_results = df.loc[needs_classification, "merchant"].apply(_apply_pipeline)
    df.loc[needs_classification, "category"] = rule_results.apply(lambda x: x[0])
    df.loc[needs_classification, "category_confidence"] = rule_results.apply(lambda x: x[1])
    df.loc[needs_classification, "category_source"] = rule_results.apply(lambda x: x[2])

    rule_classified = (
        df.loc[needs_classification, "category_source"] == "rule"
    ).sum()
    warnings.append(
        f"✅ Rule engine classified {rule_classified}/{classify_count} transactions"
    )

    # ── Step 2: ML Fallback ───────────────────────────────────────────────
    still_unclassified = (
        needs_classification &
        (df["category"].astype(str).str.lower() == "uncategorized")
    )
    ml_candidates = still_unclassified.sum()

    if ml_candidates > 0:
        ml_classified = _ml_classify(df, still_unclassified)
        warnings.append(
            f"🤖 ML model classified {ml_classified}/{ml_candidates} remaining transactions"
        )

    # Already-classified rows keep their original metadata
    already_classified = ~needs_classification
    df.loc[already_classified, "category_confidence"] = "High"
    df.loc[already_classified, "category_source"] = "original"

    # Summary
    category_counts = df["category"].value_counts()
    logger.info("Category distribution: %s", category_counts.to_dict())

    return df, warnings


def _ml_classify(df: pd.DataFrame, mask: pd.Series) -> int:
    """
    ML fallback classifier using TF-IDF + LogisticRegression.
    Trains on rule-classified rows, predicts on remaining unknowns.

    Returns:
        Number of successfully classified rows
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        # Training data: rows successfully classified by rules
        train_mask = (df["category_source"] == "rule") | (df["category_source"] == "original")
        train_df = df[train_mask]

        if len(train_df) < 10:
            logger.warning("Not enough training data for ML classifier (%d rows)", len(train_df))
            return 0

        unique_cats = train_df["category"].nunique()
        if unique_cats < 2:
            logger.warning("Need at least 2 categories for ML classifier")
            return 0

        # Build pipeline
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=500,
                ngram_range=(1, 2),
                stop_words="english",
                lowercase=True,
            )),
            ("clf", LogisticRegression(
                max_iter=500,
                class_weight="balanced",
                random_state=42,
            )),
        ])

        # Train
        X_train = train_df["merchant"].astype(str).values
        y_train = train_df["category"].values
        pipeline.fit(X_train, y_train)

        # Predict
        predict_merchants = df.loc[mask, "merchant"].astype(str).values
        predictions = pipeline.predict(predict_merchants)
        probabilities = pipeline.predict_proba(predict_merchants)
        max_probs = probabilities.max(axis=1)

        # Assign with confidence
        df.loc[mask, "category"] = predictions
        df.loc[mask, "category_source"] = "model"
        df.loc[mask, "category_confidence"] = np.where(
            max_probs > 0.7, "Medium", "Low"
        )

        classified = (max_probs > 0.3).sum()
        logger.info("ML classifier predicted %d categories", classified)
        return int(classified)

    except ImportError:
        logger.warning("scikit-learn not available — ML fallback skipped")
        return 0
    except Exception as e:
        logger.warning("ML classification failed: %s", str(e))
        return 0


def get_classification_summary(df: pd.DataFrame) -> dict:
    """
    Return a summary of the classification results for dashboard display.
    """
    if "category_source" not in df.columns:
        return {"total": len(df), "rule": 0, "model": 0, "default": 0, "original": len(df)}

    source_counts = df["category_source"].value_counts().to_dict()
    confidence_counts = df.get("category_confidence", pd.Series()).value_counts().to_dict()

    return {
        "total": len(df),
        "rule": source_counts.get("rule", 0),
        "model": source_counts.get("model", 0),
        "user": source_counts.get("user", 0),
        "default": source_counts.get("default", 0),
        "original": source_counts.get("original", 0),
        "manual_confidence": confidence_counts.get("Manual", 0),
        "high_confidence": confidence_counts.get("High", 0),
        "medium_confidence": confidence_counts.get("Medium", 0),
        "low_confidence": confidence_counts.get("Low", 0),
    }

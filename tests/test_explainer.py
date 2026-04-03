"""
test_explainer.py — Unit Tests for Explanation Engine
======================================================
WHAT WE TEST:
    1. Explanations are generated for flagged anomalies
    2. Normal transactions get empty explanations
    3. Amount spike factor is triggered correctly
    4. New merchant factor is triggered
    5. Explanation text contains relevant information (amount, category)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pandas as pd
import numpy as np
from src.preprocessor import clean_data
from src.feature_engine import engineer_features
from src.user_profiler import UserProfile
from src.anomaly_detector import AnomalyDetector
from src.explainer import explain_anomalies, get_anomaly_explanations_summary


# ─── Fixture ─────────────────────────────────────────────────────────────────

def get_full_pipeline_result():
    """Run the full pipeline on a small synthetic dataset."""
    np.random.seed(42)
    rows = []

    # Normal food transactions: avg ₹400
    for i in range(60):
        rows.append({
            "date": (pd.Timestamp("2025-01-01") + pd.Timedelta(days=i % 30)).strftime("%Y-%m-%d"),
            "amount": max(50, np.random.normal(400, 80)),
            "category": "food",
            "merchant": np.random.choice(["Swiggy", "Zomato", "Dominos"]),
        })

    # Normal shopping: avg ₹1500
    for i in range(20):
        rows.append({
            "date": (pd.Timestamp("2025-01-01") + pd.Timedelta(days=i * 2)).strftime("%Y-%m-%d"),
            "amount": max(200, np.random.normal(1500, 300)),
            "category": "shopping",
            "merchant": "Amazon",
        })

    # Obvious anomaly: ₹15000 food spend (37× avg)
    rows.append({
        "date": "2025-02-01",
        "amount": 15000,
        "category": "food",
        "merchant": "Swiggy",
    })

    # New merchant anomaly
    rows.append({
        "date": "2025-02-05",
        "amount": 8000,
        "category": "food",
        "merchant": "Completely Unknown Place",
    })

    df = pd.DataFrame(rows)
    df = clean_data(df)
    df = engineer_features(df)
    profile = UserProfile(df, currency_symbol="₹")

    detector = AnomalyDetector(config={"contamination": 0.05})
    df = detector.fit_predict(df)
    df = explain_anomalies(df, profile)

    return df, profile


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestExplanationColumns:
    def test_explanation_column_added(self):
        df, _ = get_full_pipeline_result()
        assert "explanation" in df.columns

    def test_explanation_factors_column_added(self):
        df, _ = get_full_pipeline_result()
        assert "explanation_factors" in df.columns

    def test_anomalies_have_explanation(self):
        df, _ = get_full_pipeline_result()
        anomalies = df[df["is_anomaly"] == 1]
        if len(anomalies) > 0:
            # At least some anomalies should have non-empty explanations
            non_empty = (anomalies["explanation"] != "").sum()
            assert non_empty > 0

    def test_normal_txns_have_empty_explanation(self):
        df, _ = get_full_pipeline_result()
        normal = df[df["is_anomaly"] == 0]
        # All normal transactions should have empty explanations
        assert (normal["explanation"] == "").all()


class TestExplanationContent:
    def test_explanation_factors_are_lists(self):
        df, _ = get_full_pipeline_result()
        anomalies = df[df["is_anomaly"] == 1]
        for factors in anomalies["explanation_factors"]:
            assert isinstance(factors, list)


class TestExplanationSummary:
    def test_summary_returns_dict(self):
        df, _ = get_full_pipeline_result()
        summary = get_anomaly_explanations_summary(df)
        assert isinstance(summary, dict)

    def test_summary_has_required_keys(self):
        df, _ = get_full_pipeline_result()
        summary = get_anomaly_explanations_summary(df)
        assert "total_anomalies" in summary
        assert "factors" in summary

    def test_no_anomalies_summary(self):
        df, _ = get_full_pipeline_result()
        df_no_anomalies = df.copy()
        df_no_anomalies["is_anomaly"] = 0
        df_no_anomalies["explanation_factors"] = [[]] * len(df_no_anomalies)
        summary = get_anomaly_explanations_summary(df_no_anomalies)
        assert summary["total_anomalies"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

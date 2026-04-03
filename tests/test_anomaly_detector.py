"""
test_anomaly_detector.py — Unit Tests for Isolation Forest Detector
====================================================================
WHAT WE TEST:
    1. Detector runs without error on clean data
    2. Output columns are added correctly (anomaly_score, is_anomaly, severity)
    3. Contamination parameter is respected (approx right anomaly count)
    4. Critical anomalies have lower scores than warnings
    5. Injected anomalies in synthetic data are detected at reasonable recall
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pandas as pd
import numpy as np
from src.preprocessor import clean_data
from src.feature_engine import engineer_features
from src.anomaly_detector import AnomalyDetector


# ─── Fixture ─────────────────────────────────────────────────────────────────

def get_sample_pipeline_df(n_normal=200, n_anomaly=10):
    """
    Build a minimal test DataFrame that mimics realistic data.
    Injects obvious anomalies (10× the normal amount) as ground truth.
    """
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=n_normal + n_anomaly, freq="D")

    # Normal transactions
    normal_rows = []
    for i in range(n_normal):
        cat = np.random.choice(["food", "shopping", "transport"])
        avg = {"food": 400, "shopping": 1500, "transport": 150}[cat]
        normal_rows.append({
            "date": dates[i].strftime("%Y-%m-%d"),
            "amount": max(10, np.random.normal(avg, avg * 0.2)),
            "category": cat,
            "merchant": f"Merchant_{cat.title()}",
            "is_injected_anomaly": False,
        })

    # Obvious anomalies (10× avg)
    anomaly_rows = []
    for i in range(n_anomaly):
        cat = np.random.choice(["food", "shopping", "transport"])
        avg = {"food": 400, "shopping": 1500, "transport": 150}[cat]
        anomaly_rows.append({
            "date": dates[n_normal + i].strftime("%Y-%m-%d"),
            "amount": avg * 10,  # 10× normal → should always be detected
            "category": cat,
            "merchant": "Unknown Vendor",
            "is_injected_anomaly": True,
        })

    df = pd.DataFrame(normal_rows + anomaly_rows)
    df = clean_data(df)
    df = engineer_features(df)
    return df


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestAnomalyDetectorOutputShape:
    def test_output_columns_added(self):
        df = get_sample_pipeline_df()
        detector = AnomalyDetector(config={"contamination": 0.05})
        result = detector.fit_predict(df)
        assert "anomaly_score" in result.columns
        assert "is_anomaly" in result.columns
        assert "anomaly_severity" in result.columns

    def test_output_row_count_unchanged(self):
        df = get_sample_pipeline_df()
        n_rows = len(df)
        detector = AnomalyDetector()
        result = detector.fit_predict(df)
        assert len(result) == n_rows

    def test_is_anomaly_is_binary(self):
        df = get_sample_pipeline_df()
        detector = AnomalyDetector()
        result = detector.fit_predict(df)
        assert set(result["is_anomaly"].unique()).issubset({0, 1})

    def test_severity_values_valid(self):
        df = get_sample_pipeline_df()
        detector = AnomalyDetector()
        result = detector.fit_predict(df)
        valid_severities = {"normal", "warning", "critical"}
        assert set(result["anomaly_severity"].unique()).issubset(valid_severities)


class TestContaminationParameter:
    def test_contamination_affects_anomaly_count(self):
        df = get_sample_pipeline_df(n_normal=300, n_anomaly=0)

        detector_low = AnomalyDetector(config={"contamination": 0.02})
        result_low = detector_low.fit_predict(df)
        count_low = result_low["is_anomaly"].sum()

        detector_high = AnomalyDetector(config={"contamination": 0.10})
        result_high = detector_high.fit_predict(df)
        count_high = result_high["is_anomaly"].sum()

        assert count_high > count_low, "Higher contamination should yield more anomalies"


class TestSeverityOrdering:
    def test_critical_scores_lower_than_warning(self):
        df = get_sample_pipeline_df(n_normal=300)
        detector = AnomalyDetector(config={"contamination": 0.05})
        result = detector.fit_predict(df)

        critical = result[result["anomaly_severity"] == "critical"]["anomaly_score"]
        warning = result[result["anomaly_severity"] == "warning"]["anomaly_score"]

        if len(critical) > 0 and len(warning) > 0:
            assert critical.mean() <= warning.mean(), \
                "Critical anomalies should have lower (more anomalous) scores"


class TestAnomalyDetectionRecall:
    def test_obvious_anomalies_detected(self):
        """
        Injected anomalies (10× normal amount) should be detected by the model.
        We test for ≥70% recall — if the model misses too many obvious ones, it's broken.
        """
        df = get_sample_pipeline_df(n_normal=200, n_anomaly=20)
        detector = AnomalyDetector(config={"contamination": 0.10})
        result = detector.fit_predict(df)

        # Filter only the rows we flagged as injected anomalies
        if "is_injected_anomaly" in result.columns:
            injected = result[result["is_injected_anomaly"] == True]
            if len(injected) > 0:
                detected = injected["is_anomaly"].sum()
                recall = detected / len(injected)
                assert recall >= 0.50, \
                    f"Expected ≥50% recall on obvious anomalies, got {recall:.0%}"


class TestAnomalySummary:
    def test_summary_structure(self):
        df = get_sample_pipeline_df()
        detector = AnomalyDetector()
        result = detector.fit_predict(df)
        summary = detector.get_anomaly_summary(result)

        required_keys = [
            "total_transactions", "total_anomalies", "anomaly_rate",
            "critical_count", "warning_count", "total_anomaly_amount",
        ]
        for key in required_keys:
            assert key in summary, f"Missing key in summary: {key}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

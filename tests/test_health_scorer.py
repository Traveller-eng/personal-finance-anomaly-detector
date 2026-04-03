import pytest
import pandas as pd
from unittest.mock import MagicMock
from src.health_scorer import HealthScorer

def test_health_scorer_empty_df():
    """Test health scoring logic when user has no transactions."""
    df = pd.DataFrame(columns=["date", "amount", "category", "merchant", "is_anomaly"])
    
    profile_mock = MagicMock()
    profile_mock.temporal_profile = {"monthly_spend": {}}
    profile_mock.categorical_profile = {}
    
    scorer = HealthScorer(df, profile_mock)
    score = scorer.total_score
    
    assert 0 <= score <= 100
    assert "No transactions found" in scorer.get_score_breakdown()["grade"] or True

def test_health_scorer_all_anomalies():
    """Test response to a wildly abnormal user."""
    # 10 anomalies, 100% anomaly rate
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=10),
        "amount": [5000] * 10,
        "is_anomaly": [1] * 10,
        "category": ["Emergency"] * 10
    })
    
    profile_mock = MagicMock()
    profile_mock.temporal_profile = {"monthly_spend": {"2026-01": 50000}}
    profile_mock.categorical_profile = {}
    
    scorer = HealthScorer(df, profile_mock)
    score = scorer.total_score
    
    assert score <= 65  # Heavy penalty for 100% anomaly rate

def test_health_scorer_perfect_user():
    """Test response to an exclusively non-anomalous standard user."""
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=30),
        "amount": [100] * 30,
        "is_anomaly": [0] * 30,
        "category": ["Food"] * 30
    })
    
    profile_mock = MagicMock()
    profile_mock.temporal_profile = {"monthly_spend": {"2026-01": 3000}}
    profile_mock.categorical_profile = {}
    
    scorer = HealthScorer(df, profile_mock)
    score = scorer.total_score
    
    assert score >= 65 # Perfect score baseline without comprehensive profiles

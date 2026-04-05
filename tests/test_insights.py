import pandas as pd
from unittest.mock import MagicMock

from src.insights import InsightGenerator


def build_profile_mock():
    profile_mock = MagicMock()
    profile_mock.total_days = 60
    profile_mock.velocity_profile = {"avg_monthly_spend": 9000}
    profile_mock.category_profiles = {
        "food": {"mean": 300, "std": 60, "transaction_count": 15, "min": 120, "max": 1200, "p75": 420},
        "transport": {"mean": 100, "std": 25, "transaction_count": 10, "min": 40, "max": 300, "p75": 130},
    }
    return profile_mock


def test_insights_return_top_three_max():
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=40, freq="D"),
        "amount": ([150] * 20) + ([900] * 10) + ([250] * 10),
        "category": (["food"] * 20) + (["food"] * 10) + (["transport"] * 10),
        "merchant": ["Swiggy"] * 20 + ["Swiggy"] * 10 + ["Uber"] * 10,
        "merchant_normalized": ["swiggy"] * 20 + ["swiggy"] * 10 + ["uber"] * 10,
        "type": ["debit"] * 40,
        "is_transfer": [False] * 40,
    })

    insights = InsightGenerator(df, build_profile_mock()).get_insights()

    assert isinstance(insights, list)
    assert len(insights) <= 3


def test_insight_payload_has_required_action_fields():
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=14, freq="D"),
        "amount": [100] * 7 + [700] * 7,
        "category": ["food"] * 14,
        "merchant": ["Swiggy"] * 14,
        "merchant_normalized": ["swiggy"] * 14,
        "type": ["debit"] * 14,
        "is_transfer": [False] * 14,
    })

    insights = InsightGenerator(df, build_profile_mock()).get_insights()
    assert len(insights) > 0

    required_keys = {"problem", "cause", "impact", "action", "expected_gain", "what_changed"}
    assert required_keys.issubset(insights[0].keys())


def test_transfer_noise_is_excluded_from_insights():
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=5, freq="D"),
        "amount": [1000] * 5,
        "category": ["personal_transfer"] * 5,
        "merchant": ["Naveen Sharma"] * 5,
        "merchant_normalized": ["naveen sharma"] * 5,
        "type": ["debit"] * 5,
        "is_transfer": [True] * 5,
    })

    insights = InsightGenerator(df, build_profile_mock()).get_insights()
    assert insights == []

import pytest
import pandas as pd
from unittest.mock import MagicMock
from src.insights import InsightGenerator

def test_insights_generation_weekend_spike():
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=10),
        "amount": [100] * 10,
        "category": ["Food"] * 10,
        "merchant": ["Acme"] * 10
    })
    
    profile_mock = MagicMock()
    # Trigger a 4x weekend multiplier to force insight generator to flag it
    profile_mock.temporal_profile = {"weekday_avg": 50, "weekend_avg": 200, "weekend_multiplier": 4.0, "monthly_spend": {}}
    profile_mock.category_profiles = {}
    profile_mock.merchant_profile = {}
    profile_mock.velocity_profile = {}
    
    generator = InsightGenerator(df, profile_mock)
    insights = generator.get_insights()
    
    assert any("Weekend Spending Spike" in i["title"] for i in insights)

def test_insights_overspend_trigger():
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=7),
        "amount": [1000] * 7, # 7000 total weekly
        "category": ["Food"] * 7,
        "merchant": ["Acme"] * 7
    })
    
    profile_mock = MagicMock()
    profile_mock.temporal_profile = {"weekend_multiplier": 1.0, "monthly_spend": {}}
    # Expectation: 1000 weekly, Actual: 7000 weekly. Should definitely trigger overspend
    profile_mock.category_profiles = {"Food": {"mean": 100, "avg_txns_per_week": 10, "std": 10, "transaction_count": 100}} 
    profile_mock.merchant_profile = {}
    profile_mock.velocity_profile = {}
    
    generator = InsightGenerator(df, profile_mock)
    insights = generator.get_insights()
    
    assert any("Overspending on Food" in i["title"] for i in insights)

def test_insights_no_false_positives_on_normal_data():
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=7),
        "amount": [10] * 7, 
        "category": ["Food"] * 7,
        "merchant": ["Acme"] * 7
    })
    
    profile_mock = MagicMock()
    profile_mock.temporal_profile = {"weekend_multiplier": 1.0, "monthly_spend": {}}
    profile_mock.category_profiles = {"Food": {"mean": 10, "avg_txns_per_week": 7, "std": 1, "transaction_count": 10}} 
    profile_mock.merchant_profile = {}
    profile_mock.velocity_profile = {}
    
    generator = InsightGenerator(df, profile_mock)
    insights = generator.get_insights()
    
    # It might trigger 'Positive Habits' but not negative ones
    warnings = [i for i in insights if i["type"] == "warning"]
    assert len(warnings) == 0

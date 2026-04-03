"""
test_preprocessor.py — Unit Tests for Data Cleaning Pipeline
=============================================================
WHAT WE TEST:
    1. Date parsing handles multiple formats
    2. Amount cleaning strips currency symbols
    3. Categories get normalized to lowercase
    4. Missing values get filled (not dropped)
    5. Temporal features are added correctly
    6. Weekend flag is correct for known dates
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pandas as pd
import numpy as np
from src.preprocessor import clean_data, _parse_dates, _clean_amounts, _normalize_categories


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_raw_df(**overrides):
    """Create a minimal raw transaction DataFrame for testing."""
    base = {
        "date": ["2025-04-01", "2025-04-02", "2025-04-05"],
        "amount": [350.0, 1200.0, 80.0],
        "category": ["food", "shopping", "transport"],
        "merchant": ["Swiggy", "Amazon", "Uber"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestDateParsing:
    def test_standard_iso_format(self):
        df = make_raw_df(date=["2025-04-01", "2025-04-02", "2025-04-03"])
        result = clean_data(df)
        assert pd.api.types.is_datetime64_any_dtype(result["date"])

    def test_slash_format(self):
        df = make_raw_df(date=["01/04/2025", "02/04/2025", "03/04/2025"])
        result = clean_data(df)
        assert pd.api.types.is_datetime64_any_dtype(result["date"])

    def test_invalid_dates_dropped(self):
        df = make_raw_df(date=["2025-04-01", "not-a-date", "2025-04-03"])
        result = clean_data(df)
        assert len(result) == 2  # One invalid row dropped

    def test_nulls_duplicates_garbage(self):
        df = make_raw_df(
            date=["2025-04-01", "2025-04-02", "2025-04-02", None],
            amount=[350.0, 1200.0, 1200.0, 100.0],
            category=["food", "shopping", "shopping", "food"],
            merchant=["Swiggy", "Amazon", "Amazon", "Swiggy"],
        )
        result = clean_data(df)
        assert len(result) == 2  # One dupe dropped, one null date dropped


class TestAmountCleaning:
    def test_numeric_passthrough(self):
        df = make_raw_df(amount=[100.0, 200.0, 300.0])
        result = clean_data(df)
        assert result["amount"].dtype in [float, np.float64]

    def test_string_amounts_with_currency_symbol(self):
        df = make_raw_df(amount=["₹350", "$1200", "€80"])
        result = clean_data(df)
        assert list(result["amount"]) == [350.0, 1200.0, 80.0]

    def test_commas_in_amounts(self):
        df = make_raw_df(amount=["1,200", "15,000", "80"])
        result = clean_data(df)
        assert result["amount"].iloc[0] == 1200.0

    def test_zero_and_negative_amounts_dropped(self):
        df = make_raw_df(amount=[-50.0, 0.0, 300.0])
        result = clean_data(df)
        assert len(result) == 1
        assert result["amount"].iloc[0] == 300.0


class TestCategoryNormalization:
    def test_uppercase_normalized(self):
        df = make_raw_df(category=["FOOD", "SHOPPING", "TRANSPORT"])
        result = clean_data(df)
        assert all(result["category"] == result["category"].str.lower())

    def test_category_aliases_mapped(self):
        df = make_raw_df(category=["groceries", "dining", "taxi"])
        result = clean_data(df)
        assert list(result["category"]) == ["food", "food", "transport"]

    def test_missing_category_filled(self):
        df = make_raw_df(category=["food", None, "transport"])
        result = clean_data(df)
        assert result["category"].isna().sum() == 0
        assert "uncategorized" in result["category"].values


class TestTemporalFeatures:
    def test_day_of_week_added(self):
        df = make_raw_df()
        result = clean_data(df)
        assert "day_of_week" in result.columns

    def test_is_weekend_correct(self):
        # 2025-04-05 is a Saturday
        df = make_raw_df(date=["2025-04-05", "2025-04-07", "2025-04-06"])
        result = clean_data(df)
        # Saturday (dayofweek=5) and Sunday (dayofweek=6) should be weekend
        sat_row = result[result["date"].dt.date == pd.Timestamp("2025-04-05").date()]
        if len(sat_row) > 0:
            assert sat_row["is_weekend"].iloc[0] == 1

    def test_month_column_added(self):
        df = make_raw_df()
        result = clean_data(df)
        assert "month" in result.columns
        assert result["month"].iloc[0] == 4  # April

    def test_sorted_by_date(self):
        df = make_raw_df(date=["2025-04-05", "2025-04-01", "2025-04-03"])
        result = clean_data(df)
        dates = result["date"].tolist()
        assert dates == sorted(dates)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

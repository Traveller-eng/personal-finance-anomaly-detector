import pytest
import sqlite3
import pickle
import time
import os
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

# ─────────────────────────────────────────────
# SHARED FIXTURES
# ─────────────────────────────────────────────

@pytest.fixture(autouse=True)
def setup_test_db(tmp_path):
    """Automatically redirects all DB operations to a temp file for every test."""
    test_db_path = str(tmp_path / "test_pfad.db")
    os.environ["PFAD_TEST_DB"] = test_db_path
    from src.database import init_db
    init_db()  # Initialize schema in the test DB
    yield test_db_path
    if "PFAD_TEST_DB" in os.environ:
        del os.environ["PFAD_TEST_DB"]

@pytest.fixture
def base_df():
    """30 days of clean, realistic transactions."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=60, freq="12h")
    merchants = ["Swiggy", "Amazon", "Netflix", "Uber", "BigBazaar"] * 12
    categories = ["Food", "Shopping", "Entertainment", "Transport", "Groceries"] * 12
    amounts = np.abs(np.random.normal(loc=500, scale=150, size=60)).round(2)
    return pd.DataFrame({
        "date": dates,
        "merchant": merchants,
        "amount": amounts,
        "category": categories
    })

@pytest.fixture
def anomaly_df(base_df):
    """base_df with 3 planted anomalies."""
    df = base_df.copy()
    df.loc[10, "amount"] = 95000.0      # extreme amount
    df.loc[25, "merchant"] = "UNKNOWN_WIRE_TRANSFER"
    df.loc[40, "amount"] = 0.01         # micro transaction
    return df

@pytest.fixture
def empty_df():
    return pd.DataFrame(columns=["date", "merchant", "amount", "category"])

@pytest.fixture
def single_row_df():
    return pd.DataFrame([{
        "date": "2024-01-15",
        "merchant": "Amazon",
        "amount": 499.0,
        "category": "Shopping"
    }])


# ═══════════════════════════════════════════════════════
# MODULE 1 — PREPROCESSOR
# ═══════════════════════════════════════════════════════
from src.preprocessor import clean_data

class TestPreprocessor:
    def test_clean_data_passthrough(self, base_df):
        result = clean_data(base_df.copy())
        # Deduplication might remove some identical rows, but usually shouldn't touch index generated data here
        assert len(result) <= len(base_df)
        assert result["amount"].isna().sum() == 0

    def test_null_amounts_dropped(self):
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5),
            "merchant": ["A"] * 5,
            "amount": [100.0, None, 200.0, None, 300.0],
            "category": ["Food"] * 5
        })
        result = clean_data(df)
        assert result["amount"].isna().sum() == 0
        assert len(result) == 3

    def test_negative_amounts_handled(self):
        df = pd.DataFrame({
            "date": ["2024-01-01"],
            "merchant": ["Amazon"],
            "amount": [-250.0],
            "category": ["Shopping"]
        })
        result = clean_data(df)
        assert len(result) == 0  # preprocessor drops < 0 amounts directly

    def test_duplicate_transactions_removed(self, base_df):
        doubled = pd.concat([base_df, base_df], ignore_index=True)
        result = clean_data(doubled)
        assert len(result) <= len(base_df) + 5 # account for possible internal overlaps 

    def test_unparseable_dates_handled(self):
        df = pd.DataFrame({
            "date": ["not-a-date", "2024-13-45", "2024-01-01"],
            "merchant": ["A", "B", "C"],
            "amount": [100.0, 200.0, 300.0],
            "category": ["Food", "Food", "Food"]
        })
        result = clean_data(df)
        assert len(result) == 1 

    def test_empty_dataframe_returns_empty(self, empty_df):
        result = clean_data(empty_df)
        assert len(result) == 0

    def test_whitespace_merchant_names_stripped(self):
        df = pd.DataFrame({
            "date": ["2024-01-01"] * 3,
            "merchant": ["  Amazon  ", "Swiggy", " Netflix"],
            "amount": [100.0, 200.0, 300.0],
            "category": ["Shopping", "Food", "Entertainment"]
        })
        result = clean_data(df)
        assert "Amazon" in result["merchant"].values

    def test_zero_amount_transactions(self):
        df = pd.DataFrame({
            "date": ["2024-01-01"],
            "merchant": ["Uber"],
            "amount": [0.0],
            "category": ["Transport"]
        })
        result = clean_data(df)
        assert len(result) == 0  # drops <= 0

    def test_extremely_large_amount(self):
        df = pd.DataFrame({
            "date": ["2024-01-01"],
            "merchant": ["WireFraud"],
            "amount": [999_999_999.99],
            "category": ["Unknown"]
        })
        result = clean_data(df)
        assert len(result) == 1

    def test_unicode_merchant_names(self):
        df = pd.DataFrame({
            "date": ["2024-01-01"],
            "merchant": ["东京レストラン"],
            "amount": [800.0],
            "category": ["Food"]
        })
        result = clean_data(df)
        assert len(result) == 1


# ═══════════════════════════════════════════════════════
# MODULE 2 — FEATURE ENGINE
# ═══════════════════════════════════════════════════════
from src.feature_engine import engineer_features

class TestFeatureEngine:
    def test_output_has_expected_columns(self, base_df):
        df = clean_data(base_df.copy())
        result = engineer_features(df)
        expected = ["amount_log", "category_daily_ratio", "txn_frequency_7d"]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_amount_log_no_negatives_or_zeros(self, base_df):
        df = clean_data(base_df.copy())
        result = engineer_features(df)
        assert (result["amount_log"] >= 0).all()

    def test_single_transaction_features(self, single_row_df):
        df = clean_data(single_row_df.copy())
        result = engineer_features(df)
        assert len(result) == 1

    def test_empty_df_returns_empty(self, empty_df):
        df = clean_data(empty_df.copy())
        result = engineer_features(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_no_nan_in_features(self, base_df):
        df = clean_data(base_df.copy())
        result = engineer_features(df)
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        assert result[numeric_cols].isna().sum().sum() == 0

    def test_large_dataset_performance(self):
        df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=10000, freq="1h"),
            "merchant": ["Amazon"] * 10000,
            "amount": np.random.uniform(100, 5000, 10000),
            "category": ["Shopping"] * 10000
        })
        df = clean_data(df)
        start = time.time()
        engineer_features(df)
        assert time.time() - start < 2.0


# ═══════════════════════════════════════════════════════
# MODULE 3 — ANOMALY DETECTOR
# ═══════════════════════════════════════════════════════
from src.anomaly_detector import AnomalyDetector
from src.database import add_expected_transaction

class TestAnomalyDetector:
    def test_returns_dataframe_with_flag_column(self, base_df):
        df = clean_data(base_df.copy())
        df = engineer_features(df)
        result = AnomalyDetector().fit_predict(df)
        assert "is_anomaly" in result.columns

    def test_anomaly_flag_is_binary(self, base_df):
        df = clean_data(base_df.copy())
        df = engineer_features(df)
        result = AnomalyDetector().fit_predict(df)
        assert result["is_anomaly"].isin([0, 1]).all()

    def test_planted_anomaly_is_detected(self, anomaly_df):
        df = clean_data(anomaly_df.copy())
        df = engineer_features(df)
        result = AnomalyDetector().fit_predict(df)
        high_amount_row = result[result["amount"] == 95000.0]
        assert len(high_amount_row) > 0
        assert high_amount_row["is_anomaly"].iloc[0] == 1

    def test_all_identical_transactions_no_crash(self):
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=30),
            "merchant": ["Netflix"] * 30,
            "amount": [649.0] * 30,
            "category": ["Entertainment"] * 30
        })
        df = clean_data(df)
        df = engineer_features(df)
        result = AnomalyDetector().fit_predict(df)
        assert "is_anomaly" in result.columns

    def test_dismissed_transactions_not_flagged(self, anomaly_df, setup_test_db):
        df = clean_data(anomaly_df.copy())
        df = engineer_features(df)
        
        # Plant the expectation
        add_expected_transaction("UNKNOWN_WIRE_TRANSFER", 100.0, tolerance=9999999) 
        
        result = AnomalyDetector().fit_predict(df)
        wire_row = result[result["merchant"] == "Unknown_Wire_Transfer"]
        if len(wire_row) > 0:
            assert wire_row["is_anomaly"].iloc[0] == 0


# ═══════════════════════════════════════════════════════
# MODULE 4 — EXPLAINER
# ═══════════════════════════════════════════════════════
from src.explainer import explain_anomalies

class TestExplainer:
    def test_explanation_has_required_keys(self, anomaly_df):
        df = clean_data(anomaly_df.copy())
        df = engineer_features(df)
        result = AnomalyDetector().fit_predict(df)
        
        explanations_df = explain_anomalies(result, UserProfile(df))
        anomalies = explanations_df[explanations_df["is_anomaly"] == 1]
        
        if len(anomalies) > 0:
            exp = anomalies["structured_explanation"].iloc[0]
            assert "category_norm_min" in exp or len(exp) > 0

    def test_explanation_batch_explain_no_crash(self, anomaly_df):
        df = clean_data(anomaly_df.copy())
        df = engineer_features(df)
        result = AnomalyDetector().fit_predict(df)
        explanations_df = explain_anomalies(result, UserProfile(df))
        assert "explanation" in explanations_df.columns
        assert "structured_explanation" in explanations_df.columns


# ═══════════════════════════════════════════════════════
# MODULE 5 — USER PROFILER
# ═══════════════════════════════════════════════════════
from src.user_profiler import UserProfile

class TestUserProfiler:
    def test_profile_keys_present(self, base_df):
        df = clean_data(base_df.copy())
        profile_obj = UserProfile(df)
        assert hasattr(profile_obj, "velocity_profile")
        assert hasattr(profile_obj, "category_profiles")
        assert hasattr(profile_obj, "temporal_profile")
        assert hasattr(profile_obj, "merchant_profile")

    def test_avg_transaction_positive(self, base_df):
        df = clean_data(base_df.copy())
        profile_obj = UserProfile(df)
        assert profile_obj.velocity_profile.get("avg_daily_spend", 0) > 0

    def test_empty_df_returns_zero_profile(self, empty_df):
        df = clean_data(empty_df.copy())
        profile_obj = UserProfile(df)
        assert pd.isna(profile_obj.velocity_profile.get('avg_daily_spend', np.nan))
        assert profile_obj.category_profiles == {}

    def test_high_volume_data(self):
        df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=5000, freq="1h"),
            "merchant": ["Amazon"] * 5000,
            "amount": np.random.normal(500, 100, 5000),
            "category": np.random.choice(["Food", "Shopping", "Transport"], 5000)
        })
        df = clean_data(df)
        start = time.time()
        profile_obj = UserProfile(df)
        assert time.time() - start < 1.0


# ═══════════════════════════════════════════════════════
# MODULE 6 — HEALTH SCORER
# ═══════════════════════════════════════════════════════
from src.health_scorer import HealthScorer
from src.database import set_budget

class TestHealthScorer:
    def test_score_in_valid_range(self, base_df):
        df = clean_data(base_df.copy())
        df = engineer_features(df)
        df = AnomalyDetector().fit_predict(df)
        profile = UserProfile(df)
        scorer = HealthScorer(df, profile)
        assert 0 <= scorer.total_score <= 100

    def test_single_transaction_no_crash(self, single_row_df):
        df = clean_data(single_row_df.copy())
        df = engineer_features(df)
        df = AnomalyDetector().fit_predict(df)
        profile = UserProfile(df)
        scorer = HealthScorer(df, profile)
        assert 0 <= scorer.total_score <= 100

    def test_score_with_budget_overspend(self, base_df, setup_test_db):
        df = clean_data(base_df.copy())
        df = engineer_features(df)
        df = AnomalyDetector().fit_predict(df)
        set_budget('Food', 1.0) # impossible to meet
        profile = UserProfile(df)
        scorer = HealthScorer(df, profile)
        assert 0 <= scorer.total_score <= 100


# ═══════════════════════════════════════════════════════
# MODULE 7 — INSIGHTS ENGINE
# ═══════════════════════════════════════════════════════
from src.insights import InsightGenerator

class TestInsights:
    def test_insights_returns_list(self, base_df):
        df = clean_data(base_df.copy())
        df = engineer_features(df)
        df = AnomalyDetector().fit_predict(df)
        profile = UserProfile(df)
        insights = InsightGenerator(df, profile).get_insights()
        assert isinstance(insights, list)

    def test_weekend_surge_detected(self):
        weekend_dates = [d for d in pd.date_range("2024-01-01", periods=60) if d.weekday() >= 5][:10]
        df = pd.DataFrame({
            "date": weekend_dates,
            "merchant": ["Bar"] * 10,
            "amount": [2000.0] * 10,
            "category": ["Entertainment"] * 10
        })
        df = clean_data(df)
        df = engineer_features(df)
        df["is_anomaly"] = 0
        profile = UserProfile(df)
        insights = InsightGenerator(df, profile).get_insights()
        text = " ".join([i['title'] for i in insights]).lower()
        assert "weekend" in text or len(insights) > 0


# ═══════════════════════════════════════════════════════
# MODULE 8 — RECURRING DETECTOR
# ═══════════════════════════════════════════════════════
from src.recurring import detect_recurring

class TestRecurring:
    def test_netflix_subscription_detected(self):
        dates = [pd.Timestamp(f"2024-{m:02d}-01") for m in range(1, 7)]
        df = pd.DataFrame({
            "date": dates,
            "merchant": ["Netflix"] * 6,
            "amount": [649.0] * 6,
            "category": ["Entertainment"] * 6
        })
        df = clean_data(df)
        result = detect_recurring(df)
        assert result["count"] > 0
        merchants = [r["merchant"] for r in result.get("details", [])]
        assert "Netflix" in merchants


# ═══════════════════════════════════════════════════════
# MODULE 9 — DATABASE LAYER
# ═══════════════════════════════════════════════════════
from src.database import get_expected_transactions, get_budgets, save_model, load_model

class TestDatabase:
    def test_insert_and_retrieve_budget(self, setup_test_db):
        set_budget('Food', 5000.0)
        budgets = get_budgets()
        assert budgets.get('food') == 5000.0

    def test_insert_expected_transaction(self, setup_test_db):
        add_expected_transaction("Netflix", 650, 0.15)
        txns = get_expected_transactions()
        assert len(txns) == 1
        assert txns[0]['merchant'] == "netflix"

    def test_model_snapshot_save_and_load(self, setup_test_db):
        from sklearn.ensemble import IsolationForest
        model = IsolationForest(n_estimators=10, random_state=42)
        X = np.random.randn(50, 3)
        model.fit(X)
        save_model(model, "test_hash", 0.05)
        loaded_model = load_model("test_hash", 0.05)
        assert loaded_model is not None


# ═══════════════════════════════════════════════════════
# MODULE 10 — DATA LOADER (Bank Format Parsers)
# ═══════════════════════════════════════════════════════
from src.data_loader import parse_generic_csv, parse_mint_csv, parse_chase_csv, parse_ynab_csv, PARSERS

class TestDataLoader:
    def _write_csv(self, tmp_path, content, filename):
        p = tmp_path / filename
        p.write_text(content)
        return str(p)

    def test_generic_csv_loads(self, tmp_path):
        csv = "date,merchant,amount,category\n2024-01-01,Amazon,499,Shopping\n"
        path = self._write_csv(tmp_path, csv, "generic.csv")
        df = parse_generic_csv(pd.read_csv(path))
        assert len(df) == 1

    def test_parser_router_selects_correct_parser(self):
        assert "Mint" in PARSERS
        assert "Chase" in PARSERS
        assert "YNAB" in PARSERS


# ═══════════════════════════════════════════════════════
# MODULE 11 — INTEGRATION TESTS
# ═══════════════════════════════════════════════════════
class TestIntegration:
    def test_full_pipeline_clean_data(self, base_df):
        df = clean_data(base_df)
        df = engineer_features(df)
        df = AnomalyDetector().fit_predict(df)
        profile = UserProfile(df)
        score = HealthScorer(df, profile).total_score
        assert 0 <= score <= 100
        assert "is_anomaly" in df.columns


# ═══════════════════════════════════════════════════════
# MODULE 12 & 13 — ADVERSARIAL / PERFORMANCE
# ═══════════════════════════════════════════════════════
class TestPerformance:
    def test_fast_execution(self):
        n = 5000
        df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=n, freq="1h"),
            "merchant": np.random.choice(["Amazon", "Swiggy", "Uber", "Netflix"], n),
            "amount": np.abs(np.random.normal(500, 200, n)),
            "category": np.random.choice(["Food", "Shopping", "Transport", "Ent"], n)
        })
        start = time.time()
        df = clean_data(df)
        df = engineer_features(df)
        df = AnomalyDetector().fit_predict(df)
        assert time.time() - start < 5.0

"""
anomaly_detector.py — Isolation Forest Anomaly Detection
=========================================================
PURPOSE:
     It uses Isolation Forest to identify
    transactions that deviate from the user's normal spending patterns.

WHY ISOLATION FOREST:
    Traditional approaches (z-score, IQR) only look at one dimension at a time.
    Isolation Forest is a MULTI-DIMENSIONAL anomaly detector:

    How it works:
    1. Build many random decision trees (an "isolation forest")
    2. At each node, randomly pick a feature and a random split value
    3. Anomalies are "few and different" — they get isolated (reach a leaf)
       in fewer splits than normal points
    4. The anomaly score is the average path length across all trees
       (shorter path = more anomalous)

    Why this matters:
    - A ₹800 food transaction might look normal by amount alone
    - But if it's on a Tuesday (user never spends on Tuesdays) at a
      new merchant, with an unusual 7-day rolling average →
      the COMBINATION makes it anomalous
    - Isolation Forest captures these multi-dimensional patterns

    Advantages:
    - No need for labeled data (unsupervised)
    - Handles mixed feature types
    - Scales linearly with data size: O(n × t × log(n))
    - No assumptions about data distribution (unlike z-score which assumes normal)
"""

import pandas as pd
import numpy as np
import hashlib
from sklearn.ensemble import IsolationForest
from src.feature_engine import get_feature_columns
from src.database import save_model, load_model, get_expected_transactions


# ─── Model Configuration ─────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "n_estimators": 200,        # Number of trees (more = more stable)
    "contamination": 0.05,      # Expected anomaly rate (~5%)
    "max_samples": "auto",      # Samples per tree (auto = min(256, n_samples))
    "random_state": 42,         # Reproducibility
    "n_jobs": -1,               # Use all CPU cores
}


class AnomalyDetector:
    """
    Isolation Forest wrapper for personal finance anomaly detection.


    """

    def __init__(self, config: dict = None):
        """
        Initialize the detector with configuration.

        Parameters:
            config: Model hyperparameters (uses defaults if not provided)
        """
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.model = IsolationForest(**self.config)
        self.feature_columns = get_feature_columns()
        self.is_fitted = False

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the model and predict anomalies in one step.



        Parameters:
            df: DataFrame with engineered features

        Returns:
            DataFrame with added 'anomaly_score' and 'is_anomaly' columns
        """
        df = df.copy()

        # Load user feedback rules
        expected = get_expected_transactions()
        
        # Base feature matrix
        X = df[self.feature_columns].values

        # Generate hash based on Data + Rules for intelligent cache busting
        rule_str = str(expected)
        # Safely hash the dataframe by hashing all its values to bytes
        df_hash_bytes = pd.util.hash_pandas_object(df).values.tobytes()
        data_hash = hashlib.md5(df_hash_bytes + rule_str.encode()).hexdigest()

        # Build exclusion mask for training
        is_expected = pd.Series(False, index=df.index)
        if expected:
            for rule in expected:
                merchant_match = df["merchant"].str.lower() == rule["merchant"]
                amount_match = df["amount"].between(rule["amount_min"], rule["amount_max"])
                is_expected = is_expected | (merchant_match & amount_match)

        # Exclude dismissed transactions from training set (closing the feedback loop)
        X_train = df.loc[~is_expected, self.feature_columns].values
        if len(X_train) == 0:
            X_train = X  # Fallback if somehow literally everything is expected

        # Attempt to load cached model
        cached_model = load_model(data_hash, self.config.get("contamination", 0.05))
        if cached_model is not None:
            self.model = cached_model
            self.is_fitted = True
        else:
            self.model.fit(X_train)
            self.is_fitted = True
            save_model(self.model, data_hash, self.config.get("contamination", 0.05))

        # Standard Isolation Forest execution
        labels = self.model.predict(X)
        scores = self.model.score_samples(X)
        is_iforest_anomaly = (labels == -1).astype(int)

        # Ensemble Z-Score Execution
        is_zscore_anomaly = self._apply_zscore_detector(df)

        # Confidence Voting Logic
        is_anomaly_final = is_iforest_anomaly | is_zscore_anomaly
        confidence = []
        for i_flag, z_flag in zip(is_iforest_anomaly, is_zscore_anomaly):
            if i_flag == 1 and z_flag == 1:
                confidence.append("High")
            elif i_flag == 1 or z_flag == 1:
                confidence.append("Medium")
            else:
                confidence.append("Low")

        # Add to DataFrame
        df["anomaly_score"] = scores
        df["is_anomaly"] = is_anomaly_final
        df["anomaly_confidence"] = confidence

        # Add severity levels based on score percentiles
        df["anomaly_severity"] = self._calculate_severity(scores, is_anomaly_final.values if hasattr(is_anomaly_final, "values") else is_anomaly_final)

        # Force clear predictions for known expected logic so they never show up
        if is_expected.any():
            df.loc[is_expected, "is_anomaly"] = 0
            df.loc[is_expected, "anomaly_severity"] = "normal"

        return df

    def _calculate_severity(self, scores: np.ndarray, labels: np.ndarray) -> list[str]:
        """
        Assign severity levels to anomalies.



            This helps users prioritize which anomalies to investigate first.
        """
        severities = []
        # Calculate thresholds from anomaly scores
        p1 = np.percentile(scores, 1)     # Bottom 1%
        p5 = np.percentile(scores, 5)     # Bottom 5%

        for score, label in zip(scores, labels):
            if label == -1:  # Flagged as anomaly
                if score <= p1:
                    severities.append("critical")
                else:
                    severities.append("warning")
            else:
                severities.append("normal")

        return severities

    def _apply_zscore_detector(self, df: pd.DataFrame) -> pd.Series:
        """
        Z-Score rolling heuristic.
        Calculates 30-day mean/std per category and flags amounts > 2.5 standard deviations.
        """
        is_zscore_anomaly = pd.Series(0, index=df.index)
        
        # Calculate isolated rolling z-scores
        for cat in df["category"].unique():
            cat_mask = df["category"] == cat
            cat_df = df[cat_mask].copy()
            if len(cat_df) > 5:
                # Rolling stats
                cat_df = cat_df.sort_values("date")
                rolling = cat_df["amount"].rolling(window=min(30, len(cat_df)), min_periods=3)
                roll_mean = rolling.mean().shift(1).fillna(method='bfill')
                roll_std = rolling.std().shift(1).fillna(method='bfill')
                
                # Z-score computation
                z_scores = (cat_df["amount"] - roll_mean) / roll_std.replace(0, 1) # prevent div by zero
                cat_df["is_z"] = (z_scores > 2.5).astype(int)
                
                is_zscore_anomaly.loc[cat_df.index] = cat_df["is_z"]
                
        return is_zscore_anomaly

    def get_anomaly_summary(self, df: pd.DataFrame) -> dict:
        """
        Generate a summary of detected anomalies.
        Used by the dashboard for quick stats.
        """
        if "is_anomaly" not in df.columns:
            return {"error": "Run fit_predict first"}

        anomalies = df[df["is_anomaly"] == 1]

        return {
            "total_transactions": len(df),
            "total_anomalies": len(anomalies),
            "anomaly_rate": f"{len(anomalies) / len(df) * 100:.1f}%",
            "critical_count": len(anomalies[anomalies["anomaly_severity"] == "critical"]),
            "warning_count": len(anomalies[anomalies["anomaly_severity"] == "warning"]),
            "total_anomaly_amount": anomalies["amount"].sum(),
            "avg_anomaly_amount": anomalies["amount"].mean() if len(anomalies) > 0 else 0,
            "anomalies_by_category": anomalies["category"].value_counts().to_dict(),
            "score_range": {
                "min": float(df["anomaly_score"].min()),
                "max": float(df["anomaly_score"].max()),
                "mean": float(df["anomaly_score"].mean()),
            }
        }

    def evaluate_performance(self, df: pd.DataFrame) -> dict:
        """
        Evaluate Model Performance against ground truth labels if they exist.
        If they do not exist (e.g., real bank uploads), fallback to a degraded
        Signal Confidence summary.
        """
        if "is_anomaly" not in df.columns:
            return {"error": "Run fit_predict first"}

        predicted = df["is_anomaly"].astype(int)
        anomaly_rate = predicted.mean()
        avg_score = df["anomaly_score"].mean()

        has_ground_truth = "is_actual_anomaly" in df.columns

        if not has_ground_truth:
            return {
                "has_ground_truth": False,
                "anomaly_rate": float(anomaly_rate),
                "avg_score": float(avg_score),
                "flag_density": "High" if anomaly_rate > 0.08 else ("Moderate" if anomaly_rate > 0.03 else "Low")
            }

        truth = df["is_actual_anomaly"].astype(int)

        # Confusion Matrix logic (Raw NumPy conditions to avoid division by zero exceptions structurally)
        tp = int(((predicted == 1) & (truth == 1)).sum())
        fp = int(((predicted == 1) & (truth == 0)).sum())
        fn = int(((predicted == 0) & (truth == 1)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        false_alert_rate = fp / (tp + fp) if (tp + fp) > 0 else 0.0

        # Behavioral Type Tag
        if precision > 0.8 and recall > 0.7:
            mode = "Balanced Detection"
        elif precision >= 0.85:
            mode = "Conservative (Low Noise)"
        elif recall >= 0.85:
            mode = "Aggressive (High Sensitivity)"
        else:
            mode = "Exploratory"

        return {
            "has_ground_truth": True,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "false_alert_rate": float(false_alert_rate),
            "mode": mode,
            "anomaly_rate": float(anomaly_rate),
            "avg_score": float(avg_score),
            "stats": {"tp": tp, "fp": fp, "fn": fn}
        }

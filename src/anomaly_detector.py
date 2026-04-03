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
from sklearn.ensemble import IsolationForest
from src.feature_engine import get_feature_columns


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

        # Extract feature matrix
        X = df[self.feature_columns].values

        # Fit and predict
        self.model.fit(X)
        self.is_fitted = True

        # Get anomaly labels: -1 = anomaly, 1 = normal
        labels = self.model.predict(X)

        # Get anomaly scores (more negative = more anomalous)
        # score_samples returns the anomaly score of the input samples
        # Lower scores indicate more abnormal
        scores = self.model.score_samples(X)

        # Add to DataFrame
        df["anomaly_score"] = scores
        df["is_anomaly"] = (labels == -1).astype(int)

        # Add severity levels based on score percentiles
        df["anomaly_severity"] = self._calculate_severity(scores, labels)

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

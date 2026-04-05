"""
anomaly_detector.py — Ensemble signal engine
===========================================
Combines Isolation Forest with category-aware z-score logic to produce
warning/critical behavioral signals while suppressing known transfer noise.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.database import get_expected_transactions, load_model, save_model
from src.feature_engine import get_feature_columns


DEFAULT_CONFIG = {
    "n_estimators": 200,
    "contamination": 0.05,
    "max_samples": "auto",
    "random_state": 42,
    "n_jobs": -1,
}


class AnomalyDetector:
    """Isolation Forest + z-score ensemble wrapper."""

    def __init__(self, config: dict | None = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.model = IsolationForest(**self.config)
        self.feature_columns = get_feature_columns()
        self.is_fitted = False

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit the ensemble and return the input DataFrame with signal columns added."""
        df = df.copy()

        if len(df) == 0:
            for column, default in {
                "anomaly_score": [],
                "is_anomaly": [],
                "anomaly_confidence": [],
                "anomaly_severity": [],
            }.items():
                if column not in df.columns:
                    df[column] = default
            return df

        for column in self.feature_columns:
            if column not in df.columns:
                df[column] = 0.0

        expected_rules = get_expected_transactions()
        noise_mask = df.get("is_transfer", pd.Series(False, index=df.index)).fillna(False).astype(bool)
        expected_mask = self._build_expected_mask(df, expected_rules)

        feature_frame = df[self.feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        data_hash = self._build_data_hash(df, expected_rules)

        train_mask = ~(noise_mask | expected_mask)
        if train_mask.sum() == 0:
            train_mask = ~expected_mask
        x_train = feature_frame.loc[train_mask].values
        x_all = feature_frame.values

        if len(x_train) >= 8:
            cached_model = load_model(data_hash, self.config.get("contamination", 0.05))
            if cached_model is not None:
                self.model = cached_model
                self.is_fitted = True
            else:
                self.model.fit(x_train)
                self.is_fitted = True
                save_model(self.model, data_hash, self.config.get("contamination", 0.05))
            iso_scores = self.model.score_samples(x_all)
            iso_flags = (self.model.predict(x_all) == -1).astype(int)
        else:
            self.is_fitted = False
            iso_scores = np.zeros(len(df))
            iso_flags = np.zeros(len(df), dtype=int)

        z_values, z_flags = self._apply_zscore_detector(df)
        severity, confidence = self._combine_signals(iso_flags, z_flags)

        df["signal_iso_flag"] = iso_flags
        df["signal_z_flag"] = z_flags
        df["signal_z_score"] = z_values
        df["anomaly_score"] = iso_scores
        df["is_anomaly"] = (np.array(iso_flags) | np.array(z_flags)).astype(int)
        df["anomaly_confidence"] = confidence
        df["anomaly_severity"] = severity

        suppress_mask = noise_mask | expected_mask
        if suppress_mask.any():
            df.loc[suppress_mask, "is_anomaly"] = 0
            df.loc[suppress_mask, "anomaly_confidence"] = "low"
            df.loc[suppress_mask, "anomaly_severity"] = "normal"
            df.loc[suppress_mask, "signal_iso_flag"] = 0
            df.loc[suppress_mask, "signal_z_flag"] = 0
            df.loc[suppress_mask, "signal_z_score"] = 0.0

        return df

    def _build_data_hash(self, df: pd.DataFrame, expected_rules: list[dict]) -> str:
        """Build a cache key that invalidates on data or feedback changes."""
        hash_cols = ["date", "amount", "merchant_normalized", "category", "type", "is_transfer"]
        existing_cols = [column for column in hash_cols if column in df.columns]
        data_bytes = pd.util.hash_pandas_object(df[existing_cols], index=True).values.tobytes()
        rule_bytes = repr(expected_rules).encode("utf-8")
        return hashlib.md5(data_bytes + rule_bytes).hexdigest()

    def _build_expected_mask(self, df: pd.DataFrame, expected_rules: list[dict]) -> pd.Series:
        """Return rows the user already marked as expected."""
        mask = pd.Series(False, index=df.index)
        if not expected_rules:
            return mask

        merchant_series = df.get("merchant_normalized", df.get("merchant", "")).astype(str).str.lower()
        for rule in expected_rules:
            merchant_match = merchant_series == str(rule["merchant"]).lower()
            amount_match = df["amount"].between(rule["amount_min"], rule["amount_max"])
            mask = mask | (merchant_match & amount_match)
        return mask

    def _apply_zscore_detector(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Use category-level z-scores as the second signal source."""
        z_values = df.get("amount_zscore", pd.Series(0.0, index=df.index)).abs().fillna(0.0)
        category_counts = df.groupby("category")["category"].transform("size").fillna(0)
        noise_mask = df.get("is_transfer", pd.Series(False, index=df.index)).fillna(False).astype(bool)
        flags = ((z_values > 2.5) & (category_counts >= 4) & (~noise_mask)).astype(int)
        return z_values.astype(float), flags.astype(int)

    def _combine_signals(self, iso_flags: np.ndarray, z_flags: pd.Series) -> tuple[list[str], list[str]]:
        """Map ensemble votes to user-facing severity and confidence."""
        severity = []
        confidence = []
        for iso_flag, z_flag in zip(iso_flags, z_flags):
            if iso_flag and z_flag:
                severity.append("critical")
                confidence.append("high")
            elif iso_flag or z_flag:
                severity.append("warning")
                confidence.append("medium")
            else:
                severity.append("normal")
                confidence.append("low")
        return severity, confidence

    def get_anomaly_summary(self, df: pd.DataFrame) -> dict:
        """Generate a quick anomaly summary for the dashboard."""
        if "is_anomaly" not in df.columns:
            return {"error": "Run fit_predict first"}

        anomalies = df[df["is_anomaly"] == 1]
        return {
            "total_transactions": len(df),
            "total_anomalies": len(anomalies),
            "anomaly_rate": f"{(len(anomalies) / max(len(df), 1)) * 100:.1f}%",
            "critical_count": int((anomalies["anomaly_severity"] == "critical").sum()),
            "warning_count": int((anomalies["anomaly_severity"] == "warning").sum()),
            "total_anomaly_amount": float(anomalies["amount"].sum()),
            "avg_anomaly_amount": float(anomalies["amount"].mean()) if len(anomalies) > 0 else 0.0,
            "anomalies_by_category": anomalies["category"].value_counts().to_dict(),
            "score_range": {
                "min": float(df["anomaly_score"].min()),
                "max": float(df["anomaly_score"].max()),
                "mean": float(df["anomaly_score"].mean()),
            },
        }

    def evaluate_performance(self, df: pd.DataFrame) -> dict:
        """Evaluate model performance using ground truth when available."""
        if "is_anomaly" not in df.columns:
            return {"error": "Run fit_predict first"}

        predicted = df["is_anomaly"].astype(int)
        anomaly_rate = predicted.mean() if len(predicted) else 0.0
        avg_score = df["anomaly_score"].mean() if "anomaly_score" in df.columns and len(df) else 0.0

        if "is_actual_anomaly" not in df.columns:
            return {
                "has_ground_truth": False,
                "anomaly_rate": float(anomaly_rate),
                "avg_score": float(avg_score),
                "flag_density": "High" if anomaly_rate > 0.08 else ("Moderate" if anomaly_rate > 0.03 else "Low"),
            }

        truth = df["is_actual_anomaly"].astype(int)
        tp = int(((predicted == 1) & (truth == 1)).sum())
        fp = int(((predicted == 1) & (truth == 0)).sum())
        fn = int(((predicted == 0) & (truth == 1)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        false_alert_rate = fp / (tp + fp) if (tp + fp) > 0 else 0.0

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
            "stats": {"tp": tp, "fp": fp, "fn": fn},
        }

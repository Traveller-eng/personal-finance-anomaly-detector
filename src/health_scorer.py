"""
health_scorer.py — Financial health score
=========================================
Computes a composite 0-100 score over the same noise-filtered behavioral view
used by the insight engine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.user_profiler import UserProfile


class HealthScorer:
    """Calculate a 0-100 health score from transaction data and user profile."""

    MAX_SCORE = 100
    COMPONENT_WEIGHT = 20

    def __init__(self, df: pd.DataFrame, profile: UserProfile):
        self.raw_df = df.copy() if df is not None else pd.DataFrame()
        self.df = self._filter_behavioral_spend(self.raw_df)
        self.profile = profile

        self.components = {
            "budget_adherence": self._score_budget_adherence(),
            "spending_consistency": self._score_spending_consistency(),
            "anomaly_rate": self._score_anomaly_rate(),
            "category_balance": self._score_category_balance(),
            "trend_direction": self._score_trend_direction(),
        }
        self.total_score = sum(self.components.values())

    def _filter_behavioral_spend(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or len(df) == 0:
            return pd.DataFrame(columns=df.columns if df is not None else [])

        filtered = df.copy()
        if "is_transfer" in filtered.columns:
            filtered = filtered[~filtered["is_transfer"].fillna(False).astype(bool)]
        if "category" in filtered.columns:
            filtered = filtered[~filtered["category"].astype(str).str.lower().isin(["income_transfer", "personal_transfer"])]
        return filtered

    def _score_budget_adherence(self) -> float:
        category_profiles = getattr(self.profile, "category_profiles", {})
        if not isinstance(category_profiles, dict):
            category_profiles = {}

        if len(self.df) == 0 or not category_profiles:
            return self.COMPONENT_WEIGHT * 0.5

        within_range_counts = []
        for category, profile in category_profiles.items():
            category_df = self.df[self.df["category"] == category]
            if len(category_df) == 0:
                continue

            low, high = profile["normal_range"]
            within = ((category_df["amount"] >= low) & (category_df["amount"] <= high)).mean()
            within_range_counts.append(within)

        if not within_range_counts:
            return self.COMPONENT_WEIGHT * 0.5

        return round(float(np.mean(within_range_counts)) * self.COMPONENT_WEIGHT, 1)

    def _score_spending_consistency(self) -> float:
        if len(self.df) < 2:
            return self.COMPONENT_WEIGHT * 0.5

        daily_spend = self.df.groupby("date")["amount"].sum()
        if len(daily_spend) < 2 or daily_spend.mean() <= 0:
            return self.COMPONENT_WEIGHT * 0.5

        cv = daily_spend.std() / daily_spend.mean()
        score = max(0, 1 - (cv / 2.0)) * self.COMPONENT_WEIGHT
        return round(float(score), 1)

    def _score_anomaly_rate(self) -> float:
        if "is_anomaly" not in self.df.columns or len(self.df) == 0:
            return self.COMPONENT_WEIGHT * 0.5

        anomaly_rate = self.df["is_anomaly"].mean()
        score = max(0, 1 - (anomaly_rate * 5)) * self.COMPONENT_WEIGHT
        return round(float(min(score, self.COMPONENT_WEIGHT)), 1)

    def _score_category_balance(self) -> float:
        if len(self.df) == 0:
            return self.COMPONENT_WEIGHT * 0.5

        category_totals = self.df.groupby("category")["amount"].sum()
        if len(category_totals) == 0 or category_totals.sum() <= 0:
            return self.COMPONENT_WEIGHT * 0.5

        proportions = category_totals / category_totals.sum()
        entropy = -np.sum(proportions * np.log2(proportions + 1e-10))
        max_entropy = np.log2(len(proportions)) if len(proportions) > 1 else 1
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        return round(float(normalized_entropy * self.COMPONENT_WEIGHT), 1)

    def _score_trend_direction(self) -> float:
        if len(self.df) < 7:
            return self.COMPONENT_WEIGHT * 0.75

        daily_spend = self.df.groupby("date")["amount"].sum().sort_index()
        if len(daily_spend) < 7:
            return self.COMPONENT_WEIGHT * 0.75

        rolling = daily_spend.rolling(window=7, min_periods=1).mean()
        x_axis = np.arange(len(rolling))
        slope = np.polyfit(x_axis, rolling.values, 1)[0]
        mean_spend = daily_spend.mean()
        normalized_slope = slope / mean_spend if mean_spend > 0 else 0

        if normalized_slope <= -0.01:
            score = self.COMPONENT_WEIGHT
        elif normalized_slope <= 0:
            score = self.COMPONENT_WEIGHT * 0.85
        elif normalized_slope <= 0.01:
            score = self.COMPONENT_WEIGHT * 0.7
        else:
            penalty = min(normalized_slope * 50, 1.0)
            score = self.COMPONENT_WEIGHT * max(0.2, 0.7 - penalty)

        return round(float(score), 1)

    def get_score_breakdown(self) -> dict:
        descriptions = {
            "budget_adherence": "How often you stay inside your normal category ranges",
            "spending_consistency": "How stable your day-to-day spending pattern is",
            "anomaly_rate": "How often the signal engine flagged unusual transactions",
            "category_balance": "How diversified your spending mix is across categories",
            "trend_direction": "Whether your spending trajectory is stabilizing or accelerating",
        }
        labels = {
            "budget_adherence": "Budget Adherence",
            "spending_consistency": "Spending Consistency",
            "anomaly_rate": "Anomaly Rate",
            "category_balance": "Category Balance",
            "trend_direction": "Trend Direction",
        }

        breakdown = {
            "total_score": round(self.total_score, 1),
            "grade": self._get_grade(),
            "components": {},
        }

        for key, score in self.components.items():
            breakdown["components"][key] = {
                "label": labels[key],
                "score": score,
                "max": self.COMPONENT_WEIGHT,
                "percentage": round(score / self.COMPONENT_WEIGHT * 100, 1),
                "description": descriptions[key],
            }

        return breakdown

    def _get_grade(self) -> str:
        score = self.total_score
        if score >= 90:
            return "A+"
        if score >= 80:
            return "A"
        if score >= 70:
            return "B+"
        if score >= 60:
            return "B"
        if score >= 50:
            return "C"
        if score >= 40:
            return "D"
        return "F"

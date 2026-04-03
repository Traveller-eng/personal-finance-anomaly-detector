"""
health_scorer.py — Financial Health Score (0-100)
==================================================
PURPOSE:
    Reduces complex spending behavior into a single, intuitive number (0-100)
    that tells the user "how healthy are my finances?" at a glance.

    This is like a credit score, but for SPENDING BEHAVIOR rather than
    creditworthiness. It rewards consistency, budget adherence, and
    balanced spending across categories.

WHY A COMPOSITE SCORE:
    A single metric is powerful for two reasons:
    1. COMMUNICATION: "Your health score is 72" is instantly understandable
    2. TRACKING: Users can see if their score improves over weeks/months

    But a single number can be misleading, so we break it into 5 components
    (20 points each) that users can drill into:
    - Budget Adherence: Are you staying within your normal ranges?
    - Spending Consistency: Is your spending predictable?
    - Anomaly Rate: How many unusual transactions do you have?
    - Category Balance: Is spending well-distributed?
    - Trend Direction: Is spending trending up or down?
"""

import pandas as pd
import numpy as np
from src.user_profiler import UserProfile


class HealthScorer:
    """
    Calculates a 0-100 financial health score from transaction data.


    """

    MAX_SCORE = 100
    COMPONENT_WEIGHT = 20  # Each component worth 20 points

    def __init__(self, df: pd.DataFrame, profile: UserProfile):
        """
        Calculate the health score from transaction data and user profile.

        Parameters:
            df: Feature-engineered DataFrame with anomaly detection results
            profile: User's spending profile
        """
        self.df = df
        self.profile = profile

        # Calculate each component
        self.components = {
            "budget_adherence": self._score_budget_adherence(),
            "spending_consistency": self._score_spending_consistency(),
            "anomaly_rate": self._score_anomaly_rate(),
            "category_balance": self._score_category_balance(),
            "trend_direction": self._score_trend_direction(),
        }

        # Total score
        self.total_score = sum(self.components.values())

    def _score_budget_adherence(self) -> float:
        """
        How well does the user stay within their normal spending ranges?



        Scoring:
            - 100% within range → 20 points
            - 80% within range → 16 points
            - <50% within range → <10 points
        """
        within_range_counts = []

        for category, prof in self.profile.category_profiles.items():
            cat_df = self.df[self.df["category"] == category]
            if len(cat_df) == 0:
                continue

            low, high = prof["normal_range"]
            within = ((cat_df["amount"] >= low) & (cat_df["amount"] <= high)).mean()
            within_range_counts.append(within)

        if not within_range_counts:
            return self.COMPONENT_WEIGHT * 0.5  # Default to middle score

        avg_adherence = np.mean(within_range_counts)
        return round(avg_adherence * self.COMPONENT_WEIGHT, 1)

    def _score_spending_consistency(self) -> float:
        """
        How consistent (predictable) is daily spending?



            CV is better than raw std because it's scale-independent:
            a std of ₹500 means different things for someone spending ₹1000/day
            vs. ₹50,000/day.

        Scoring:
            - CV < 0.3 (very consistent): 20 points
            - CV 0.3-0.8 (moderate): 10-18 points
            - CV > 1.5 (very erratic): < 5 points
        """
        daily_spend = self.df.groupby("date")["amount"].sum()

        if len(daily_spend) < 2:
            return self.COMPONENT_WEIGHT * 0.5

        cv = daily_spend.std() / daily_spend.mean() if daily_spend.mean() > 0 else 1.0

        # Map CV to score: lower CV = higher score
        # CV of 0 → 20, CV of 2+ → 0
        score = max(0, 1 - (cv / 2.0)) * self.COMPONENT_WEIGHT
        return round(score, 1)

    def _score_anomaly_rate(self) -> float:
        """
        Fewer anomalies = healthier spending.



        Scoring:
            score = (1 - anomaly_rate) × 20, with anomaly_rate capped at 1.0
        """
        if "is_anomaly" not in self.df.columns:
            return self.COMPONENT_WEIGHT * 0.5

        anomaly_rate = self.df["is_anomaly"].mean()
        # Scale so even 10% anomaly rate significantly reduces score
        score = max(0, 1 - (anomaly_rate * 5)) * self.COMPONENT_WEIGHT
        return round(min(score, self.COMPONENT_WEIGHT), 1)

    def _score_category_balance(self) -> float:
        """
        Is spending well-distributed across categories?



            Shannon Entropy: H = -Σ(p_i × log2(p_i))
            Max entropy = log2(n_categories) (uniform distribution)

        Why entropy?
            Someone spending 90% on food and 10% on everything else has
            a spending concentration problem. Balanced spending across
            needs (food, transport, bills, etc.) indicates financial stability.
        """
        category_totals = self.df.groupby("category")["amount"].sum()
        proportions = category_totals / category_totals.sum()

        # Shannon entropy
        entropy = -np.sum(proportions * np.log2(proportions + 1e-10))
        max_entropy = np.log2(len(proportions)) if len(proportions) > 1 else 1

        # Normalize to 0-1 range
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        return round(normalized_entropy * self.COMPONENT_WEIGHT, 1)

    def _score_trend_direction(self) -> float:
        """
        Is spending trending down (good) or up (concerning)?



            This is PREDICTIVE — it tells the user about their trajectory,
            not just their current state.

        Scoring:
            - Strong downward trend: 20 points
            - Flat trend: 15 points (stable is good)
            - Strong upward trend: 5 points (concerning)
        """
        daily_spend = self.df.groupby("date")["amount"].sum().sort_index()

        if len(daily_spend) < 7:
            return self.COMPONENT_WEIGHT * 0.75  # Not enough data, assume decent

        # Rolling 7-day average to smooth noise
        rolling = daily_spend.rolling(window=7, min_periods=1).mean()

        # Linear regression slope
        x = np.arange(len(rolling))
        slope = np.polyfit(x, rolling.values, 1)[0]

        # Normalize slope relative to mean spending
        mean_spend = daily_spend.mean()
        normalized_slope = slope / mean_spend if mean_spend > 0 else 0

        # Map: negative slope → high score, positive → low score
        # flat (slope ≈ 0) → 15 points
        if normalized_slope <= -0.01:
            score = self.COMPONENT_WEIGHT  # Strongly decreasing = perfect
        elif normalized_slope <= 0:
            score = self.COMPONENT_WEIGHT * 0.85  # Slightly decreasing = great
        elif normalized_slope <= 0.01:
            score = self.COMPONENT_WEIGHT * 0.7  # Flat = good
        else:
            # Increasing: penalize proportionally
            penalty = min(normalized_slope * 50, 1.0)
            score = self.COMPONENT_WEIGHT * max(0.2, 0.7 - penalty)

        return round(score, 1)

    def get_score_breakdown(self) -> dict:
        """
        Get the full score breakdown for display.

        Returns dict with total score, each component score,
        descriptions, and improvement suggestions.
        """
        descriptions = {
            "budget_adherence": "Staying within your normal spending ranges per category",
            "spending_consistency": "How predictable and stable your daily spending is",
            "anomaly_rate": "Fewer unusual transactions = healthier spending",
            "category_balance": "Spending distributed well across different categories",
            "trend_direction": "Whether your spending is trending up or down over time",
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
            "components": {}
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
        """Convert score to a letter grade."""
        s = self.total_score
        if s >= 90:
            return "A+"
        elif s >= 80:
            return "A"
        elif s >= 70:
            return "B+"
        elif s >= 60:
            return "B"
        elif s >= 50:
            return "C"
        elif s >= 40:
            return "D"
        else:
            return "F"

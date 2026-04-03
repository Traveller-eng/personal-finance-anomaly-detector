"""
user_profiler.py — Behavioral Spending Profiling
==================================================
PURPOSE (Interview Talking Point):
    Creates a "spending fingerprint" for the user — a comprehensive profile
    of their normal financial behavior. This profile serves as the BASELINE
    against which anomalies are explained.

    Without a profile, we can only say "this is unusual."
    With a profile, we can say "this is unusual BECAUSE your average food
    spend is ₹400 and this is ₹2200 — that's 5.5× your normal."

WHY THIS MATTERS:
    This is what separates a basic anomaly flagger from a behavior-aware
    financial assistant. The profile enables:
    1. Personalized explanations
    2. Category-specific baselines
    3. Temporal pattern detection
    4. Financial health scoring
"""

import pandas as pd
import numpy as np


class UserProfile:
    """
    Comprehensive spending behavior profile for a single user.

    Interview Talking Points:
        - Built from historical transaction data
        - Captures 5 dimensions of behavior:
          1. Per-category spending (mean, median, std, min, max)
          2. Temporal patterns (weekday vs weekend, monthly distribution)
          3. Merchant preferences (top merchants per category)
          4. Spending velocity (daily spend rate and trend)
          5. Budget utilization (how spending distributes across the month)
    """

    def __init__(self, df: pd.DataFrame, currency_symbol: str = "₹"):
        """
        Build the user profile from transaction data.

        Parameters:
            df: Cleaned, feature-engineered DataFrame
            currency_symbol: Currency symbol for display (configurable)
        """
        self.currency = currency_symbol
        self.total_transactions = len(df)
        self.date_range = (df["date"].min(), df["date"].max())
        self.total_days = (self.date_range[1] - self.date_range[0]).days + 1

        # Build all profile components
        self.category_profiles = self._build_category_profiles(df)
        self.temporal_profile = self._build_temporal_profile(df)
        self.merchant_profile = self._build_merchant_profile(df)
        self.velocity_profile = self._build_velocity_profile(df)
        self.overall_stats = self._build_overall_stats(df)

    def _build_category_profiles(self, df: pd.DataFrame) -> dict:
        """
        Per-category spending statistics.

        Interview Deep-Dive:
            This is the heart of the profiler. For each category, we compute:
            - mean: The baseline expectation
            - std: How much variation is normal
            - percentiles: What range of amounts is typical (25th-75th)
            - frequency: How often the user transacts in this category

            The "normal range" is mean ± 1.5 × std — anything outside is
            potentially anomalous in that category.
        """
        profiles = {}
        for category, group in df.groupby("category"):
            profiles[category] = {
                "mean": round(group["amount"].mean(), 2),
                "median": round(group["amount"].median(), 2),
                "std": round(group["amount"].std(), 2) if len(group) > 1 else 0,
                "min": round(group["amount"].min(), 2),
                "max": round(group["amount"].max(), 2),
                "p25": round(group["amount"].quantile(0.25), 2),
                "p75": round(group["amount"].quantile(0.75), 2),
                "total_spend": round(group["amount"].sum(), 2),
                "transaction_count": len(group),
                "avg_txns_per_week": round(len(group) / (self.total_days / 7), 2),
                "normal_range": (
                    round(max(0, group["amount"].mean() - 1.5 * group["amount"].std()), 2),
                    round(group["amount"].mean() + 1.5 * group["amount"].std(), 2)
                ) if len(group) > 1 else (group["amount"].mean(), group["amount"].mean()),
                "share_of_total": round(
                    group["amount"].sum() / df["amount"].sum() * 100, 1
                )
            }
        return profiles

    def _build_temporal_profile(self, df: pd.DataFrame) -> dict:
        """
        When does the user typically spend?

        Captures:
        - Day-of-week distribution (weekday vs weekend preference)
        - Monthly spending distribution
        - Weekly spending average
        """
        # Day of week spending
        dow_spend = df.groupby("day_of_week")["amount"].agg(["mean", "sum", "count"])
        dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        weekday_avg = df[df["is_weekend"] == 0]["amount"].mean() if len(df[df["is_weekend"] == 0]) > 0 else 0
        weekend_avg = df[df["is_weekend"] == 1]["amount"].mean() if len(df[df["is_weekend"] == 1]) > 0 else 0

        # Monthly spending
        monthly_spend = df.groupby("month")["amount"].sum()

        return {
            "day_of_week_avg_spend": {
                dow_names[i]: round(dow_spend.loc[i, "mean"], 2) if i in dow_spend.index else 0
                for i in range(7)
            },
            "day_of_week_txn_count": {
                dow_names[i]: int(dow_spend.loc[i, "count"]) if i in dow_spend.index else 0
                for i in range(7)
            },
            "weekday_avg": round(weekday_avg, 2),
            "weekend_avg": round(weekend_avg, 2),
            "weekend_multiplier": round(weekend_avg / weekday_avg, 2) if weekday_avg > 0 else 0,
            "monthly_spend": {int(m): round(v, 2) for m, v in monthly_spend.items()},
            "highest_spend_month": int(monthly_spend.idxmax()) if len(monthly_spend) > 0 else None,
            "lowest_spend_month": int(monthly_spend.idxmin()) if len(monthly_spend) > 0 else None,
        }

    def _build_merchant_profile(self, df: pd.DataFrame) -> dict:
        """
        Merchant preferences per category.

        Interview Deep-Dive:
            Tracking merchant frequency helps detect "new merchant" anomalies.
            If a user always orders from Swiggy and suddenly has a large
            transaction at "Unknown Store", that's suspicious even if the
            amount is normal.
        """
        merchant_profile = {}
        for category, group in df.groupby("category"):
            top_merchants = (
                group.groupby("merchant")["amount"]
                .agg(["count", "sum", "mean"])
                .sort_values("count", ascending=False)
                .head(5)
            )
            merchant_profile[category] = {
                "top_merchants": [
                    {
                        "name": merchant,
                        "count": int(row["count"]),
                        "total_spend": round(row["sum"], 2),
                        "avg_spend": round(row["mean"], 2)
                    }
                    for merchant, row in top_merchants.iterrows()
                ],
                "unique_merchants": group["merchant"].nunique(),
            }

        # Overall known merchants
        all_merchants = set(df["merchant"].unique())

        return {
            "by_category": merchant_profile,
            "all_known_merchants": list(all_merchants),
            "total_unique_merchants": len(all_merchants),
        }

    def _build_velocity_profile(self, df: pd.DataFrame) -> dict:
        """
        Spending velocity — how fast the user spends over time.

        Interview Deep-Dive:
            Velocity captures the RATE of spending, not just the amount.
            A user who normally spends ₹2000/day suddenly spending ₹5000/day
            for several days indicates a behavioral shift, even if individual
            transactions look normal.
        """
        daily_spend = df.groupby("date")["amount"].sum()

        return {
            "avg_daily_spend": round(daily_spend.mean(), 2),
            "max_daily_spend": round(daily_spend.max(), 2),
            "min_daily_spend": round(daily_spend.min(), 2),
            "std_daily_spend": round(daily_spend.std(), 2),
            "avg_weekly_spend": round(daily_spend.mean() * 7, 2),
            "avg_monthly_spend": round(daily_spend.mean() * 30, 2),
        }

    def _build_overall_stats(self, df: pd.DataFrame) -> dict:
        """Overall spending statistics."""
        return {
            "total_spend": round(df["amount"].sum(), 2),
            "avg_transaction": round(df["amount"].mean(), 2),
            "median_transaction": round(df["amount"].median(), 2),
            "total_transactions": len(df),
            "categories_used": df["category"].nunique(),
            "date_range_days": self.total_days,
        }

    def is_merchant_known(self, merchant: str, category: str) -> bool:
        """Check if a merchant has been seen before in a category."""
        if category not in self.merchant_profile["by_category"]:
            return False
        known = [
            m["name"] for m in self.merchant_profile["by_category"][category]["top_merchants"]
        ]
        return merchant in known

    def get_category_baseline(self, category: str) -> dict:
        """Get the spending baseline for a specific category."""
        if category in self.category_profiles:
            return self.category_profiles[category]
        return None

    def get_profile_summary(self) -> dict:
        """Get a high-level profile summary for display."""
        return {
            "overall": self.overall_stats,
            "top_category": max(
                self.category_profiles.items(),
                key=lambda x: x[1]["total_spend"]
            )[0] if self.category_profiles else None,
            "spending_velocity": self.velocity_profile,
            "temporal_highlights": {
                "weekend_multiplier": self.temporal_profile["weekend_multiplier"],
                "highest_spend_month": self.temporal_profile["highest_spend_month"],
            }
        }

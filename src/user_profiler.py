"""
user_profiler.py — Behavioral spending profile
==============================================
Builds the user's spending baseline from validated, noise-filtered transactions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.entity_resolution import normalize_merchant


class UserProfile:
    """Comprehensive spending behavior profile for a single user."""

    def __init__(self, df: pd.DataFrame, currency_symbol: str = "₹"):
        self.currency = currency_symbol
        self.raw_df = df.copy() if df is not None else pd.DataFrame()
        self.df = self._filter_behavioral_spend(self.raw_df)
        self.total_transactions = len(self.df)

        if len(self.df) == 0:
            self.date_range = (pd.NaT, pd.NaT)
            self.total_days = 0
            self.category_profiles = {}
            self.temporal_profile = {
                "day_of_week_avg_spend": {},
                "day_of_week_txn_count": {},
                "weekday_avg": np.nan,
                "weekend_avg": np.nan,
                "weekend_multiplier": np.nan,
                "monthly_spend": {},
                "highest_spend_month": None,
                "lowest_spend_month": None,
            }
            self.merchant_profile = {
                "by_category": {},
                "all_known_merchants": [],
                "known_merchants_by_category": {},
                "total_unique_merchants": 0,
            }
            self.velocity_profile = {
                "avg_daily_spend": np.nan,
                "max_daily_spend": np.nan,
                "min_daily_spend": np.nan,
                "std_daily_spend": np.nan,
                "avg_weekly_spend": np.nan,
                "avg_monthly_spend": np.nan,
            }
            self.overall_stats = {
                "total_spend": 0.0,
                "avg_transaction": 0.0,
                "median_transaction": 0.0,
                "total_transactions": 0,
                "categories_used": 0,
                "date_range_days": 0,
            }
            return

        self.date_range = (self.df["date"].min(), self.df["date"].max())
        self.total_days = max((self.date_range[1] - self.date_range[0]).days + 1, 1)
        if "day_of_week" not in self.df.columns:
            self.df["day_of_week"] = self.df["date"].dt.dayofweek
        if "is_weekend" not in self.df.columns:
            self.df["is_weekend"] = self.df["day_of_week"].isin([5, 6]).astype(int)
        if "month" not in self.df.columns:
            self.df["month"] = self.df["date"].dt.month

        self.category_profiles = self._build_category_profiles(self.df)
        self.temporal_profile = self._build_temporal_profile(self.df)
        self.merchant_profile = self._build_merchant_profile(self.df)
        self.velocity_profile = self._build_velocity_profile(self.df)
        self.overall_stats = self._build_overall_stats(self.df)

    def _filter_behavioral_spend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove transfer noise from baseline profiling."""
        if df is None or len(df) == 0:
            return pd.DataFrame(columns=df.columns if df is not None else [])

        filtered = df.copy()
        transfer_mask = (
            filtered["is_transfer"].fillna(False).astype(bool)
            if "is_transfer" in filtered.columns
            else pd.Series(False, index=filtered.index)
        )
        filtered = filtered[~transfer_mask]
        if "category" in filtered.columns:
            filtered = filtered[~filtered["category"].astype(str).str.lower().isin(["income_transfer", "personal_transfer"])]
        return filtered.copy()

    def _build_category_profiles(self, df: pd.DataFrame) -> dict:
        """Per-category spending statistics."""
        profiles = {}
        for category, group in df.groupby("category"):
            std = group["amount"].std() if len(group) > 1 else 0.0
            mean = group["amount"].mean()
            profiles[category] = {
                "mean": round(mean, 2),
                "median": round(group["amount"].median(), 2),
                "std": round(std if not pd.isna(std) else 0.0, 2),
                "min": round(group["amount"].min(), 2),
                "max": round(group["amount"].max(), 2),
                "p25": round(group["amount"].quantile(0.25), 2),
                "p75": round(group["amount"].quantile(0.75), 2),
                "total_spend": round(group["amount"].sum(), 2),
                "transaction_count": int(len(group)),
                "avg_txns_per_week": round(len(group) / (self.total_days / 7), 2),
                "normal_range": (
                    round(max(0, mean - 1.5 * (std if not pd.isna(std) else 0.0)), 2),
                    round(mean + 1.5 * (std if not pd.isna(std) else 0.0), 2),
                ),
                "share_of_total": round(group["amount"].sum() / max(df["amount"].sum(), 1) * 100, 1),
            }
        return profiles

    def _build_temporal_profile(self, df: pd.DataFrame) -> dict:
        """Day-of-week and month behavior summary."""
        dow_spend = df.groupby("day_of_week")["amount"].agg(["mean", "sum", "count"])
        dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        weekday_df = df[df["is_weekend"] == 0]
        weekend_df = df[df["is_weekend"] == 1]
        weekday_avg = weekday_df["amount"].mean() if len(weekday_df) > 0 else 0.0
        weekend_avg = weekend_df["amount"].mean() if len(weekend_df) > 0 else 0.0
        monthly_spend = df.groupby("month")["amount"].sum()

        return {
            "day_of_week_avg_spend": {
                dow_names[i]: round(dow_spend.loc[i, "mean"], 2) if i in dow_spend.index else 0.0
                for i in range(7)
            },
            "day_of_week_txn_count": {
                dow_names[i]: int(dow_spend.loc[i, "count"]) if i in dow_spend.index else 0
                for i in range(7)
            },
            "weekday_avg": round(float(weekday_avg), 2),
            "weekend_avg": round(float(weekend_avg), 2),
            "weekend_multiplier": round(float(weekend_avg / weekday_avg), 2) if weekday_avg > 0 else 0.0,
            "monthly_spend": {int(month): round(value, 2) for month, value in monthly_spend.items()},
            "highest_spend_month": int(monthly_spend.idxmax()) if len(monthly_spend) > 0 else None,
            "lowest_spend_month": int(monthly_spend.idxmin()) if len(monthly_spend) > 0 else None,
        }

    def _build_merchant_profile(self, df: pd.DataFrame) -> dict:
        """Merchant familiarity profile used by the explainer and insights engine."""
        df = df.copy()
        if "merchant_normalized" not in df.columns:
            df["merchant_normalized"] = df["merchant"].apply(normalize_merchant)

        merchant_profile = {}
        known_by_category = {}

        for category, group in df.groupby("category"):
            top_merchants = (
                group.groupby(["merchant_normalized", "merchant"])["amount"]
                .agg(["count", "sum", "mean"])
                .sort_values("count", ascending=False)
                .head(5)
            )
            merchant_profile[category] = {
                "top_merchants": [
                    {
                        "name": merchant,
                        "merchant_normalized": merchant_normalized,
                        "count": int(row["count"]),
                        "total_spend": round(row["sum"], 2),
                        "avg_spend": round(row["mean"], 2),
                    }
                    for (merchant_normalized, merchant), row in top_merchants.iterrows()
                ],
                "unique_merchants": int(group["merchant_normalized"].nunique()),
            }
            known_by_category[category] = set(group["merchant_normalized"].unique())

        all_known_merchants = sorted(set(df["merchant_normalized"].unique()))
        return {
            "by_category": merchant_profile,
            "all_known_merchants": all_known_merchants,
            "known_merchants_by_category": known_by_category,
            "total_unique_merchants": len(all_known_merchants),
        }

    def _build_velocity_profile(self, df: pd.DataFrame) -> dict:
        """Daily, weekly, and monthly spending velocity."""
        daily_spend = df.groupby("date")["amount"].sum()
        return {
            "avg_daily_spend": round(daily_spend.mean(), 2),
            "max_daily_spend": round(daily_spend.max(), 2),
            "min_daily_spend": round(daily_spend.min(), 2),
            "std_daily_spend": round(daily_spend.std(), 2) if len(daily_spend) > 1 else 0.0,
            "avg_weekly_spend": round(daily_spend.mean() * 7, 2),
            "avg_monthly_spend": round(daily_spend.mean() * 30, 2),
        }

    def _build_overall_stats(self, df: pd.DataFrame) -> dict:
        """Overall profile statistics."""
        return {
            "total_spend": round(df["amount"].sum(), 2),
            "avg_transaction": round(df["amount"].mean(), 2),
            "median_transaction": round(df["amount"].median(), 2),
            "total_transactions": int(len(df)),
            "categories_used": int(df["category"].nunique()),
            "date_range_days": int(self.total_days),
        }

    def is_merchant_known(self, merchant: str, category: str) -> bool:
        """Check if a merchant has been seen before in a category."""
        merchant_key = normalize_merchant(merchant)
        known = self.merchant_profile.get("known_merchants_by_category", {}).get(category, set())
        return merchant_key in known

    def get_category_baseline(self, category: str) -> dict | None:
        """Get baseline stats for a specific category."""
        return self.category_profiles.get(category)

    def get_profile_summary(self) -> dict:
        """Get a concise profile summary for the UI."""
        return {
            "overall": self.overall_stats,
            "top_category": max(self.category_profiles.items(), key=lambda item: item[1]["total_spend"])[0]
            if self.category_profiles
            else None,
            "spending_velocity": self.velocity_profile,
            "temporal_highlights": {
                "weekend_multiplier": self.temporal_profile.get("weekend_multiplier"),
                "highest_spend_month": self.temporal_profile.get("highest_spend_month"),
            },
        }

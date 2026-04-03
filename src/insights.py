"""
insights.py — Actionable Financial Recommendations
=====================================================
PURPOSE (Interview Talking Point):
    Transforms raw data analysis into ACTIONABLE ADVICE. This is what
    makes PFAD a "financial assistant" rather than just a "data viewer."

    The system doesn't just show charts — it tells the user:
    - What to fix
    - How to fix it
    - What happens if they don't

WHY ACTIONABLE INSIGHTS MATTER:
    Most finance apps show data. Very few tell you what to DO about it.
    This module bridges that gap using rule-based logic on top of the
    user profile and anomaly detection results.

    Each insight has:
    - Type: (warning, positive, suggestion, prediction)
    - Priority: (high, medium, low)
    - Message: natural language recommendation
    - Data: supporting numbers
"""

import pandas as pd
import numpy as np
from src.user_profiler import UserProfile


class InsightGenerator:
    """
    Generates actionable financial insights from transaction data and user profile.

    Interview Talking Points:
        - Rule-based system (not ML) — clear, debuggable, interpretable
        - Categorized insights: warnings, positive reinforcement, suggestions, predictions
        - Priority-ranked so the most important insights appear first
        - Each insight includes supporting data for credibility
    """

    def __init__(
        self,
        df: pd.DataFrame,
        profile: UserProfile,
        currency_symbol: str = "₹"
    ):
        self.df = df
        self.profile = profile
        self.currency = currency_symbol
        self.insights = []

        # Generate all insights
        self._check_category_overspending()
        self._check_weekend_spending()
        self._check_spending_trend()
        self._check_category_consistency()
        self._check_budget_forecast()
        self._check_merchant_concentration()
        self._check_positive_habits()

        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        self.insights.sort(key=lambda x: priority_order.get(x["priority"], 3))

    def _check_category_overspending(self):
        """
        Check if any category is significantly over its baseline this week/month.

        Rule:
            If current week's spend in a category > baseline_avg + 1.5 × std → warning
        """
        if len(self.df) == 0:
            return

        recent_cutoff = self.df["date"].max() - pd.Timedelta(days=7)
        recent = self.df[self.df["date"] >= recent_cutoff]

        for category, prof in self.profile.category_profiles.items():
            cat_recent = recent[recent["category"] == category]
            if len(cat_recent) == 0:
                continue

            weekly_spend = cat_recent["amount"].sum()
            baseline_weekly = prof["mean"] * prof["avg_txns_per_week"]

            if baseline_weekly > 0 and weekly_spend > baseline_weekly * 1.5:
                overspend_pct = ((weekly_spend / baseline_weekly) - 1) * 100

                self.insights.append({
                    "type": "warning",
                    "priority": "high",
                    "category": category,
                    "title": f"Overspending on {category.title()}",
                    "message": (
                        f"You spent {self.currency}{weekly_spend:,.0f} on {category} "
                        f"this week — that's **{overspend_pct:.0f}% above** your weekly "
                        f"baseline of {self.currency}{baseline_weekly:,.0f}."
                    ),
                    "suggestion": (
                        f"Consider reducing {category} spending by "
                        f"{self.currency}{weekly_spend - baseline_weekly:,.0f} next week "
                        f"to stay on track."
                    ),
                    "data": {
                        "current_weekly": weekly_spend,
                        "baseline_weekly": baseline_weekly,
                        "overspend_pct": overspend_pct,
                    }
                })

    def _check_weekend_spending(self):
        """
        Check if weekend spending is significantly higher than weekday.
        """
        multiplier = self.profile.temporal_profile.get("weekend_multiplier", 1.0)

        if multiplier > 1.8:
            weekday_avg = self.profile.temporal_profile["weekday_avg"]
            weekend_avg = self.profile.temporal_profile["weekend_avg"]

            self.insights.append({
                "type": "suggestion",
                "priority": "medium",
                "category": "overall",
                "title": "Weekend Spending Spike",
                "message": (
                    f"Your weekend spending ({self.currency}{weekend_avg:,.0f}/txn) is "
                    f"**{multiplier:.1f}× higher** than weekdays "
                    f"({self.currency}{weekday_avg:,.0f}/txn)."
                ),
                "suggestion": (
                    "Consider setting a separate weekend budget to control "
                    "impulse spending."
                ),
                "data": {
                    "weekend_avg": weekend_avg,
                    "weekday_avg": weekday_avg,
                    "multiplier": multiplier,
                }
            })

    def _check_spending_trend(self):
        """
        Is overall spending trending up month-over-month?
        """
        monthly = self.profile.temporal_profile.get("monthly_spend", {})
        if len(monthly) < 2:
            return

        months_sorted = sorted(monthly.keys())
        recent_months = months_sorted[-3:]  # Last 3 months

        if len(recent_months) >= 2:
            values = [monthly[m] for m in recent_months]
            # Check if consistently increasing
            if all(values[i] < values[i + 1] for i in range(len(values) - 1)):
                increase_pct = ((values[-1] / values[0]) - 1) * 100

                self.insights.append({
                    "type": "warning",
                    "priority": "high",
                    "category": "overall",
                    "title": "Rising Spending Trend",
                    "message": (
                        f"Your monthly spending has increased by **{increase_pct:.0f}%** "
                        f"over the last {len(recent_months)} months. "
                        f"If this continues, you may exceed your typical monthly budget."
                    ),
                    "suggestion": (
                        "Review your recent purchases and identify non-essential categories "
                        "where you can cut back."
                    ),
                    "data": {
                        "recent_monthly_spend": dict(zip(recent_months, values)),
                        "increase_pct": increase_pct,
                    }
                })

    def _check_category_consistency(self):
        """
        Identify categories with high spending variance (inconsistent).
        """
        for category, prof in self.profile.category_profiles.items():
            if prof["std"] == 0 or prof["mean"] == 0:
                continue

            cv = prof["std"] / prof["mean"]  # Coefficient of variation

            if cv > 1.0 and prof["transaction_count"] > 5:
                self.insights.append({
                    "type": "suggestion",
                    "priority": "low",
                    "category": category,
                    "title": f"Inconsistent {category.title()} Spending",
                    "message": (
                        f"Your {category} spending varies a lot — from "
                        f"{self.currency}{prof['min']:,.0f} to "
                        f"{self.currency}{prof['max']:,.0f} "
                        f"(avg: {self.currency}{prof['mean']:,.0f})."
                    ),
                    "suggestion": (
                        f"Setting a per-transaction budget of "
                        f"{self.currency}{prof['p75']:,.0f} for {category} "
                        f"could help stabilize your spending."
                    ),
                    "data": {
                        "cv": round(cv, 2),
                        "mean": prof["mean"],
                        "std": prof["std"],
                    }
                })

    def _check_budget_forecast(self):
        """
        Predict if the user will overspend this month based on current trajectory.

        Interview Deep-Dive:
            Simple linear extrapolation: if we're halfway through the month
            and you've already spent 60% of your monthly average, you're
            on track to overshoot by 20%.
        """
        if len(self.df) == 0:
            return

        # Current month data
        latest_date = self.df["date"].max()
        month_start = latest_date.replace(day=1)
        month_data = self.df[self.df["date"] >= month_start]

        if len(month_data) == 0:
            return

        day_of_month = latest_date.day
        days_in_month = pd.Timestamp(latest_date).days_in_month

        current_month_spend = month_data["amount"].sum()
        projected_month_spend = current_month_spend * (days_in_month / day_of_month)

        avg_monthly = self.profile.velocity_profile.get("avg_monthly_spend", 0)

        if avg_monthly > 0 and projected_month_spend > avg_monthly * 1.2:
            overshoot_pct = ((projected_month_spend / avg_monthly) - 1) * 100
            overshoot_amount = projected_month_spend - avg_monthly

            self.insights.append({
                "type": "prediction",
                "priority": "high",
                "category": "overall",
                "title": "Budget Overshoot Forecast",
                "message": (
                    f"At your current pace, you'll spend about "
                    f"{self.currency}{projected_month_spend:,.0f} this month — "
                    f"**{overshoot_pct:.0f}% more** than your monthly average of "
                    f"{self.currency}{avg_monthly:,.0f}."
                ),
                "suggestion": (
                    f"To stay within budget, limit remaining daily spending to "
                    f"{self.currency}{max(0, (avg_monthly - current_month_spend) / max(1, days_in_month - day_of_month)):,.0f}."
                ),
                "data": {
                    "current_month_spend": current_month_spend,
                    "projected_month_spend": projected_month_spend,
                    "avg_monthly": avg_monthly,
                    "days_remaining": days_in_month - day_of_month,
                }
            })

    def _check_merchant_concentration(self):
        """
        Check if spending is too concentrated in few merchants.
        """
        merchant_totals = self.df.groupby("merchant")["amount"].sum().sort_values(ascending=False)
        total_spend = merchant_totals.sum()

        if total_spend == 0:
            return

        # Top merchant's share
        top_merchant = merchant_totals.index[0]
        top_share = merchant_totals.iloc[0] / total_spend

        if top_share > 0.25:
            self.insights.append({
                "type": "suggestion",
                "priority": "low",
                "category": "overall",
                "title": "Merchant Concentration",
                "message": (
                    f"**{top_share * 100:.0f}%** of your total spending goes to "
                    f"**{top_merchant}**. Diversifying vendors could help you find "
                    f"better deals."
                ),
                "suggestion": "Compare prices across different merchants for your regular purchases.",
                "data": {
                    "top_merchant": top_merchant,
                    "share_pct": round(top_share * 100, 1),
                }
            })

    def _check_positive_habits(self):
        """
        Highlight positive spending behaviors — positive reinforcement matters!
        """
        # Find most consistent category
        min_cv = float("inf")
        most_consistent_cat = None

        for category, prof in self.profile.category_profiles.items():
            if prof["mean"] > 0 and prof["transaction_count"] > 5:
                cv = prof["std"] / prof["mean"]
                if cv < min_cv:
                    min_cv = cv
                    most_consistent_cat = category

        if most_consistent_cat and min_cv < 0.5:
            self.insights.append({
                "type": "positive",
                "priority": "low",
                "category": most_consistent_cat,
                "title": f"Great Discipline in {most_consistent_cat.title()}!",
                "message": (
                    f"Your {most_consistent_cat} spending is very consistent "
                    f"(avg: {self.currency}"
                    f"{self.profile.category_profiles[most_consistent_cat]['mean']:,.0f}). "
                    f"This shows great financial discipline! 🎉"
                ),
                "suggestion": "Keep it up! Consistency is key to financial health.",
                "data": {"cv": round(min_cv, 2)}
            })

    def get_insights(self) -> list[dict]:
        """Return all generated insights, sorted by priority."""
        return self.insights

    def get_insights_by_type(self, insight_type: str) -> list[dict]:
        """Get insights filtered by type (warning, suggestion, positive, prediction)."""
        return [i for i in self.insights if i["type"] == insight_type]

    def get_top_insights(self, n: int = 5) -> list[dict]:
        """Get the top N most important insights."""
        return self.insights[:n]

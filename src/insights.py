"""
insights.py — Behavior-aware insight and action engine
=====================================================
Turns validated transaction data into ranked, action-oriented behavioral
insights with explicit problem/cause/impact/action framing.
"""

from __future__ import annotations

import pandas as pd

from src.database import get_budgets
from src.recurring import detect_recurring
from src.user_profiler import UserProfile


class InsightGenerator:
    """Generate ranked top insights from behavioral transaction patterns."""

    def __init__(self, df: pd.DataFrame, profile: UserProfile, currency_symbol: str = "₹"):
        self.raw_df = df.copy() if df is not None else pd.DataFrame()
        self.df = self._filter_noise(self.raw_df)
        self.profile = profile
        self.currency = currency_symbol
        self.insights = []

        self._detect_behavioral_drift()
        self._detect_recurring_leaks()
        self._detect_budget_risk()
        self._detect_category_dominance()
        self._detect_volatility()
        self._rank_insights()

    def _filter_noise(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def _register_insight(
        self,
        *,
        insight_type: str,
        display_type: str,
        priority: str,
        title: str,
        category: str,
        what_changed: str,
        problem: str,
        cause: str,
        impact: str,
        action: str,
        expected_gain: str,
        impact_value: float,
        deviation_value: float,
        frequency_value: float,
        action_plan: dict | None = None,
    ):
        score = max(impact_value, 1.0) * max(deviation_value, 1.0) * max(frequency_value, 1.0)
        
        # Default action plan if none provided
        plan = action_plan or {
            "target": action,
            "timeframe": "Next 7 days",
            "strategy": cause,
            "impact": expected_gain,
        }

        self.insights.append({
            "type": display_type,
            "insight_type": insight_type,
            "priority": priority,
            "category": category,
            "title": title,
            "score": round(float(score), 2),
            "what_changed": what_changed,
            "problem": problem,
            "cause": cause,
            "impact": impact,
            "action": action,
            "expected_gain": expected_gain,
            "message": (
                f"What changed: {what_changed}\n"
                f"Why: {cause}\n"
                f"Impact: {impact}\n"
                f"Action: {action}"
            ),
            "action_plan": plan,
        })

    def _detect_behavioral_drift(self):
        """Detect category-level spend drift against trailing baseline."""
        if len(self.df) == 0:
            return

        latest_date = self.df["date"].max()
        window_days = 30 if self.profile.total_days >= 45 else 7
        recent_cutoff = latest_date - pd.Timedelta(days=window_days - 1)
        recent = self.df[self.df["date"] >= recent_cutoff]

        for category, category_df in self.df.groupby("category"):
            recent_category = recent[recent["category"] == category]
            if len(recent_category) == 0:
                continue

            current_spend = recent_category["amount"].sum()
            baseline_spend = (category_df["amount"].sum() / max(self.profile.total_days, 1)) * window_days
            if baseline_spend <= 0:
                continue

            deviation = current_spend / baseline_spend
            delta = current_spend - baseline_spend
            if deviation <= 1.3 or delta <= 0:
                continue

            priority = "critical" if deviation >= 1.8 else "high"
            self._register_insight(
                insight_type="drift",
                display_type="warning",
                priority=priority,
                title=f"{category.title()} spend drift",
                category=category,
                what_changed=f"{category.title()} spend is {deviation:.1f}x above its trailing {window_days}-day baseline.",
                problem=f"{category.title()} spending accelerated faster than your normal pattern.",
                cause=f"{len(recent_category)} recent transactions pushed the category from {self.currency}{baseline_spend:,.0f} to {self.currency}{current_spend:,.0f}.",
                impact=f"At the current pace, you are leaking roughly {self.currency}{delta:,.0f} above baseline in this category.",
                action=f"Set a short-term cap on {category.title()} and review the last {min(len(recent_category), 5)} transactions.",
                expected_gain=f"Recovering this drift would protect about {self.currency}{delta:,.0f} of discretionary cashflow.",
                impact_value=float(delta),
                deviation_value=float(deviation),
                frequency_value=float(len(recent_category)),
                action_plan={
                    "target": f"Stabilize {category.title()} velocity",
                    "timeframe": "Over the next 48 hours",
                    "strategy": f"Implement a total freeze on discretionary {category} purchases.",
                    "impact": f"Reclaim ~{self.currency}{delta:,.0f} and restore trailing averages"
                }
            )

    def _detect_recurring_leaks(self):
        """Detect recurring subscription-style outflows."""
        recurring_data = detect_recurring(self.raw_df)
        if recurring_data["count"] == 0:
            return

        top_merchants = ", ".join(item["merchant"] for item in recurring_data["details"][:3])
        priority = "critical" if recurring_data["total_monthly"] >= 3000 else "medium"
        self._register_insight(
            insight_type="recurring",
            display_type="info",
            priority=priority,
            title="Recurring charge load",
            category="overall",
            what_changed=f"{recurring_data['count']} recurring merchants were detected from your payment history.",
            problem="Fixed charges are silently reducing monthly flexibility.",
            cause=f"Repeated billing patterns were found for {top_merchants or 'multiple merchants'}.",
            impact=f"These commitments are locking in about {self.currency}{recurring_data['total_monthly']:,.0f} per month.",
            action="Audit active subscriptions and cancel at least one low-value recurring charge.",
            expected_gain=f"Even one cancellation could immediately free part of the {self.currency}{recurring_data['total_monthly']:,.0f} monthly fixed outflow.",
            impact_value=float(recurring_data["total_monthly"]),
            deviation_value=float(max(recurring_data["count"], 1)),
            frequency_value=float(max(recurring_data["count"], 1)),
            action_plan={
                "target": "Reduce recurring overhead",
                "timeframe": "This weekend",
                "strategy": "Audit active mandates and cancel at least one unused service.",
                "impact": f"Instant {self.currency}{recurring_data['total_monthly']//recurring_data['count']:,.0f}+ monthly cashflow boost"
            }
        )

    def _detect_budget_risk(self):
        """Project end-of-month risk against budgets or baseline monthly spend."""
        if len(self.df) == 0:
            return

        latest_date = self.df["date"].max()
        month_start = latest_date.replace(day=1)
        month_data = self.df[self.df["date"] >= month_start]
        if len(month_data) == 0:
            return

        day_of_month = latest_date.day
        days_in_month = pd.Timestamp(latest_date).days_in_month
        current_month_spend = month_data["amount"].sum()
        projected_spend = current_month_spend * (days_in_month / max(day_of_month, 1))

        budgets = get_budgets()
        explicit_budget = sum(budgets.values()) if budgets else 0.0
        baseline_monthly = self.profile.velocity_profile.get("avg_monthly_spend", 0.0) or 0.0
        comparison_target = explicit_budget if explicit_budget > 0 else baseline_monthly
        if comparison_target <= 0:
            return

        deviation = projected_spend / comparison_target
        overshoot = projected_spend - comparison_target
        if deviation <= 1.1 or overshoot <= 0:
            return

        priority = "critical" if deviation >= 1.2 else "high"
        target_label = "budget" if explicit_budget > 0 else "baseline"
        self._register_insight(
            insight_type="budget_risk",
            display_type="prediction",
            priority=priority,
            title="Budget run-rate risk",
            category="overall",
            what_changed=f"Projected month-end spend is tracking toward {self.currency}{projected_spend:,.0f}.",
            problem=f"Your current run-rate is above your normal monthly {target_label}.",
            cause=f"You have already spent {self.currency}{current_month_spend:,.0f} by day {day_of_month}, which projects to {deviation:.1f}x of the target.",
            impact=f"If the pace holds, month-end spend could overshoot by about {self.currency}{overshoot:,.0f}.",
            action="Reduce discretionary spending for the rest of the cycle.",
            expected_gain=f"Pulling the run-rate back to target would protect roughly {self.currency}{overshoot:,.0f} this month.",
            impact_value=float(overshoot),
            deviation_value=float(deviation),
            frequency_value=float(day_of_month),
            action_plan={
                "target": "Emergency Burn-rate Reduction",
                "timeframe": "Remainder of the billing cycle",
                "strategy": "Switch to 'Hard Necessity' mode for all discretionary categories.",
                "impact": f"Prevent a {self.currency}{overshoot:,.0f} net-worth erosion"
            }
        )

    def _detect_category_dominance(self):
        """Detect when one category dominates too much of spending."""
        if len(self.df) == 0:
            return

        category_totals = self.df.groupby("category")["amount"].sum().sort_values(ascending=False)
        total_spend = category_totals.sum()
        if total_spend <= 0 or len(category_totals) < 2:
            return

        top_category = category_totals.index[0]
        top_share = category_totals.iloc[0] / total_spend
        if top_share <= 0.35:
            return

        dominant_amount = category_totals.iloc[0]
        self._register_insight(
            insight_type="category_dominance",
            display_type="info",
            priority="medium",
            title=f"{top_category.title()} dominates spend",
            category=top_category,
            what_changed=f"{top_category.title()} now accounts for {top_share * 100:.0f}% of total tracked spending.",
            problem="A single category is carrying too much weight in your spending mix.",
            cause=f"Spending concentration built up to {self.currency}{dominant_amount:,.0f} in {top_category.title()}.",
            impact="High concentration makes savings more sensitive to price or habit changes in one area.",
            action=f"Review the largest {top_category.title()} transactions for optimization.",
            expected_gain=f"Even a 10% optimization in {top_category.title()} would return about {self.currency}{dominant_amount * 0.1:,.0f}.",
            impact_value=float(dominant_amount),
            deviation_value=float(top_share * 100),
            frequency_value=float(self.df[self.df["category"] == top_category]["amount"].count()),
            action_plan={
                "target": f"Diversify {top_category.title()} spending",
                "timeframe": "Next 30 days",
                "strategy": "Research cheaper substitutes or micro-optimize vendor choice in this sector.",
                "impact": "Improved structural resilience and personal savings rate"
            }
        )

    def _detect_volatility(self):
        """Detect categories with unstable transaction amounts."""
        for category, profile in self.profile.category_profiles.items():
            mean = profile.get("mean", 0.0)
            std = profile.get("std", 0.0)
            txn_count = profile.get("transaction_count", 0)
            if mean <= 0 or std <= 0 or txn_count < 6:
                continue

            cv = std / mean
            if cv <= 1.1:
                continue

            self._register_insight(
                insight_type="volatility",
                display_type="suggestion",
                priority="medium" if cv > 1.4 else "low",
                title=f"{category.title()} is volatile",
                category=category,
                what_changed=f"{category.title()} transaction amounts are swinging more than your normal categories.",
                problem=f"{category.title()} is hard to forecast because spend size is inconsistent.",
                cause=f"The category ranges from {self.currency}{profile['min']:,.0f} to {self.currency}{profile['max']:,.0f} with a CV of {cv:.2f}.",
                impact="This volatility makes month-end planning and budget controls less reliable.",
                action=f"Create a soft ceiling near {self.currency}{profile['p75']:,.0f} for this category.",
                expected_gain="Stabilizing this category should improve forecasting accuracy.",
                impact_value=float(std),
                deviation_value=float(cv),
                frequency_value=float(txn_count),
                action_plan={
                    "target": f"Normalize {category.title()} spend",
                    "timeframe": "Ongoing",
                    "strategy": f"Enforce a per-transaction ceiling of {self.currency}{profile['p75']:,.0f} for this category.",
                    "impact": "Predictable monthly cash flow models"
                }
            )

    def _rank_insights(self):
        """Sort by score and keep only the top 3 highest-impact insights."""
        priority_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        self.insights.sort(key=lambda item: (priority_rank.get(item["priority"], 99), -item["score"]))
        # Already ranked by score/priority, we'll slice in get_top_insights

    def get_insights(self) -> list[dict]:
        return self.insights

    def get_top_insights(self, n: int = 3) -> list[dict]:
        return self.insights[:n]

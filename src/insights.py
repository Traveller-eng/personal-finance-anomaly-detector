"""
insights.py — Actionable Financial Recommendations
=====================================================
PURPOSE:
    Transforms raw data analysis into ACTIONABLE ADVICE. This is what
    makes PFAD a "financial decision engine" rather than just an "analytics tool."

    The system doesn't just show charts — it tells the user:
    - What to fix
    - When to fix it (Time constraint)
    - How to fix it (Strategy hint)
    - What happens if they do (Impact projection)
"""

import pandas as pd
import numpy as np
from src.user_profiler import UserProfile
from src.recurring import detect_recurring
from src.database import get_budgets
import datetime
import calendar


class InsightGenerator:
    """
    Generates actionable financial insights from transaction data and user profile.
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
        self._check_budget_burn()
        self._check_recurring()
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

    def _check_budget_burn(self):
        """Generates dynamic burn rate alerts based on SQLite budgets."""
        budgets = get_budgets()
        if not budgets:
            return

        current_date_series = self.df['date'].max()
        if pd.isna(current_date_series): return
        current_date = pd.to_datetime(current_date_series)
        
        current_month_df = self.df[(self.df['date'].dt.year == current_date.year) & 
                                   (self.df['date'].dt.month == current_date.month)]
        
        _, days_in_month = calendar.monthrange(current_date.year, current_date.month)
        days_left = days_in_month - current_date.day

        for cat, limit in budgets.items():
            cat_df = current_month_df[current_month_df['category'].str.lower() == cat]
            spent = cat_df['amount'].sum()
            if limit > 0:
                burn_pct = (spent / limit) * 100
                if burn_pct >= 75:
                    is_critical = burn_pct > 100
                    priority = "high" if is_critical else "medium"
                    
                    self.insights.append({
                        "type": "warning",
                        "priority": priority,
                        "category": cat.title(),
                        "title": f"Budget Alert: {cat.title()}",
                        "message": (
                            f"**{cat.title()} spending reached {burn_pct:.0f}% of budget**\n"
                            f"Driven by:\n"
                            f"• {self.currency}{spent:,.0f} total spent vs {self.currency}{limit:,.0f} limit\n"
                            f"• Only {days_left} days remaining in cycle\n"
                        ),
                        "action_plan": {
                            "target": f"Hard lock {cat.title()} spending",
                            "timeframe": f"over the next {days_left} days",
                            "strategy": "by utilizing pre-purchased items or completely avoiding this category",
                            "impact": "Expected Impact: +5 to +10 score preservation"
                        }
                    })

    def _check_recurring(self):
        """Identifies recurring subscription overhead."""
        recurring_data = detect_recurring(self.df)
        if recurring_data["count"] > 0:
            self.insights.append({
                "type": "info",
                "priority": "high",
                "category": "overall",
                "title": "Recurring Overhead Detected",
                "message": (
                    f"**{recurring_data['count']} active recurring charges identified**\n"
                    f"Driven by:\n"
                    f"• {self.currency}{recurring_data['total_monthly']:,.0f} fixed monthly drain\n"
                    f"• Automatic silent deductions\n"
                ),
                "action_plan": {
                    "target": "Audit subscriptions and cancel 1 unused service",
                    "timeframe": "this weekend",
                    "strategy": "by reviewing payment auto-mandates in your banking app",
                    "impact": "Expected Impact: +2 to +4 score improvement and immediate cashflow boost"
                }
            })

    def _check_category_overspending(self):
        """Check if any category is significantly over its baseline this week/month."""
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
                deviation_amount = weekly_spend - baseline_weekly

                self.insights.append({
                    "type": "warning",
                    "priority": "high",
                    "category": category,
                    "title": f"Overspending on {category.title()}",
                    "message": (
                        f"**{category.title()} spending increased by {self.currency}{deviation_amount:,.0f} this week**\n"
                        f"Driven by:\n"
                        f"• +{overspend_pct:.0f}% variance against trailing baseline\n"
                        f"• {len(cat_recent)} active transactions\n"
                    ),
                    "action_plan": {
                        "target": f"Reduce {category} by {self.currency}{deviation_amount:,.0f}",
                        "timeframe": "over the next 7 days",
                        "strategy": "by halting impulse buys and adhering to baseline averages",
                        "impact": "Expected Impact: +3 to +6 score improvement"
                    }
                })

    def _check_weekend_spending(self):
        """Check if weekend spending is significantly higher than weekday."""
        multiplier = self.profile.temporal_profile.get("weekend_multiplier", 1.0)
        weekday_avg = self.profile.temporal_profile.get("weekday_avg", 0)
        weekend_avg = self.profile.temporal_profile.get("weekend_avg", 0)

        if multiplier > 1.8 and weekday_avg > 0:
            self.insights.append({
                "type": "suggestion",
                "priority": "medium",
                "category": "overall",
                "title": "Weekend Spending Spike",
                "message": (
                    f"**Weekend transaction velocity is {multiplier:.1f}× higher than normal**\n"
                    f"Driven by:\n"
                    f"• {self.currency}{weekend_avg:,.0f}/txn average on weekends\n"
                    f"• {self.currency}{weekday_avg:,.0f}/txn average on weekdays\n"
                ),
                "action_plan": {
                    "target": f"Cap weekend impulse threshold at {self.currency}{weekday_avg * 1.2:,.0f}/txn",
                    "timeframe": "starting this Friday",
                    "strategy": "by establishing a separate recreational soft-limit",
                    "impact": "Expected Impact: +2 to +5 score stabilization"
                }
            })

    def _check_spending_trend(self):
        """Is overall spending trending up month-over-month?"""
        monthly = self.profile.temporal_profile.get("monthly_spend", {})
        if len(monthly) < 2: return

        months_sorted = sorted(monthly.keys())
        recent_months = months_sorted[-3:]

        if len(recent_months) >= 2:
            values = [monthly[m] for m in recent_months]
            if all(values[i] < values[i + 1] for i in range(len(values) - 1)):
                increase_pct = ((values[-1] / values[0]) - 1) * 100
                delta = values[-1] - values[0]

                self.insights.append({
                    "type": "warning",
                    "priority": "high",
                    "category": "overall",
                    "title": "Rising Spending Trend",
                    "message": (
                        f"**Progressive spending creep of {self.currency}{delta:,.0f} detected**\n"
                        f"Driven by:\n"
                        f"• +{increase_pct:.0f}% volume over the last {len(recent_months)} consecutive periods\n"
                        f"• Consistent upward friction across general categories\n"
                    ),
                    "action_plan": {
                        "target": "Audit recent purchases and isolate lifestyle inflation",
                        "timeframe": "before the next billing cycle",
                        "strategy": "by freezing discretionary spending for 48 hours to reset habits",
                        "impact": "Expected Impact: Prevents massive -10 point long-term downgrade"
                    }
                })

    def _check_category_consistency(self):
        """Identify categories with high spending variance."""
        for category, prof in self.profile.category_profiles.items():
            if prof["std"] == 0 or prof["mean"] == 0: continue
            cv = prof["std"] / prof["mean"]

            if cv > 1.0 and prof["transaction_count"] > 5:
                self.insights.append({
                    "type": "suggestion",
                    "priority": "low",
                    "category": category,
                    "title": f"Inconsistent {category.title()} Spending",
                    "message": (
                        f"**Severe volatility detected in {category.title()}**\n"
                        f"Driven by:\n"
                        f"• Massive variance from {self.currency}{prof['min']:,.0f} to {self.currency}{prof['max']:,.0f}\n"
                        f"• Unpredictable velocity impacting overall budget safety\n"
                    ),
                    "action_plan": {
                        "target": f"Enforce a per-transaction ceiling of {self.currency}{prof['p75']:,.0f}",
                        "timeframe": "effective immediately",
                        "strategy": "by standardizing vendors or restricting premium tier purchases",
                        "impact": "Expected Impact: +1 to +3 score improvement via variance reduction"
                    }
                })

    def _check_budget_forecast(self):
        """Predict if the user will overspend this month based on current trajectory."""
        if len(self.df) == 0: return

        latest_date = self.df["date"].max()
        month_start = latest_date.replace(day=1)
        month_data = self.df[self.df["date"] >= month_start]

        if len(month_data) == 0: return

        day_of_month = latest_date.day
        days_in_month = pd.Timestamp(latest_date).days_in_month

        current_month_spend = month_data["amount"].sum()
        projected_month_spend = current_month_spend * (days_in_month / max(1, day_of_month))
        avg_monthly = self.profile.velocity_profile.get("avg_monthly_spend", 0)

        if avg_monthly > 0 and projected_month_spend > avg_monthly * 1.2:
            overshoot_pct = ((projected_month_spend / avg_monthly) - 1) * 100
            days_remaining = days_in_month - day_of_month
            daily_limit = max(0, (avg_monthly - current_month_spend) / max(1, days_remaining))

            self.insights.append({
                "type": "prediction",
                "priority": "high",
                "category": "overall",
                "title": "Budget Overshoot Forecast",
                "message": (
                    f"**Projected to exceed monthly baseline by {overshoot_pct:.0f}%**\n"
                    f"Driven by:\n"
                    f"• {self.currency}{current_month_spend:,.0f} already spent by day {day_of_month}\n"
                    f"• Dangerous trajectory toward {self.currency}{projected_month_spend:,.0f}\n"
                ),
                "action_plan": {
                    "target": f"Throttle daily spending to {self.currency}{daily_limit:,.0f}",
                    "timeframe": f"for the remaining {days_remaining} days",
                    "strategy": "by switching to an absolute-necessity operational mode",
                    "impact": "Expected Impact: Prevents a severe -15 point Health Score penalty"
                }
            })

    def _check_merchant_concentration(self):
        """Check if spending is too concentrated in few merchants."""
        merchant_totals = self.df.groupby("merchant")["amount"].sum().sort_values(ascending=False)
        total_spend = merchant_totals.sum()
        if total_spend == 0: return

        top_merchant = merchant_totals.index[0]
        top_share = merchant_totals.iloc[0] / total_spend

        if top_share > 0.25:
            self.insights.append({
                "type": "suggestion",
                "priority": "low",
                "category": "overall",
                "title": "Merchant Concentration",
                "message": (
                    f"**Massive capital lock-in with {top_merchant}**\n"
                    f"Driven by:\n"
                    f"• {top_share * 100:.0f}% of total lifetime volume routed to a single vendor\n"
                    f"• High dependency risk on vendor pricing shifts\n"
                ),
                "action_plan": {
                    "target": f"Divert 10% of {top_merchant} volume to competitors",
                    "timeframe": "over the next 30 days",
                    "strategy": "by price-comparing your core recurring basket against alternate providers",
                    "impact": "Expected Impact: Long-term savings through volume diversification"
                }
            })

    def _check_positive_habits(self):
        """Highlight positive spending behaviors."""
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
                "title": f"Great Discipline in {most_consistent_cat.title()}",
                "message": (
                    f"**Exceptional stability detected in {most_consistent_cat.title()}**\n"
                    f"Driven by:\n"
                    f"• {self.currency}{self.profile.category_profiles[most_consistent_cat]['mean']:,.0f} average execution\n"
                    f"• Extremely low volatility (CV < 0.5)\n"
                ),
                "action_plan": {
                    "target": "Maintain current operational parameters",
                    "timeframe": "ongoing",
                    "strategy": "by continuing your existing psychological anchoring for this category",
                    "impact": "Expected Impact: Continues to anchor your Health Score near 90+"
                }
            })

    def get_insights(self) -> list[dict]:
        return self.insights

    def get_insights_by_type(self, insight_type: str) -> list[dict]:
        return [i for i in self.insights if i["type"] == insight_type]

    def get_top_insights(self, n: int = 5) -> list[dict]:
        return self.insights[:n]

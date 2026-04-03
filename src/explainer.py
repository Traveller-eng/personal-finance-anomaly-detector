"""
explainer.py — Human-Readable Anomaly Explanations
=====================================================
PURPOSE:
    The MOST IMPORTANT module for user trust. An anomaly detector that says
    "this is anomalous" without explanation is useless — users need to know WHY.

    This module compares each anomaly against the user's behavioral profile
    and generates natural language explanations like:
    "You usually spend ₹300-₹500 on food. This ₹2200 transaction is 4.4× higher
    than your average. You also rarely spend this much on Mondays."

WHY EXPLAINABILITY MATTERS:
    In fintech/ML systems, explainability is CRITICAL because:
    1. Users won't trust a black-box system with their finances
    2. False positives (normal transactions flagged as anomalies) are common —
       explanations help users quickly dismiss false alarms
    3. Regulatory compliance (GDPR, RBI) may require explanations for
       automated financial decisions
    4. It transforms ML output into ACTIONABLE insights
"""

import pandas as pd
import numpy as np
from src.user_profiler import UserProfile


def explain_anomalies(
    df: pd.DataFrame,
    profile: UserProfile
) -> pd.DataFrame:
    """
    Generate explanations for all anomaly-flagged transactions.

    Parameters:
        df: DataFrame with anomaly detection results
        profile: User's spending profile

    Returns:
        DataFrame with added 'explanation' and 'explanation_factors' columns
    """
    df = df.copy()
    explanations = []
    structured_explanations = []
    factors_list = []

    for _, row in df.iterrows():
        if row.get("is_anomaly", 0) == 1:
            explanation, structured, factors = _explain_single_anomaly(row, profile)
            explanations.append(explanation)
            structured_explanations.append(structured)
            factors_list.append(factors)
        else:
            explanations.append("")
            structured_explanations.append({})
            factors_list.append([])

    df["explanation"] = explanations
    df["structured_explanation"] = structured_explanations
    df["explanation_factors"] = factors_list

    return df


def _explain_single_anomaly(row: pd.Series, profile: UserProfile) -> tuple[str, dict, list]:
    """
    Generate explanation for a single anomalous transaction.



        Each factor that triggers gets added to the explanation.
        The final explanation is a concatenation of all triggered factors.
    """
    currency = profile.currency
    category = row["category"]
    amount = row["amount"]
    factors = []
    explanation_parts = []
    
    structured = {
        "typical": "N/A",
        "deviation": "N/A",
        "reason": ""
    }

    baseline = profile.get_category_baseline(category)

    # ─── Factor 1: Amount vs. Category Baseline ────────────────────────
    if baseline:
        avg = baseline["mean"]
        normal_low, normal_high = baseline["normal_range"]
        
        structured["typical"] = f"{currency}{max(0, normal_low):,.0f}–{currency}{normal_high:,.0f}"

        if amount > normal_high:
            factors.append("amount_spike")
            multiplier = amount / avg if avg > 0 else 0
            deviation_pct = ((amount / avg) - 1) * 100 if avg > 0 else 0
            structured["deviation"] = f"+{deviation_pct:.0f}%"
            structured["reason"] += f"Significantly higher than average {category} spending. "
            
            explanation_parts.append(
                f"You usually spend {currency}{max(0, normal_low):,.0f}–{currency}{normal_high:,.0f} "
                f"on {category}. This {currency}{amount:,.0f} is **{multiplier:.1f}× higher** "
                f"than your average ({currency}{avg:,.0f})."
            )
        elif amount < normal_low and normal_low > 0:
            factors.append("amount_drop")
            deviation_pct = ((amount / avg) - 1) * 100
            structured["deviation"] = f"{deviation_pct:.0f}%"
            structured["reason"] += f"Unusually low {category} spending. "
            
            explanation_parts.append(
                f"This {currency}{amount:,.0f} on {category} is unusually low. "
                f"You typically spend {currency}{normal_low:,.0f}–{currency}{normal_high:,.0f}."
            )

    # ─── Factor 2: Amount vs. Rolling Average ──────────────────────────
    rolling_30d = row.get("rolling_30d_avg", 0)
    if rolling_30d > 0:
        daily_avg = rolling_30d
        if amount > daily_avg * 3:
            factors.append("above_trend")
            structured["reason"] += "Exceeds 30-day daily average by 3x. "
            explanation_parts.append(
                f"Your recent daily spending average is {currency}{daily_avg:,.0f}. "
                f"This single transaction exceeds 3× that average."
            )

    # ─── Factor 3: Merchant Novelty ────────────────────────────────────
    merchant = row.get("merchant", "")
    if merchant and baseline:
        if not profile.is_merchant_known(merchant, category):
            factors.append("new_merchant")
            structured["reason"] += f"New or rare vendor ({merchant}). "
            explanation_parts.append(
                f"**{merchant}** is not among your usual {category} merchants. "
                f"This is a new or rare vendor for you."
            )

    # ─── Factor 4: Temporal Pattern ────────────────────────────────────
    day_name = row.get("day_name", "")
    if day_name and day_name in profile.temporal_profile.get("day_of_week_txn_count", {}):
        day_count = profile.temporal_profile["day_of_week_txn_count"][day_name]
        total_count = sum(profile.temporal_profile["day_of_week_txn_count"].values())
        day_share = day_count / total_count if total_count > 0 else 0

        if day_share < 0.08:  # Less than 8% of transactions happen on this day
            factors.append("unusual_day")
            structured["reason"] += f"Unusual day for transactions ({day_name}). "
            explanation_parts.append(
                f"You rarely make transactions on {day_name}s "
                f"(only {day_share * 100:.1f}% of your transactions)."
            )

    # ─── Factor 5: Weekend/Weekday Deviation ───────────────────────────
    is_weekend = row.get("is_weekend", 0)
    if is_weekend:
        weekday_avg = profile.temporal_profile.get("weekday_avg", 0)
        if weekday_avg > 0 and amount > weekday_avg * 4:
            factors.append("weekend_spike")
            structured["reason"] += "Weekend spending spike. "
            explanation_parts.append(
                f"This weekend transaction is significantly higher than your "
                f"weekday average of {currency}{weekday_avg:,.0f}."
            )

    # ─── Combine explanations ──────────────────────────────────────────
    if not explanation_parts:
        explanation = (
            f"This {currency}{amount:,.0f} transaction in {category} "
            f"was flagged as unusual based on a combination of spending patterns."
        )
        factors.append("multi_factor")
        structured["reason"] = "Combination of multi-factor deviations."
    else:
        explanation = " ".join(explanation_parts)

    structured["reason"] = structured["reason"].strip()
    return explanation, structured, factors


def get_anomaly_explanations_summary(df: pd.DataFrame) -> dict:
    """
    Summarize explanation factors across all anomalies.
    Useful for understanding what TYPES of anomalies are most common.
    """
    anomalies = df[df["is_anomaly"] == 1]
    if len(anomalies) == 0:
        return {"total_anomalies": 0, "factors": {}, "most_common_factor": None}

    all_factors = []
    for factors in anomalies["explanation_factors"]:
        if isinstance(factors, list):
            all_factors.extend(factors)

    factor_counts = pd.Series(all_factors).value_counts().to_dict()

    return {
        "total_anomalies": len(anomalies),
        "factors": factor_counts,
        "most_common_factor": max(factor_counts, key=factor_counts.get) if factor_counts else None,
    }

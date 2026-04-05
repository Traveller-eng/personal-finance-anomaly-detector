from __future__ import annotations

import numpy as np
import pandas as pd


def detect_recurring(df: pd.DataFrame) -> dict:
    """
    Identify likely recurring transactions such as subscriptions and bills.

    The detector ignores transfer noise and prefers debit transactions because
    recurring credits rarely represent a money leak.
    """
    if df is None or df.empty or "date" not in df.columns or "merchant" not in df.columns:
        return {"count": 0, "total_monthly": 0.0, "details": []}

    working = df.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    working = working.dropna(subset=["date"])

    if "type" in working.columns:
        working = working[working["type"].fillna("debit").astype(str).str.lower() == "debit"]
    if "is_transfer" in working.columns:
        working = working[~working["is_transfer"].fillna(False).astype(bool)]

    merchant_key = "merchant_normalized" if "merchant_normalized" in working.columns else "merchant"
    display_key = "merchant"

    recurring_charges = []
    total_monthly = 0.0

    working = working.sort_values(by=[merchant_key, "date"])

    for merchant, group in working.groupby(merchant_key):
        if len(group) < 2:
            continue

        amounts = group["amount"].astype(float).values
        dates = group["date"].values
        mean_amount = amounts.mean()
        if mean_amount <= 0:
            continue

        cv_amount = amounts.std() / mean_amount
        if cv_amount > 0.2:
            continue

        date_diffs = np.diff(dates).astype("timedelta64[D]").astype(int)
        if len(date_diffs) == 0:
            continue

        mean_diff = date_diffs.mean()
        if not ((25 <= mean_diff <= 35) or (date_diffs.std() <= 3 and len(group) >= 3)):
            continue

        monthly_impact = (30 / mean_diff) * mean_amount if mean_diff > 0 else mean_amount
        total_monthly += monthly_impact

        recurring_charges.append({
            "merchant": str(group[display_key].iloc[0]),
            "merchant_normalized": str(merchant),
            "amount": float(round(mean_amount, 2)),
            "frequency_days": float(round(mean_diff, 1)),
            "monthly_impact": float(round(monthly_impact, 2)),
        })

    return {
        "count": len(recurring_charges),
        "total_monthly": float(round(total_monthly, 2)),
        "details": sorted(recurring_charges, key=lambda item: item["monthly_impact"], reverse=True),
    }

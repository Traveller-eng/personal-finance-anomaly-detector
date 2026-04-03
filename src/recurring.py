import pandas as pd
import numpy as np

def detect_recurring(df: pd.DataFrame) -> dict:
    """
    Identifies likely recurring transactions (subscriptions/bills).
    Finds merchants with >= 2 transactions, relatively consistent amounts,
    and a periodicity of roughly 30 days or consistent spacing.
    """
    if df.empty or 'date' not in df.columns or 'merchant' not in df.columns:
        return {"count": 0, "total_monthly": 0.0, "details": []}

    df = df.copy()
    recurring_charges = []
    total_monthly = 0.0
    
    # Ensure dates are datetime
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['merchant', 'date'])
    
    for merchant, group in df.groupby('merchant'):
        if len(group) < 2:
            continue
            
        amounts = group['amount'].values
        dates = group['date'].values
        
        # Check amount variation (coefficient of variation <= 15%)
        mean_amt = amounts.mean()
        if mean_amt <= 0:
            continue
            
        cv_amount = amounts.std() / mean_amt
        if cv_amount > 0.15:
            continue
            
        # Check date differences
        date_diffs = np.diff(dates).astype('timedelta64[D]').astype(int)
        if len(date_diffs) == 0:
            continue
            
        mean_diff = date_diffs.mean()
        
        # Pattern matching: either ~monthly (25-35 days) or highly regular (std <= 3)
        if (25 <= mean_diff <= 35) or (date_diffs.std() <= 3 and len(group) >= 3):
            recurring_charges.append({
                "merchant": merchant.title(),
                "amount": mean_amt,
                "frequency_days": mean_diff
            })
            
            # Estimate monthly impact: normalizes weekly/bi-weekly to monthly
            if mean_diff > 0:
                monthly_impact = (30 / mean_diff) * mean_amt
                total_monthly += monthly_impact
            
    return {
        "count": len(recurring_charges),
        "total_monthly": total_monthly,
        "details": sorted(recurring_charges, key=lambda x: x["amount"], reverse=True)
    }

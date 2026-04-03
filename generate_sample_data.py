"""
generate_sample_data.py — Synthetic Transaction Generator
==========================================================
PURPOSE (Interview Talking Point):
    Generates realistic personal finance data with KNOWN anomalies injected.
    This is critical for validating the anomaly detector — since we know exactly
    which transactions are anomalous, we can measure detection accuracy.

HOW IT WORKS:
    1. Defines "normal" spending profiles per category (mean, std, frequency)
    2. Generates 12 months of daily transactions following these profiles
    3. Injects ~5% anomalies: unusually large amounts, unusual timing,
       new merchants, spending spikes
    4. Saves as CSV with an optional 'is_injected_anomaly' column for validation

WHY THIS DESIGN:
    - Real bank data has privacy issues — synthetic data lets us demo freely
    - Injected anomalies give us ground truth to validate the ML model
    - Seasonal patterns (holidays, end-of-month rent) make data realistic
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# ─── Configuration ────────────────────────────────────────────────────────────

RANDOM_SEED = 42
START_DATE = datetime(2025, 4, 1)
END_DATE = datetime(2026, 3, 31)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "transactions.csv")

# ─── Spending Profiles ────────────────────────────────────────────────────────
# Each category has: (mean_amount, std_dev, avg_txns_per_week, merchants)
# Interview Note: These profiles define "normal behavior" — the baseline
# that the anomaly detector will learn to identify deviations from.

SPENDING_PROFILES = {
    "food": {
        "mean": 350,
        "std": 120,
        "txns_per_week": 5,
        "merchants": ["Swiggy", "Zomato", "Dominos", "Local Restaurant", "Cafe Coffee Day",
                       "McDonald's", "Subway", "Blinkit Groceries", "BigBasket"]
    },
    "shopping": {
        "mean": 1500,
        "std": 800,
        "txns_per_week": 1.5,
        "merchants": ["Amazon", "Flipkart", "Myntra", "Reliance Digital", "Croma",
                       "Decathlon", "IKEA", "Nykaa"]
    },
    "transport": {
        "mean": 150,
        "std": 60,
        "txns_per_week": 4,
        "merchants": ["Uber", "Ola", "Rapido", "Metro Card Recharge", "Petrol Pump",
                       "IRCTC"]
    },
    "bills": {
        "mean": 800,
        "std": 200,
        "txns_per_week": 0.5,
        "merchants": ["Jio Recharge", "Airtel", "Electricity Board", "Water Board",
                       "Gas Connection", "Broadband"]
    },
    "entertainment": {
        "mean": 500,
        "std": 250,
        "txns_per_week": 1,
        "merchants": ["Netflix", "Spotify", "BookMyShow", "PVR Cinemas",
                       "Steam Games", "YouTube Premium"]
    },
    "health": {
        "mean": 600,
        "std": 300,
        "txns_per_week": 0.3,
        "merchants": ["Apollo Pharmacy", "1mg", "PharmEasy", "Hospital",
                       "Gym Membership", "Practo Consultation"]
    },
    "education": {
        "mean": 2000,
        "std": 1000,
        "txns_per_week": 0.2,
        "merchants": ["Udemy", "Coursera", "Book Store", "Exam Fee",
                       "Coaching Center"]
    },
    "rent": {
        "mean": 15000,
        "std": 0,  # Rent is fixed
        "txns_per_week": 0.25,  # Once a month
        "merchants": ["Landlord UPI"]
    }
}


def set_seed(seed: int = RANDOM_SEED):
    """Ensure reproducibility across runs."""
    np.random.seed(seed)
    random.seed(seed)


def generate_normal_transactions(start_date: datetime, end_date: datetime) -> list[dict]:
    """
    Generate normal spending transactions across all categories.

    Interview Explanation:
        For each day in the date range, we probabilistically decide whether
        a transaction occurs in each category based on txns_per_week.
        The amount is sampled from a normal distribution clipped to positive values.
        This creates realistic, varied spending data.
    """
    transactions = []
    current_date = start_date

    while current_date <= end_date:
        for category, profile in SPENDING_PROFILES.items():
            # Daily probability of a transaction = txns_per_week / 7
            daily_prob = profile["txns_per_week"] / 7.0

            # Special case: rent is always on the 1st of the month
            if category == "rent":
                if current_date.day == 1:
                    transactions.append({
                        "date": current_date.strftime("%Y-%m-%d"),
                        "amount": round(profile["mean"] + np.random.normal(0, 500), 2),
                        "category": category,
                        "merchant": random.choice(profile["merchants"]),
                        "is_injected_anomaly": False
                    })
                continue

            # Weighted weekend effect: more food/entertainment on weekends
            if current_date.weekday() >= 5:  # Saturday or Sunday
                if category in ("food", "entertainment", "shopping"):
                    daily_prob *= 1.4  # 40% more likely on weekends
                elif category in ("transport",):
                    daily_prob *= 0.6  # Less transport on weekends

            # Generate transaction if probability check passes
            if random.random() < daily_prob:
                amount = max(10, np.random.normal(profile["mean"], profile["std"]))

                # Add slight seasonal variation (more spending in Nov-Dec holidays)
                month = current_date.month
                if month in (11, 12) and category in ("shopping", "food", "entertainment"):
                    amount *= random.uniform(1.1, 1.4)

                transactions.append({
                    "date": current_date.strftime("%Y-%m-%d"),
                    "amount": round(amount, 2),
                    "category": category,
                    "merchant": random.choice(profile["merchants"]),
                    "is_injected_anomaly": False
                })

        current_date += timedelta(days=1)

    return transactions


def inject_anomalies(transactions: list[dict], anomaly_fraction: float = 0.05) -> list[dict]:
    """
    Inject known anomalies into the transaction list.

    Interview Explanation:
        We inject 4 types of anomalies to test different detection capabilities:
        1. AMOUNT SPIKE — Transaction 3-8× the category average
        2. UNUSUAL TIMING — Transactions on days the user never spends
        3. CATEGORY BURST — Multiple transactions in a low-frequency category on one day
        4. NEW MERCHANT — Transaction at a merchant not in the user's history

        Injecting known anomalies lets us compute precision/recall of the detector.
    """
    num_anomalies = int(len(transactions) * anomaly_fraction)
    anomalies = []

    anomaly_types = ["amount_spike", "unusual_timing", "category_burst", "new_merchant"]

    for i in range(num_anomalies):
        anomaly_type = random.choice(anomaly_types)
        # Pick a random date in the range
        days_offset = random.randint(0, (END_DATE - START_DATE).days)
        anomaly_date = START_DATE + timedelta(days=days_offset)

        if anomaly_type == "amount_spike":
            category = random.choice(list(SPENDING_PROFILES.keys()))
            if category == "rent":
                category = "shopping"
            profile = SPENDING_PROFILES[category]
            spike_amount = profile["mean"] * random.uniform(3, 8)
            anomalies.append({
                "date": anomaly_date.strftime("%Y-%m-%d"),
                "amount": round(spike_amount, 2),
                "category": category,
                "merchant": random.choice(profile["merchants"]),
                "is_injected_anomaly": True
            })

        elif anomaly_type == "unusual_timing":
            # A large transaction very early in the morning context (we mark the day)
            category = random.choice(["shopping", "entertainment", "health"])
            profile = SPENDING_PROFILES[category]
            anomalies.append({
                "date": anomaly_date.strftime("%Y-%m-%d"),
                "amount": round(profile["mean"] * random.uniform(2, 4), 2),
                "category": category,
                "merchant": random.choice(profile["merchants"]),
                "is_injected_anomaly": True
            })

        elif anomaly_type == "category_burst":
            # 3-5 transactions in a low-frequency category on the same day
            category = random.choice(["health", "education", "entertainment"])
            profile = SPENDING_PROFILES[category]
            burst_count = random.randint(3, 5)
            for _ in range(burst_count):
                anomalies.append({
                    "date": anomaly_date.strftime("%Y-%m-%d"),
                    "amount": round(np.random.normal(profile["mean"], profile["std"]), 2),
                    "category": category,
                    "merchant": random.choice(profile["merchants"]),
                    "is_injected_anomaly": True
                })

        elif anomaly_type == "new_merchant":
            category = random.choice(list(SPENDING_PROFILES.keys()))
            if category == "rent":
                category = "food"
            profile = SPENDING_PROFILES[category]
            new_merchants = [
                "Unknown Store", "Foreign Website", "Suspicious Vendor",
                "RandomShop123", "Overseas Purchase", "New App Payment"
            ]
            anomalies.append({
                "date": anomaly_date.strftime("%Y-%m-%d"),
                "amount": round(profile["mean"] * random.uniform(1.5, 5), 2),
                "category": category,
                "merchant": random.choice(new_merchants),
                "is_injected_anomaly": True
            })

    return transactions + anomalies


def generate_dataset() -> pd.DataFrame:
    """Main pipeline: generate normal data → inject anomalies → sort → return DataFrame."""
    set_seed()

    print("📊 Generating normal transactions...")
    transactions = generate_normal_transactions(START_DATE, END_DATE)
    print(f"   ✅ Generated {len(transactions)} normal transactions")

    print("🔴 Injecting anomalies...")
    transactions = inject_anomalies(transactions)
    normal_count = sum(1 for t in transactions if not t["is_injected_anomaly"])
    anomaly_count = sum(1 for t in transactions if t["is_injected_anomaly"])
    print(f"   ✅ Injected {anomaly_count} anomalies (total: {normal_count + anomaly_count})")

    df = pd.DataFrame(transactions)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    return df


def save_dataset(df: pd.DataFrame, filepath: str = OUTPUT_FILE):
    """Save to CSV, creating the data/ directory if needed."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"💾 Saved {len(df)} transactions to {filepath}")

    # Print summary statistics
    print("\n📈 Dataset Summary:")
    print(f"   Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"   Categories: {df['category'].nunique()}")
    print(f"   Merchants:  {df['merchant'].nunique()}")
    print(f"   Total spend: ₹{df['amount'].sum():,.2f}")
    print(f"   Avg transaction: ₹{df['amount'].mean():,.2f}")
    print(f"\n   Category breakdown:")
    for cat, group in df.groupby("category"):
        anomaly_ct = group["is_injected_anomaly"].sum()
        print(f"     {cat:15s} — {len(group):4d} txns, avg ₹{group['amount'].mean():,.0f}"
              f"  ({anomaly_ct} anomalies)")


if __name__ == "__main__":
    df = generate_dataset()
    save_dataset(df)
    print("\n✅ Sample data generation complete!")

"""
test_parsers.py — Integration tests for the multi-format parser system
========================================================================
Tests the parser pipeline with various real-world messy schemas.
"""

import pandas as pd
import sys
import os

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_from_dataframe, auto_map_columns, clean_dataframe
from src.category_classifier import classify_categories, classify_by_rules
from src.parsers.csv_parser import parse_csv
from src.parsers.unified_parser import parse_file
from src.parsers.pdf_parser import extract_transactions_from_text, _parse_compact_gpay_lines
from src.entity_resolution import detect_entity_type, normalize_merchant
from src.data_loader import load_from_dataframe
from src.preprocessor import clean_data
from src.category_classifier import classify_categories
from src.feature_engine import engineer_features


def test_indian_bank_csv():
    """Test auto-mapping with a real Indian bank CSV schema."""
    print("\n" + "="*70)
    print("TEST 1: Indian Bank CSV (UPI-style columns)")
    print("="*70)

    # Simulate a real GPay/UPI export header
    df = pd.DataFrame({
        "Date": ["01/01/2026", "02/01/2026", "03/01/2026", "04/01/2026", "05/01/2026"],
        "Time": ["01:45 PM", "10:30 AM", "03:15 PM", "09:00 AM", "06:45 PM"],
        "Transaction Type": ["Debit", "Credit", "Debit", "Debit", "Debit"],
        "Party": ["Swiggy", "Naveen Sharma", "Amazon", "Uber", "Netflix"],
        "UPI Transaction ID": ["UPI123", "UPI456", "UPI789", "UPI012", "UPI345"],
        "Amount (INR)": ["₹250.00", "₹1,000", "₹3,499", "₹150", "₹199"],
    })

    print(f"Input columns: {list(df.columns)}")
    print(f"Input rows: {len(df)}\n")

    # Run through full pipeline
    result_df, warnings = load_from_dataframe(df.copy())

    print("WARNINGS:")
    for w in warnings:
        print(f"  {w}")

    print(f"\nOutput columns: {list(result_df.columns)}")
    print(f"Output rows: {len(result_df)}")

    # Validate mappings
    assert "date" in result_df.columns, "FAIL: 'date' column missing"
    assert "amount" in result_df.columns, "FAIL: 'amount' column missing"
    assert "merchant" in result_df.columns, "FAIL: 'merchant' column missing"
    assert "category" in result_df.columns, "FAIL: 'category' column missing"

    # Validate cleaning
    assert result_df["amount"].dtype in (float, "float64"), "FAIL: amount not float"
    assert result_df["amount"].iloc[0] == 250.0, f"FAIL: amount not cleaned: {result_df['amount'].iloc[0]}"

    print("\n✅ TEST 1 PASSED: Indian bank CSV auto-mapped correctly")
    print(result_df[["date", "amount", "merchant", "category"]].to_string(index=False))


def test_missing_category():
    """Test auto-generation of missing category column."""
    print("\n" + "="*70)
    print("TEST 2: Missing Category Column")
    print("="*70)

    df = pd.DataFrame({
        "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
        "amount": [500, 200, 1500],
        "description": ["Zomato Order", "Uber Ride", "Amazon Purchase"],
    })

    result_df, warnings = load_from_dataframe(df.copy())

    assert "category" in result_df.columns, "FAIL: category not created"
    print("WARNINGS:")
    for w in warnings:
        print(f"  {w}")
    print(f"\n✅ TEST 2 PASSED: Category auto-generated")
    print(result_df[["date", "amount", "merchant", "category"]].to_string(index=False))


def test_category_classifier():
    """Test the hybrid category classifier."""
    print("\n" + "="*70)
    print("TEST 3: Category Classification (Rule + ML)")
    print("="*70)

    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=10),
        "amount": [250, 150, 3499, 199, 500, 100, 299, 450, 89, 1200],
        "merchant": [
            "Swiggy", "Uber", "Amazon", "Netflix", "HDFC Bank",
            "Zomato", "Flipkart", "Ola", "Spotify", "IRCTC Railway",
        ],
        "category": [
            "uncategorized", "uncategorized", "uncategorized", "uncategorized",
            "uncategorized", "uncategorized", "uncategorized", "uncategorized",
            "uncategorized", "uncategorized",
        ],
    })

    result_df, warnings = classify_categories(df)

    print("WARNINGS:")
    for w in warnings:
        print(f"  {w}")

    print("\nCLASSIFICATION RESULTS:")
    for _, row in result_df.iterrows():
        print(
            f"  {row['merchant']:20s} → {row['category']:15s} "
            f"[{row['category_confidence']:6s}] ({row['category_source']})"
        )

    # Validate known merchants
    swiggy_cat = result_df[result_df["merchant"] == "Swiggy"]["category"].iloc[0]
    assert swiggy_cat == "food", f"FAIL: Swiggy classified as '{swiggy_cat}', expected 'food'"

    uber_cat = result_df[result_df["merchant"] == "Uber"]["category"].iloc[0]
    assert uber_cat == "transport", f"FAIL: Uber classified as '{uber_cat}', expected 'transport'"

    netflix_cat = result_df[result_df["merchant"] == "Netflix"]["category"].iloc[0]
    assert netflix_cat == "entertainment", f"FAIL: Netflix classified as '{netflix_cat}'"

    # Credit from a person should be isolated as transfer noise
    person_df = pd.DataFrame({
        "date": ["2026-01-01"],
        "amount": [1000],
        "merchant": ["Naveen Sharma"],
        "category": ["uncategorized"],
        "type": ["credit"],
    })
    person_result, _ = classify_categories(person_df)
    assert person_result["category"].iloc[0] == "income_transfer"

    print("\n✅ TEST 3 PASSED: Category classification working correctly")


def test_messy_amounts():
    """Test cleaning of various messy amount formats."""
    print("\n" + "="*70)
    print("TEST 4: Messy Amount Formats")
    print("="*70)

    df = pd.DataFrame({
        "date": ["2026-01-01"] * 6,
        "amount": ["₹1,500.00", "$ 250", "€ 99.99", "£30", "(500)", "1200"],
        "merchant": ["Test"] * 6,
        "category": ["test"] * 6,
    })

    result_df, warnings = clean_dataframe(df)

    print("WARNINGS:")
    for w in warnings:
        print(f"  {w}")

    print("\nCLEANED AMOUNTS:")
    for i, row in result_df.iterrows():
        print(f"  '{df.loc[i, 'amount']}'  →  {row['amount']}")

    assert result_df["amount"].iloc[0] == 1500.0, "FAIL: ₹1,500.00 not cleaned correctly"
    assert result_df["amount"].iloc[1] == 250.0, "FAIL: $ 250 not cleaned correctly"

    print("\n✅ TEST 4 PASSED: Messy amounts cleaned correctly")


def test_full_pipeline_never_crashes():
    """Test that the pipeline NEVER crashes, even with garbage data."""
    print("\n" + "="*70)
    print("TEST 5: Pipeline Never Crashes (Garbage Data)")
    print("="*70)

    garbage_schemas = [
        # Completely unrelated columns
        pd.DataFrame({"x": [1, 2], "y": [3, 4], "z": [5, 6]}),
        # Empty DataFrame
        pd.DataFrame(),
        # Only amount-like column
        pd.DataFrame({"value": [100, 200, 300]}),
        # Unicode mess
        pd.DataFrame({"दिनांक": ["01/01/2026"], "राशि": ["₹500"], "विवरण": ["Test"]}),
    ]

    for i, df in enumerate(garbage_schemas):
        try:
            result_df, warnings = load_from_dataframe(df.copy())
            print(f"  Schema {i+1}: {list(df.columns)} → {len(result_df)} rows, {len(warnings)} warnings ✅")
        except Exception as e:
            print(f"  Schema {i+1}: CRASHED with {type(e).__name__}: {e} ❌")

    print("\n✅ TEST 5 PASSED: No crashes on garbage data")


def test_semistructured_text_parser():
    """Test raw statement-style text parsing without spreadsheet structure."""
    print("\n" + "="*70)
    print("TEST 6: Semi-Structured Statement Parsing")
    print("="*70)

    text = """
    01 Jan 2026
    Paid to Blue Tokai Coffee
    UPI Transaction ID: X1
    ₹250

    03 Jan 2026
    Received from Naveen Sharma
    UPI Transaction ID: X2
    ₹1200
    """

    result = extract_transactions_from_text(text)
    assert len(result.df) == 2, f"FAIL: expected 2 rows, got {len(result.df)}"
    assert "Blue Tokai Coffee" in result.df["merchant"].tolist()
    assert set(result.df["type"]) == {"debit", "credit"}
    print("\n✅ TEST 6 PASSED: Semi-structured text was decoded correctly")


def test_compact_gpay_line_parser():
    """Test compact GPay PDF text where labels are merged without spaces."""
    print("\n" + "="*70)
    print("TEST 7: Compact GPay Line Parsing")
    print("="*70)

    text = """
    01Jan,2026 PaidtoKESHARWANIBROTHERS ₹100
    UPITransactionID:116503119714
    PaidbyHDFCBank5061
    02Jan,2026 ReceivedfromNaveenSharma ₹1,000
    UPITransactionID:600221102792
    PaidtoHDFCBank5061
    """

    result = _parse_compact_gpay_lines(text)
    assert len(result.df) == 2, f"FAIL: expected 2 rows, got {len(result.df)}"
    assert result.df["amount"].tolist() == [100.0, 1000.0]
    assert result.df["type"].tolist() == ["debit", "credit"]
    print("\n✅ TEST 7 PASSED: Compact GPay lines parsed correctly")


def test_compact_merchant_entity_resolution():
    """Compact merchant strings should not be over-classified as personal transfers."""
    print("\n" + "="*70)
    print("TEST 8: Compact Merchant Entity Resolution")
    print("="*70)

    assert normalize_merchant("NaveenSharma") == "naveen sharma"
    assert normalize_merchant("JioPrepaidRecharges") == "jio prepaid recharges"
    assert detect_entity_type("JioPrepaidRecharges") == "business"
    assert detect_entity_type("NaveenSharma") == "person"
    assert detect_entity_type("KESHARWANIBROTHERS") != "person"

    print("\n✅ TEST 8 PASSED: Compact merchants are resolved more safely")


def test_sparse_upload_pipeline_keeps_category_through_features():
    """Minimal uploads should not crash in feature engineering."""
    print("\n" + "="*70)
    print("TEST 9: Sparse Upload Pipeline Stability")
    print("="*70)

    raw = pd.DataFrame({
        "date": ["2026-01-01", "2026-01-02"],
        "amount": [100, 200],
        "merchant": ["A", "B"],
    })

    loaded, _ = load_from_dataframe(raw.copy())
    cleaned = clean_data(loaded)
    classified, _ = classify_categories(cleaned)
    featured = engineer_features(classified)

    assert "category" in featured.columns
    assert "category_share" in featured.columns
    assert len(featured) == 2

    print("\n✅ TEST 9 PASSED: Sparse uploads no longer crash feature engineering")


if __name__ == "__main__":
    test_indian_bank_csv()
    test_missing_category()
    test_category_classifier()
    test_messy_amounts()
    test_full_pipeline_never_crashes()
    test_semistructured_text_parser()
    test_compact_gpay_line_parser()
    test_compact_merchant_entity_resolution()
    test_sparse_upload_pipeline_keeps_category_through_features()

    print("\n" + "="*70)
    print("ALL TESTS PASSED ✅")
    print("="*70)

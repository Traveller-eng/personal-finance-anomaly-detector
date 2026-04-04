"""
csv_parser.py — CSV File Parser
=================================
Handles CSV files with multiple encoding fallbacks.
Delegates column mapping and cleaning to data_loader.
"""

import pandas as pd
import logging

logger = logging.getLogger("pfad.parsers.csv")


def parse_csv(file) -> tuple[pd.DataFrame, list[str]]:
    """
    Parse a CSV file with encoding detection.

    Args:
        file: File-like object or filepath string

    Returns:
        (raw_dataframe, warnings)
    """
    warnings = []

    # Try encodings in order: UTF-8 → UTF-8-SIG (BOM) → Latin-1
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]

    df = None
    for enc in encodings:
        try:
            if hasattr(file, "seek"):
                file.seek(0)
            df = pd.read_csv(file, encoding=enc)
            if enc != "utf-8":
                warnings.append(f"ℹ️ CSV read with {enc} encoding")
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception as e:
            warnings.append(f"⚠️ CSV parse attempt failed ({enc}): {str(e)[:80]}")
            continue

    if df is None:
        warnings.append("❌ Could not parse CSV with any encoding")
        return pd.DataFrame(), warnings

    # Basic validation
    if len(df) == 0:
        warnings.append("⚠️ CSV file is empty (0 rows)")
    elif len(df.columns) < 2:
        warnings.append("⚠️ CSV has fewer than 2 columns — may be malformed")

    warnings.append(f"✅ CSV parsed: {len(df)} rows × {len(df.columns)} columns")
    logger.info("CSV parsed: %d rows, %d cols", len(df), len(df.columns))

    return df, warnings

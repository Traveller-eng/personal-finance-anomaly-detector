"""
excel_parser.py — Excel File Parser (.xlsx, .xls)
====================================================
Handles Excel files with sheet detection.
Tries the first sheet by default, or uses the largest one.
"""

import pandas as pd
import logging

logger = logging.getLogger("pfad.parsers.excel")


def parse_excel(file) -> tuple[pd.DataFrame, list[str]]:
    """
    Parse an Excel file, selecting the best sheet automatically.

    Args:
        file: File-like object or filepath string

    Returns:
        (raw_dataframe, warnings)
    """
    warnings = []

    try:
        # Read all sheet names first
        if hasattr(file, "seek"):
            file.seek(0)

        xls = pd.ExcelFile(file)
        sheet_names = xls.sheet_names

        if not sheet_names:
            warnings.append("❌ Excel file has no sheets")
            return pd.DataFrame(), warnings

        # Strategy: try sheets in order, pick the one with most rows
        best_df = None
        best_sheet = None
        best_rows = 0

        for sheet in sheet_names:
            try:
                df = pd.read_excel(xls, sheet_name=sheet)
                if len(df) > best_rows:
                    best_df = df
                    best_sheet = sheet
                    best_rows = len(df)
            except Exception:
                continue

        if best_df is None:
            warnings.append("❌ Could not read any sheet from Excel file")
            return pd.DataFrame(), warnings

        if len(sheet_names) > 1:
            warnings.append(
                f"ℹ️ Excel has {len(sheet_names)} sheets — "
                f"using '{best_sheet}' ({best_rows} rows)"
            )

        if best_rows == 0:
            warnings.append("⚠️ Excel sheet is empty (0 rows)")

        warnings.append(
            f"✅ Excel parsed: {best_rows} rows × {len(best_df.columns)} columns"
        )
        logger.info(
            "Excel parsed: sheet='%s', %d rows, %d cols",
            best_sheet, best_rows, len(best_df.columns)
        )

        return best_df, warnings

    except ImportError:
        warnings.append(
            "❌ openpyxl is required for Excel files. "
            "Install with: pip install openpyxl"
        )
        return pd.DataFrame(), warnings

    except Exception as e:
        warnings.append(f"❌ Excel parsing failed: {str(e)[:120]}")
        logger.error("Excel parse error: %s", e)
        return pd.DataFrame(), warnings

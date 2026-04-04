"""
parsers — Multi-Format Financial Document Parsing
===================================================
Supports: CSV, Excel (.xlsx), PDF (GPay, PhonePe, bank statements)

Architecture:
    unified_parser.parse_file(uploaded_file)
        → detects format from filename extension
        → delegates to csv_parser / excel_parser / pdf_parser
        → returns (DataFrame, warnings, parse_metadata)

All parsers produce a standardized output:
    DataFrame with columns: date, amount, merchant, category (+ extras)
"""

from .unified_parser import parse_file, ParseResult

__all__ = ["parse_file", "ParseResult"]

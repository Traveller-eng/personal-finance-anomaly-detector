"""
unified_parser.py — Single Entry Point for All File Formats
=============================================================
PURPOSE:
    Detects file format and delegates to the right parser.
    Every parser returns (DataFrame, warnings).
    This module also runs column mapping and cleaning on the result.

USAGE:
    from src.parsers import parse_file

    result = parse_file(uploaded_file)
    df = result.df
    warnings = result.warnings

SUPPORTED:
    .csv   → csv_parser
    .xlsx  → excel_parser
    .xls   → excel_parser
    .pdf   → pdf_parser (layered regex + table extraction)
"""

import pandas as pd
import logging
from dataclasses import dataclass, field

from .csv_parser import parse_csv
from .excel_parser import parse_excel
from .pdf_parser import parse_pdf

logger = logging.getLogger("pfad.parsers.unified")

# Supported extensions
SUPPORTED_EXTENSIONS = {
    ".csv": "CSV",
    ".xlsx": "Excel",
    ".xls": "Excel",
    ".pdf": "PDF",
}


@dataclass
class ParseResult:
    """
    Unified result from any parser.

    Attributes:
        df:         Parsed DataFrame (may need column mapping still)
        warnings:   List of human-readable warning/info strings
        file_type:  Detected file type string
        file_name:  Original filename
        success:    Whether parsing produced usable data
    """
    df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    warnings: list[str] = field(default_factory=list)
    file_type: str = "unknown"
    file_name: str = ""
    success: bool = False


def _detect_file_type(uploaded_file) -> tuple[str, str]:
    """
    Detect file type from the uploaded file's name.

    Returns:
        (extension, type_label)
    """
    filename = ""

    if hasattr(uploaded_file, "name"):
        filename = uploaded_file.name
    elif isinstance(uploaded_file, str):
        filename = uploaded_file

    filename_lower = filename.lower().strip()

    for ext, label in SUPPORTED_EXTENSIONS.items():
        if filename_lower.endswith(ext):
            return ext, label

    return "", "Unknown"


def parse_file(uploaded_file) -> ParseResult:
    """
    Universal file parser. Detects format and delegates to the right parser.
    NEVER raises — always returns a ParseResult with warnings on failure.

    Args:
        uploaded_file: Streamlit UploadedFile, file path string, or file-like object

    Returns:
        ParseResult with df, warnings, metadata
    """
    result = ParseResult()

    # Get filename
    if hasattr(uploaded_file, "name"):
        result.file_name = uploaded_file.name
    elif isinstance(uploaded_file, str):
        result.file_name = uploaded_file

    # Detect file type
    ext, file_type = _detect_file_type(uploaded_file)
    result.file_type = file_type

    if not ext:
        result.warnings.append(
            f"❌ Unsupported file format: '{result.file_name}'. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS.keys())}"
        )
        return result

    result.warnings.append(f"📂 Detected file type: {file_type} ({ext})")

    # Route to parser
    try:
        if file_type == "CSV":
            df, warnings = parse_csv(uploaded_file)

        elif file_type == "Excel":
            df, warnings = parse_excel(uploaded_file)

        elif file_type == "PDF":
            df, warnings = parse_pdf(uploaded_file)

        else:
            result.warnings.append(f"❌ No parser available for: {file_type}")
            return result

        result.df = df
        result.warnings.extend(warnings)

        # Check result quality
        if df is not None and len(df) > 0:
            result.success = True
            result.warnings.append(
                f"✅ Parsed successfully: {len(df)} rows × {len(df.columns)} columns"
            )
        else:
            result.warnings.append("⚠️ Parser returned empty data")

    except Exception as e:
        result.warnings.append(f"❌ Parser failed: {str(e)[:150]}")
        logger.error("Parse error for %s: %s", result.file_name, e, exc_info=True)

    return result


def get_supported_extensions() -> list[str]:
    """Return list of supported file extensions for Streamlit uploader."""
    return [ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS.keys()]

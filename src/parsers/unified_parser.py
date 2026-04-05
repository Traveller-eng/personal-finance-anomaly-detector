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
    .csv/.tsv          → delimited parser
    .xlsx/.xls         → excel parser
    .pdf               → layered document parser
    .txt/.json/.log    → generic document parser
"""

import pandas as pd
import logging
from dataclasses import dataclass, field

from .csv_parser import parse_csv
from .excel_parser import parse_excel
from .pdf_parser import parse_pdf
from .text_parser import parse_text_document

logger = logging.getLogger("pfad.parsers.unified")

# Supported extensions
SUPPORTED_EXTENSIONS = {
    ".csv": "CSV",
    ".xlsx": "Excel",
    ".xls": "Excel",
    ".pdf": "PDF",
    ".txt": "Text",
    ".json": "Text",
    ".tsv": "Text",
    ".log": "Text",
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
    parse_confidence: float = 0.0
    expected_rows: int = 0
    extracted_rows: int = 0
    fallback_used: bool = False


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

    if ext:
        result.warnings.append(f"📂 Detected file type: {file_type} ({ext})")
    else:
        result.warnings.append(
            f"⚠️ Unknown extension for '{result.file_name or 'uploaded file'}' — trying generic document parsing"
        )
        file_type = "Text"
        result.file_type = file_type

    # Route to parser
    try:
        if file_type == "CSV":
            df, warnings = parse_csv(uploaded_file)

        elif file_type == "Excel":
            df, warnings = parse_excel(uploaded_file)

        elif file_type == "PDF":
            df, warnings = parse_pdf(uploaded_file)

        elif file_type == "Text":
            df, warnings = parse_text_document(uploaded_file)

        else:
            df, warnings = parse_text_document(uploaded_file)

        result.df = df
        result.warnings.extend(warnings)

        (
            result.parse_confidence,
            result.expected_rows,
            result.extracted_rows,
        ) = _estimate_parse_metrics(df)
        result.fallback_used = result.parse_confidence < 0.75

        # Check result quality
        if df is not None and len(df) > 0:
            result.success = True
            result.warnings.append(
                f"✅ Parsed successfully: {len(df)} rows × {len(df.columns)} columns"
            )
            result.warnings.append(
                f"📊 Parse confidence: {int(result.parse_confidence * 100)}% "
                f"({result.extracted_rows}/{max(result.expected_rows, 1)} usable rows)"
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


def _estimate_parse_metrics(df: pd.DataFrame) -> tuple[float, int, int]:
    """Estimate extraction quality for downstream validation and UI."""
    if df is None or len(df) == 0:
        return 0.0, 0, 0

    if "parse_confidence" in df.attrs:
        confidence = float(df.attrs.get("parse_confidence", 0.0))
        expected_rows = int(df.attrs.get("expected_rows", len(df)))
        extracted_rows = int(df.attrs.get("extracted_rows", len(df)))
        return confidence, expected_rows, extracted_rows

    expected_rows = int(len(df))
    extracted_mask = pd.Series(True, index=df.index)
    if "date" in df.columns:
        extracted_mask = extracted_mask & df["date"].notna()
    if "amount" in df.columns:
        extracted_mask = extracted_mask & pd.to_numeric(df["amount"], errors="coerce").notna()

    extracted_rows = int(extracted_mask.sum())
    confidence = round(extracted_rows / max(expected_rows, 1), 2)
    return confidence, expected_rows, extracted_rows

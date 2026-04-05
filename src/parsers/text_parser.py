"""
text_parser.py — Generic Text/JSON Financial Document Parser
============================================================
Purpose:
    Handle semi-structured uploads that are not CSV/Excel but still contain
    transaction-like content, such as exported text, copied statements, or
    simple JSON payloads.
"""

from __future__ import annotations

import json
import logging
from io import StringIO

import pandas as pd

from src.data_loader import clean_dataframe, load_from_dataframe
from .pdf_parser import extract_transactions_from_text

logger = logging.getLogger("pfad.parsers.text")


def _read_text_content(file) -> tuple[str, list[str]]:
    warnings: list[str] = []

    try:
        if isinstance(file, str):
            with open(file, "rb") as handle:
                raw = handle.read()
        elif hasattr(file, "getvalue"):
            raw = file.getvalue()
        elif hasattr(file, "read"):
            if hasattr(file, "seek"):
                file.seek(0)
            raw = file.read()
        else:
            return "", ["❌ Could not read uploaded document"]
    except Exception as exc:
        return "", [f"❌ Failed to read document: {str(exc)[:120]}"]

    if isinstance(raw, str):
        return raw, warnings

    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return raw.decode(encoding), warnings
        except Exception:
            continue

    warnings.append("❌ Document bytes could not be decoded as text")
    return "", warnings


def _try_json_parse(text: str) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    stripped = text.strip()
    if not stripped or not (stripped.startswith("{") or stripped.startswith("[")):
        return pd.DataFrame(), warnings

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return pd.DataFrame(), warnings

    if isinstance(payload, dict):
        payload = payload.get("transactions", [payload])

    if not isinstance(payload, list):
        return pd.DataFrame(), warnings

    rows = [row for row in payload if isinstance(row, dict)]
    if not rows:
        return pd.DataFrame(), warnings

    df = pd.DataFrame(rows)
    df, load_warnings = load_from_dataframe(df)
    warnings.extend(load_warnings)
    warnings.append(f"✅ JSON structure parsed into {len(df)} transaction row(s)")
    return df, warnings


def _try_delimited_parse(text: str) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return pd.DataFrame(), warnings

    header = lines[0]
    for delimiter in (",", "\t", ";", "|"):
        if header.count(delimiter) < 1:
            continue
        try:
            df = pd.read_csv(StringIO(text), sep=delimiter)
        except Exception:
            continue
        if len(df.columns) >= 2:
            df, load_warnings = load_from_dataframe(df)
            warnings.extend(load_warnings)
            warnings.append(
                f"✅ Delimited text parsed using '{delimiter}' separator"
            )
            return df, warnings

    return pd.DataFrame(), warnings


def parse_text_document(file) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = ["🧠 Generic document parser engaged"]
    text, read_warnings = _read_text_content(file)
    warnings.extend(read_warnings)

    if not text.strip():
        warnings.append("❌ No readable text found in document")
        return pd.DataFrame(columns=["date", "amount", "merchant", "category"]), warnings

    json_df, json_warnings = _try_json_parse(text)
    if len(json_df) > 0:
        warnings.extend(json_warnings)
        return json_df, warnings

    delimited_df, delimited_warnings = _try_delimited_parse(text)
    if len(delimited_df) > 0:
        warnings.extend(delimited_warnings)
        return delimited_df, warnings

    result = extract_transactions_from_text(text)
    warnings.extend(result.warnings)

    if len(result.df) == 0:
        warnings.append("❌ Generic parser could not infer transactions from document text")
        return pd.DataFrame(columns=["date", "amount", "merchant", "category"]), warnings

    df = result.df.copy()
    try:
        df, clean_warnings = clean_dataframe(df)
        warnings.extend(clean_warnings)
    except Exception as exc:
        logger.warning("Generic text cleaning failed: %s", exc)

    df.attrs["parse_confidence"] = result.confidence
    df.attrs["expected_rows"] = len(df)
    df.attrs["extracted_rows"] = len(df)
    warnings.append(
        f"✅ Generic text extraction recovered {len(df)} transaction row(s)"
    )
    return df, warnings

"""
pdf_parser.py — Robust Financial PDF Parser
==============================================
PURPOSE:
    Extracts transaction data from bank statement and payment app PDFs.
    Uses a layered regex approach with multiple fallback patterns.

SUPPORTED FORMATS:
    - Google Pay (GPay) statements
    - PhonePe statements
    - Generic UPI transaction PDFs
    - Bank statement PDFs (tabular)

ARCHITECTURE:
    Layer 1: Table extraction (pdfplumber's built-in table parser)
    Layer 2: GPay/UPI regex patterns (primary + fallback)
    Layer 3: Generic line-by-line amount extraction (last resort)

    Each layer feeds into a confidence scorer to assess parse quality.

NEVER CRASHES — returns empty DataFrame + warnings on failure.
"""

import pandas as pd
import re
import logging
from dataclasses import dataclass, field

from src.data_loader import auto_map_columns, clean_dataframe

logger = logging.getLogger("pfad.parsers.pdf")

DATE_PATTERN = re.compile(
    r"("
    r"\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}"
    r"|"
    r"\d{1,2}\s*\w{3,9},?\s*\d{2,4}"
    r")",
    re.IGNORECASE,
)
AMOUNT_PATTERN = re.compile(r"(?:₹|Rs\.?|INR)\s?([\d,]+\.?\d*)", re.IGNORECASE)
TYPE_PATTERN = re.compile(
    r"(Paid\s*to|Sent\s*to|Received\s*from|Received|Debited\s*to|Credited\s*from|Withdrawal|Deposit)",
    re.IGNORECASE,
)
NOISE_LINE_PATTERN = re.compile(
    r"^(UPI\s*Transaction\s*ID|UTR|Paid\s*by|Debited\s*from|Credited\s*to|Paid\s*toHDFC|From\s*bank\s*account|To\s*bank\s*account|Status|Ref(?:erence)?\s*ID)\b",
    re.IGNORECASE,
)


@dataclass
class PDFParseResult:
    """Result of a PDF parse attempt."""
    df: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    confidence: float = 0.0
    method: str = "none"
    raw_text: str = ""


# ─── Text Extraction ─────────────────────────────────────────────────────────

def _extract_text(file) -> tuple[str, list[str]]:
    """
    Extract all text from a PDF using pdfplumber.
    
    Returns:
        (full_text, warnings)
    """
    warnings = []

    try:
        import pdfplumber
    except ImportError:
        warnings.append(
            "❌ pdfplumber is required for PDF parsing. "
            "Install with: pip install pdfplumber"
        )
        return "", warnings

    try:
        if hasattr(file, "seek"):
            file.seek(0)

        text = ""
        page_count = 0

        with pdfplumber.open(file) as pdf:
            page_count = len(pdf.pages)
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        if not text.strip():
            warnings.append("⚠️ PDF has no extractable text (may be scanned/image-based)")
        else:
            warnings.append(f"✅ PDF text extracted: {page_count} pages, {len(text)} chars")

        return text, warnings

    except Exception as e:
        warnings.append(f"❌ PDF text extraction failed: {str(e)[:120]}")
        logger.error("PDF extraction error: %s", e)
        return "", warnings


def _extract_tables(file) -> tuple[list[pd.DataFrame], list[str]]:
    """
    Extract tabular data from PDF using pdfplumber's table detector.
    
    Returns:
        (list_of_dataframes, warnings)
    """
    warnings = []

    try:
        import pdfplumber
    except ImportError:
        return [], warnings

    try:
        if hasattr(file, "seek"):
            file.seek(0)

        tables = []
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_tables = page.extract_tables()
                for table in page_tables:
                    if table and len(table) > 1:
                        # First row as header
                        header = [str(h).strip() if h else f"col_{i}" for i, h in enumerate(table[0])]
                        rows = table[1:]
                        df = pd.DataFrame(rows, columns=header)
                        if len(df) > 0:
                            tables.append(df)

        if tables:
            warnings.append(f"✅ Found {len(tables)} table(s) in PDF")
        return tables, warnings

    except Exception as e:
        logger.warning("PDF table extraction failed: %s", e)
        return [], warnings


# ─── Text Pre-Processing ─────────────────────────────────────────────────────

def _clean_pdf_text(text: str) -> str:
    """
    Remove noise from PDF text before regex parsing.
    This dramatically improves regex accuracy by removing lines that
    look like transaction metadata but aren't merchant names.
    """
    # Remove UPI transaction IDs (noise)
    text = re.sub(r"UPI [Tt]ransaction ID[:\s]*\S+", "", text)
    text = re.sub(r"UTR[:\s]*\S+", "", text)

    # Remove bank account references that pollute merchant extraction
    text = re.sub(r"Paid by\s+.*", "", text)
    text = re.sub(r"Debited from\s+.*", "", text)
    text = re.sub(r"Credited to\s+.*", "", text)

    # Remove timestamps (they break line-based patterns)
    text = re.sub(r"\d{1,2}:\d{2}\s*[APap][Mm]", "", text)

    # Collapse multiple whitespace
    text = re.sub(r"[ \t]+", " ", text)

    return text


def _normalize_amount_str(amount_str: str) -> str:
    """Normalize currency strings to a plain numeric value string."""
    if amount_str is None:
        return ""
    amt = str(amount_str).strip()
    amt = re.sub(r"^(?:₹|Rs\.?|INR)\s*", "", amt, flags=re.IGNORECASE)
    amt = amt.replace(",", "")
    return amt


def _normalize_merchant_candidate(line: str) -> str:
    candidate = re.sub(r"\s+", " ", str(line or "").strip(" :-"))
    candidate = re.sub(
        r"^(Paid\s*to|Sent\s*to|Received\s*from|Received|Debited\s*to|Credited\s*from)\s*",
        "",
        candidate,
        flags=re.IGNORECASE,
    )
    candidate = re.sub(r"(?:₹|Rs\.?|INR)\s?[\d,]+\.?\d*", "", candidate, flags=re.IGNORECASE)
    candidate = candidate.strip(" -:")
    if not candidate or NOISE_LINE_PATTERN.match(candidate):
        return ""
    if candidate.lower() in {"transaction", "payment", "transfer", "upi"}:
        return ""
    return candidate[:100]


def _extract_transaction_blocks(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return []

    blocks: list[list[str]] = []
    current: list[str] = []

    for line in lines:
        if DATE_PATTERN.search(line) and current:
            blocks.append(current)
            current = [line]
        else:
            current.append(line)

    if current:
        blocks.append(current)

    normalized_blocks = ["\n".join(block).strip() for block in blocks if len(block) >= 2]
    return normalized_blocks


def _infer_merchant_from_block(block_lines: list[str]) -> str:
    for line in block_lines:
        if NOISE_LINE_PATTERN.match(line):
            continue
        if DATE_PATTERN.search(line) and len(line.split()) <= 4:
            continue
        normalized = _normalize_merchant_candidate(line)
        if normalized:
            return normalized
    return "Unknown (PDF extract)"


def _parse_semistructured_blocks(text: str) -> PDFParseResult:
    result = PDFParseResult(df=pd.DataFrame(), method="semistructured_text")
    blocks = _extract_transaction_blocks(text)

    if not blocks:
        return result

    data = []
    for block in blocks:
        block_lines = [line.strip() for line in block.splitlines() if line.strip()]
        block_text = "\n".join(block_lines)

        date_match = DATE_PATTERN.search(block_text)
        amount_matches = AMOUNT_PATTERN.findall(block_text)
        if not date_match or not amount_matches:
            continue

        type_match = TYPE_PATTERN.search(block_text)
        merchant = _infer_merchant_from_block(block_lines[1:] if len(block_lines) > 1 else block_lines)

        try:
            amount = float(_normalize_amount_str(amount_matches[-1]))
        except ValueError:
            continue

        tx_type = "debit"
        if type_match:
            label = type_match.group(1).lower()
            if "receive" in label or "credit" in label or "deposit" in label:
                tx_type = "credit"

        data.append(
            {
                "date": date_match.group(1).strip(),
                "merchant": merchant,
                "amount": amount,
                "type": tx_type,
                "category": "uncategorized",
            }
        )

    if data:
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
        df = df.dropna(subset=["date"])
        result.df = df
        result.warnings.append(
            f"✅ Semi-structured statement parser matched {len(df)} transaction blocks"
        )

    return result


def _parse_compact_gpay_lines(text: str) -> PDFParseResult:
    """
    Handle compact GPay statements where words are concatenated, e.g.
    `01Jan,2026 PaidtoKESHARWANIBROTHERS ₹100`.
    """
    result = PDFParseResult(df=pd.DataFrame(), method="gpay_compact_lines")
    line_pattern = re.compile(
        r"^"
        r"(?P<date>\d{1,2}\w{3,9},\d{4})"
        r"\s+"
        r"(?P<label>Paidto|Sentto|Receivedfrom|Received)"
        r"(?P<merchant>.*?)"
        r"\s+"
        r"(?:₹|Rs\.?|INR)\s?(?P<amount>[\d,]+\.?\d*)"
        r"$",
        re.IGNORECASE,
    )

    rows = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("Transaction statement") or line.startswith("Date&time"):
            continue
        if line.startswith("Note:") or line.startswith("Page"):
            continue

        match = line_pattern.match(line)
        if not match:
            continue

        merchant = _normalize_merchant_candidate(match.group("merchant"))
        if not merchant:
            continue

        label = match.group("label").lower()
        tx_type = "credit" if "received" in label else "debit"

        try:
            amount = float(_normalize_amount_str(match.group("amount")))
        except ValueError:
            continue

        rows.append(
            {
                "date": match.group("date"),
                "merchant": merchant,
                "amount": amount,
                "type": tx_type,
                "category": "uncategorized",
            }
        )

    if rows:
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"], format="%d%b,%Y", errors="coerce")
        df = df.dropna(subset=["date"])
        result.df = df
        result.warnings.append(
            f"✅ Compact GPay line parser matched {len(df)} transactions"
        )

    return result


# ─── Layer 2: GPay / UPI Regex Patterns ───────────────────────────────────────

def _parse_gpay_pattern(text: str) -> PDFParseResult:
    """
    Primary regex for Google Pay / UPI statement PDFs.
    
    Expected pattern per transaction:
        01 Jan, 2026
        01:45 PM
        Paid to KESHARWANI BROTHERS
        UPI Transaction ID: XXXXX
        Paid by HDFC Bank 5061
        ₹100

    We've already cleaned the text to remove noise lines.
    """
    result = PDFParseResult(df=pd.DataFrame(), method="gpay_primary")

    # ── Primary Pattern: date → type → merchant → ₹amount
    primary_pattern = r"""
        (\d{1,2}\s*\w{3,9},?\s*\d{4})    # date: 1 Jan 2026 / 01Jan,2026 / 01 January 2026
        .*?                                   # skip time, spacing, noise
        (Paid\s*to|Sent\s*to|Received\s*from|Received)  # transaction type
        \s*(.*?)                              # merchant name (lazy capture)
        [\r\n]+.*?                          # skip remaining lines until amount
        (?:₹|Rs\.?|INR)\s?([\d,]+\.?\d*)  # amount: ₹100 or Rs.1,200.50 or INR 100
    """

    matches = re.findall(primary_pattern, text, re.DOTALL | re.VERBOSE | re.IGNORECASE)

    if matches:
        data = []
        for date_str, txn_type, merchant, amount_str in matches:
            try:
                amount = float(_normalize_amount_str(amount_str))
                data.append({
                    "date": date_str.strip(),
                    "merchant": merchant.strip(),
                    "amount": amount,
                    "type": "debit" if "paid" in txn_type.lower() or "sent" in txn_type.lower() else "credit",
                    "category": "uncategorized",
                })
            except (ValueError, AttributeError):
                continue

        if data:
            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
            result.df = df
            result.method = "gpay_primary"
            result.warnings.append(
                f"✅ GPay primary pattern matched {len(data)} transactions"
            )

    return result


def _parse_gpay_fallback(text: str) -> PDFParseResult:
    """
    Fallback regex — more lenient pattern for GPay-like PDFs
    when primary pattern underfits.
    """
    result = PDFParseResult(df=pd.DataFrame(), method="gpay_fallback")

    # More lenient: just date + ₹amount, try to find merchant between
    fallback_pattern = r"""
        (\d{1,2}\s*\w{3,9},?\s*\d{4})    # date
        [\s\S]*?                             # anything
        (Paid\s*to|Sent\s*to|Received\s*from|Received)  # type
        \s*(.+?)                              # merchant (greedy until newline)
        [\s\S]*?                             # skip
        (?:₹|Rs\.?|INR)\s?([\d,]+\.?\d*)  # amount (no decimals required)
    """

    matches = re.findall(fallback_pattern, text, re.DOTALL | re.VERBOSE | re.IGNORECASE)

    if matches:
        data = []
        for date_str, txn_type, merchant, amount_str in matches:
            try:
                amount = float(_normalize_amount_str(amount_str))
                if amount > 0:
                    data.append({
                        "date": date_str.strip(),
                        "merchant": merchant.strip().split("\n")[0],  # first line only
                        "amount": amount,
                        "type": "debit" if "paid" in txn_type.lower() or "sent" in txn_type.lower() else "credit",
                        "category": "uncategorized",
                    })
            except (ValueError, AttributeError):
                continue

        if data:
            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
            result.df = df
            result.warnings.append(
                f"✅ GPay fallback pattern matched {len(data)} transactions"
            )

    return result


def _parse_phonepe_pattern(text: str) -> PDFParseResult:
    """
    PhonePe statement pattern.
    Similar to GPay but different header structure.
    """
    result = PDFParseResult(df=pd.DataFrame(), method="phonepe")

    pattern = r"""
        (\d{1,2}\s*\w{3,9},?\s*\d{4})    # date: 1 Jan 2026 / 01Jan,2026 / 01 January 2026
        .*?
        (Sent\s*to|Received\s*from|Paid\s*to|Received)    # type
        \s*(.*?)                        # merchant
        .*?
        (?:₹|Rs\.?|INR)\s?([\d,]+\.?\d*)             # amount
    """

    matches = re.findall(pattern, text, re.DOTALL | re.VERBOSE | re.IGNORECASE)

    if matches:
        data = []
        for date_str, txn_type, merchant, amount_str in matches:
            try:
                amount = float(_normalize_amount_str(amount_str))
                if amount > 0:
                    data.append({
                        "date": date_str.strip(),
                        "merchant": merchant.strip().split("\n")[0],
                        "amount": amount,
                        "type": "debit" if "sent" in txn_type.lower() or "paid" in txn_type.lower() else "credit",
                        "category": "uncategorized",
                    })
            except (ValueError, AttributeError):
                continue

        if data:
            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
            result.df = df
            result.warnings.append(
                f"✅ PhonePe pattern matched {len(data)} transactions"
            )

    return result


def _parse_generic_amounts(text: str) -> PDFParseResult:
    """
    Layer 3: Last-resort parser.
    Finds ANY date + ₹amount pair in the text.
    Lower confidence, but catches edge cases.
    """
    result = PDFParseResult(df=pd.DataFrame(), method="generic_amount")

    # Find dates
    date_pattern = r"(\d{1,2}[\s/\-]\w{3}[\s/\-,]\s?\d{2,4})"
    # Find amounts with ₹, Rs, or INR
    amount_pattern = r"(?:₹|Rs\.?|INR)\s?([\d,]+\.?\d*)"

    dates = re.findall(date_pattern, text, re.IGNORECASE)
    amounts = re.findall(amount_pattern, text)

    if dates and amounts:
        # Pair them by order of appearance (best effort)
        pair_count = min(len(dates), len(amounts))
        data = []
        for i in range(pair_count):
            try:
                amount = float(amounts[i].replace(",", ""))
                if amount > 0:
                    data.append({
                        "date": dates[i].strip(),
                        "merchant": "Unknown (PDF extract)",
                        "amount": amount,
                        "category": "uncategorized",
                    })
            except (ValueError, AttributeError):
                continue

        if data:
            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
            # Drop rows where date parsing failed
            df = df.dropna(subset=["date"])
            result.df = df
            result.warnings.append(
                f"⚠️ Generic extraction: found {len(data)} date-amount pairs "
                f"(lower confidence — merchants not identified)"
            )

    return result


def extract_transactions_from_text(text: str) -> PDFParseResult:
    """
    Public helper for generic document parsing outside native PDFs.
    Reuses the same layered extraction logic on already-decoded text.
    """
    cleaned_text = _clean_pdf_text(text)
    candidates = [
        _parse_compact_gpay_lines(cleaned_text),
        _parse_gpay_pattern(cleaned_text),
        _parse_gpay_fallback(cleaned_text),
        _parse_phonepe_pattern(cleaned_text),
        _parse_semistructured_blocks(cleaned_text),
        _parse_bank_statement_line(cleaned_text),
        _parse_generic_amounts(cleaned_text),
    ]

    best_result = PDFParseResult(df=pd.DataFrame(), method="none")
    best_confidence = 0.0
    for candidate in candidates:
        if len(candidate.df) == 0:
            continue
        confidence = _calculate_confidence(candidate, text)
        candidate.confidence = confidence
        if confidence >= best_confidence:
            best_confidence = confidence
            best_result = candidate

    return best_result


def _parse_bank_statement_line(text: str) -> PDFParseResult:
    """
    Generic bank statement parser for line-formatted statements.
    Pattern: date | narration/description | debit | credit | balance
    """
    result = PDFParseResult(df=pd.DataFrame(), method="bank_statement")

    # Common bank statement pattern (Indian banks)
    # DD/MM/YYYY or DD-MM-YYYY  |  description  |  amount
    pattern = r"""
        (\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})    # date
        \s+                                     # separator
        (.+?)                                   # description/narration
        \s+                                     # separator
        ([\d,]+\.\d{2})                         # amount (with decimals)
    """

    matches = re.findall(pattern, text, re.VERBOSE)

    if len(matches) >= 3:  # Need at least 3 transactions to be meaningful
        data = []
        for date_str, desc, amount_str in matches:
            try:
                amount = float(amount_str.replace(",", ""))
                if amount > 0:
                    data.append({
                        "date": date_str.strip(),
                        "merchant": desc.strip()[:80],  # Cap length
                        "amount": amount,
                        "category": "uncategorized",
                    })
            except (ValueError, AttributeError):
                continue

        if data:
            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
            df = df.dropna(subset=["date"])
            result.df = df
            result.warnings.append(
                f"✅ Bank statement pattern matched {len(data)} transactions"
            )

    return result


# ─── Confidence Scoring ──────────────────────────────────────────────────────

def _calculate_confidence(result: PDFParseResult, raw_text: str) -> float:
    """
    Calculate parsing confidence as a ratio of:
    - Transactions found vs expected (based on ₹ symbol count)
    - Data completeness (non-null dates, valid amounts)

    Returns float between 0.0 and 1.0
    """
    if result.df is None or len(result.df) == 0:
        return 0.0

    df = result.df

    # Expected transaction count (rough estimate from ₹ occurrences)
    rupee_count = raw_text.count("₹")
    if rupee_count == 0:
        rupee_count = 1

    extraction_ratio = min(len(df) / rupee_count, 1.0)

    # Data completeness
    has_date = df["date"].notna().mean() if "date" in df.columns else 0
    has_amount = (df["amount"] > 0).mean() if "amount" in df.columns else 0
    has_merchant = (
        df["merchant"].notna() & (df["merchant"] != "Unknown (PDF extract)")
    ).mean() if "merchant" in df.columns else 0

    # Weighted score
    confidence = (
        extraction_ratio * 0.4 +
        has_date * 0.25 +
        has_amount * 0.25 +
        has_merchant * 0.1
    )

    return round(min(confidence, 1.0), 2)


# ─── Layer 1: Table-Based Parsing ─────────────────────────────────────────────

def _try_table_parse(file) -> PDFParseResult:
    """
    Try extracting transactions from tabular PDF structure.
    This works for formal bank statements with grid layouts.
    """
    result = PDFParseResult(df=pd.DataFrame(), method="table")

    tables, warnings = _extract_tables(file)
    result.warnings.extend(warnings)

    if not tables:
        return result

    # Concatenate all tables and pick the largest
    best = max(tables, key=len)

    # Check if it has enough columns to be a transaction table
    if len(best.columns) >= 3:
        df = best.copy()
        rename_map, map_warnings = auto_map_columns(df)
        result.warnings.extend(map_warnings)

        if rename_map:
            df = df.rename(columns=rename_map)

        try:
            df, clean_warnings = clean_dataframe(df)
            result.warnings.extend(clean_warnings)
        except Exception as exc:
            logger.warning("PDF table cleaning failed: %s", exc)

        if "date" in df.columns and "amount" in df.columns:
            result.df = df
            result.warnings.append(
                f"✅ Table extraction: {len(df)} rows from PDF tables"
            )

    return result


# ─── Main Parse Function ─────────────────────────────────────────────────────

def parse_pdf(file) -> tuple[pd.DataFrame, list[str]]:
    """
    Master PDF parser with layered fallback strategy.

    Strategy:
        1. Try table extraction (formal bank statements)
        2. Extract text → clean → try GPay pattern
        3. If GPay < 5 matches → try GPay fallback
        4. Try PhonePe pattern
        5. Try generic bank statement line pattern
        6. Last resort: generic date-amount extraction

    Returns:
        (dataframe, warnings_list)
    """
    all_warnings = []
    all_warnings.append("📄 PDF file detected — running layered extraction")

    # ── Layer 1: Table extraction ─────────────────────────────────────────
    table_result = _try_table_parse(file)
    all_warnings.extend(table_result.warnings)

    if len(table_result.df) >= 5:
        # Table extraction looks good — use it
        table_result.confidence = _calculate_confidence(table_result, "")
        confidence_pct = int(table_result.confidence * 100)
        all_warnings.append(
            f"📊 Using table extraction ({confidence_pct}% confidence)"
        )
        return table_result.df, all_warnings

    # ── Layer 2+: Text-based extraction ───────────────────────────────────
    raw_text, text_warnings = _extract_text(file)
    all_warnings.extend(text_warnings)

    if not raw_text.strip():
        all_warnings.append("❌ No text extracted from PDF — cannot parse")
        return pd.DataFrame(columns=["date", "amount", "merchant", "category"]), all_warnings

    # Clean text for regex
    cleaned_text = _clean_pdf_text(raw_text)

    # Try parsers in priority order, keep the best result
    parsers = [
        ("GPay Compact", _parse_compact_gpay_lines),
        ("GPay Primary", _parse_gpay_pattern),
        ("GPay Fallback", _parse_gpay_fallback),
        ("PhonePe", _parse_phonepe_pattern),
        ("Semi-Structured", _parse_semistructured_blocks),
        ("Bank Statement", _parse_bank_statement_line),
        ("Generic Amount", _parse_generic_amounts),
    ]

    best_result = None
    best_confidence = 0.0

    for name, parser_fn in parsers:
        try:
            result = parser_fn(cleaned_text)
            result.raw_text = raw_text

            if len(result.df) > 0:
                conf = _calculate_confidence(result, raw_text)
                result.confidence = conf

                logger.info(
                    "Parser '%s': %d rows, confidence=%.2f",
                    name, len(result.df), conf
                )

                if conf > best_confidence:
                    best_confidence = conf
                    best_result = result

                # If primary GPay gives >= 5 good results, stop
                if name == "GPay Primary" and len(result.df) >= 5 and conf >= 0.5:
                    all_warnings.extend(result.warnings)
                    break

        except Exception as e:
            logger.warning("Parser '%s' failed: %s", name, e)
            all_warnings.append(f"⚠️ {name} parser encountered error: {str(e)[:80]}")
            continue

    # Use best result
    if best_result and len(best_result.df) > 0:
        all_warnings.extend(best_result.warnings)
        confidence_pct = int(best_result.confidence * 100)
        confidence_label = (
            "High" if confidence_pct >= 80 else
            "Medium" if confidence_pct >= 50 else
            "Low"
        )
        all_warnings.append(
            f"📊 Parse confidence: {confidence_pct}% ({confidence_label}) "
            f"via {best_result.method}"
        )

        # Validate: drop rows with missing critical data
        df = best_result.df.copy()
        pre_validate = len(df)

        if "date" in df.columns:
            df = df.dropna(subset=["date"])
        if "amount" in df.columns:
            df = df[df["amount"] > 0]

        dropped = pre_validate - len(df)
        if dropped > 0:
            all_warnings.append(
                f"⚠️ Validation dropped {dropped} rows (missing date or invalid amount)"
            )

        all_warnings.append(f"✅ Final: {len(df)} valid transactions extracted from PDF")
        return df, all_warnings

    # Complete failure
    all_warnings.append("❌ No transactions could be extracted from PDF")
    return pd.DataFrame(columns=["date", "amount", "merchant", "category"]), all_warnings

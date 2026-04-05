# PFAD

![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-10B981?style=flat-square)
![License](https://img.shields.io/badge/Use-Local%20Project-64748B?style=flat-square)

PFAD is a Streamlit-based personal finance analysis system that turns messy transaction exports into structured behavioral insights. It ingests bank and wallet statements, normalizes them into a canonical schema, filters transfer noise, classifies spending, detects unusual patterns, and presents a small set of actionable recommendations.

## Overview

PFAD is built for financial data that is messy in practice:

- inconsistent CSV headers
- multi-sheet Excel exports
- semi-structured PDF statements
- compact wallet statements where labels are merged into text

The system converts those inputs into a clean analytical pipeline so the dashboard works from one canonical representation instead of file-specific logic.

## Highlights

- Multi-format ingestion for CSV, Excel, PDF, and text-like financial exports
- Compact GPay PDF support, including statements where labels are merged like `Paidto...` and `Receivedfrom...`
- Strict transaction normalization with entity resolution and transfer filtering
- Hybrid category classification using memory, rules, and ML fallback
- Ensemble anomaly detection with severity ranking
- Behavior-oriented insights instead of raw anomaly dumps
- Streamlit dashboard with trend, anomaly, profile, and health views

## Feature Summary

| Area | What PFAD Does |
| --- | --- |
| Ingestion | Parses CSV, TSV, Excel, PDF, and text-like financial exports |
| Normalization | Maps messy source columns into a strict transaction schema |
| Entity Resolution | Distinguishes likely people, businesses, and transfer noise |
| Classification | Uses user memory, rules, and ML fallback to label spend |
| Detection | Combines Isolation Forest and z-score signals |
| Insights | Produces ranked behavioral explanations and suggested actions |
| UI | Visualizes spend, anomalies, cashflow, profile, and health score |

## Architecture

```text
Ingestion
  -> Normalization
  -> Entity Resolution
  -> Classification
  -> Feature Engineering
  -> Signal Engine
  -> Insight Engine
  -> Action Layer
  -> Streamlit UI
```

## Canonical Transaction Schema

Every transaction is normalized toward the following shape before downstream analysis:

```python
{
    "date": datetime,
    "amount": float,
    "type": "debit" | "credit",
    "merchant": str,
    "merchant_normalized": str,
    "entity_type": "person" | "business" | "unknown",
    "category": str,
    "category_confidence": "high" | "low" | "manual",
    "source": "rule" | "ml" | "user",
    "is_transfer": bool,
}
```

## Supported Inputs

PFAD is designed for imperfect real-world exports rather than only clean spreadsheets.

- CSV and TSV files with inconsistent headers
- Excel files with one or more sheets
- PDF statements from Google Pay, PhonePe, and generic tabular statements
- Text-like exports such as `.txt`, `.json`, and `.log`

The parser stack uses layered extraction:

1. File-type routing
2. Table extraction where available
3. Semi-structured text parsing
4. Compact statement parsing for merged labels
5. Fallback normalization and validation

## Core Capabilities

### 1. Ingestion and Validation

- Fuzzy column mapping for inconsistent source schemas
- Parse-confidence metadata for parser quality checks
- Safe validation that drops invalid rows instead of crashing
- Generic fallback for semi-structured text documents

### 2. Entity Resolution

- Merchant normalization for stable matching
- Compact-name cleanup for exports like `NaveenSharma` and `JioPrepaidRecharges`
- Person, business, and unknown entity inference
- Transfer detection to suppress person-to-person noise from behavioral analytics

### 3. Category Classification

Priority order:

1. User memory
2. Entity rules
3. Merchant keyword rules
4. Existing uploaded category
5. ML fallback
6. `others`

Persistent user overrides are stored in [data/merchant_category_map.json](/c:/Users/Lenovo/OneDrive/Desktop/Personal%20Finance%20Anomaly%20Detector/data/merchant_category_map.json).

### 4. Feature Engineering

PFAD computes behavioral features such as:

- `rolling_mean_30d`
- `rolling_std_30d`
- `transaction_frequency`
- `merchant_frequency`
- `category_share`
- `velocity_tx_per_day`
- `weekend_flag`
- `new_merchant_flag`

### 5. Signal Engine

- Isolation Forest for unsupervised outlier detection
- Category-aware z-score checks
- Severity labels: `critical`, `warning`, `normal`

### 6. Insight and Action Layer

Instead of showing only statistical anomalies, PFAD produces ranked behavioral insights with:

- `problem`
- `cause`
- `impact`
- `action`
- `expected_gain`

## Example

### Input

```text
Date,Party,Transaction Type,Amount (INR)
01/01/2026,Naveen Sharma,Credit,1000
02/01/2026,Swiggy,Debit,450
03/01/2026,Swiggy,Debit,1200
```

### Normalized Interpretation

```text
Naveen Sharma -> entity_type=person   -> category=income_transfer  -> is_transfer=True
Swiggy        -> entity_type=business -> category=food             -> is_transfer=False
```

### Example Insight

```json
{
  "title": "Food spend drift",
  "problem": "Food spending accelerated faster than your usual pattern.",
  "cause": "Recent transactions pushed the category above its trailing baseline.",
  "impact": "Current pace suggests avoidable discretionary leakage.",
  "action": "Set a short-term spending cap and review the latest food transactions.",
  "expected_gain": "Bringing this category back to baseline improves cash discipline."
}
```

## Project Structure

```text
app.py
src/
  anomaly_detector.py
  category_classifier.py
  data_loader.py
  database.py
  entity_resolution.py
  explainer.py
  feature_engine.py
  health_scorer.py
  insights.py
  preprocessor.py
  recurring.py
  transaction_schema.py
  user_profiler.py
  parsers/
tests/
data/
requirements.txt
config_example.yaml
```

## Local Setup

### Requirements

- Python 3.12 or later recommended
- `pip`

### Install

```bash
python -m pip install -r requirements.txt
```

### Configure Authentication

Copy the example config and update credentials as needed:

```bash
copy config_example.yaml config.yaml
```

## Account Setup

PFAD supports both guest access and permanent local accounts.

### Guest Mode

Guest mode is useful for trying the app without saving account-specific changes. In guest mode, database-backed actions such as persistent overrides or saved expectations are intentionally limited.

### Permanent Local Accounts

Permanent accounts are configured in `config.yaml` using hashed passwords compatible with `streamlit-authenticator`.

#### 1. Generate a password hash

Use the included utility:

```bash
python src/utils/hash_gen.py
```

This script will:

- prompt for a password
- generate a bcrypt hash
- print a `password: "<hash>"` value you can paste into `config.yaml`

#### 2. Add the user to `config.yaml`

Example:

```yaml
cookie:
  expiry_days: 1
  key: random_signature_key
  name: pfad_auth

credentials:
  usernames:
    admin:
      email: admin@pfad.local
      name: Admin User
      password: "$2b$12$existinghash..."

    analyst:
      email: analyst@pfad.local
      name: Analyst User
      password: "$2b$12$paste_generated_hash_here..."
```

#### 3. Restart the app

After updating `config.yaml`, restart Streamlit:

```bash
streamlit run app.py
```

The new account will then be available on the login screen.

### Notes

- Do not store plain-text passwords in `config.yaml`
- Always use the generated bcrypt hash
- Keep `config.yaml` local and private if it contains real credentials
- `config_example.yaml` is only a template, not your live account store

### Run the App

```bash
streamlit run app.py
```

## Dashboard Views

- Overview: health summary, spending mix, cashflow, top insights
- Anomalies: ensemble anomaly explorer with severity and explanations
- Trends: daily, weekly, and monthly debit-spend trends
- Profile: category and temporal spending behavior
- Health: score breakdown and financial guidance

## Performance Notes

The UI exposes operational metrics such as:

- Precision
- Recall
- False Alert Rate

When true labels are unavailable, PFAD falls back to proxy diagnostics such as anomaly density and average anomaly score.

## Security and Local State

- Authentication uses `streamlit-authenticator`
- Passwords are stored as hashed values in `config.yaml`
- Guest mode supports evaluation without persistent changes
- Local state is stored in files such as `pfad.db` and merchant memory JSON

## Limitations

- Entity resolution is heuristic and may still leave ambiguous counterparties as `unknown`
- PDF parsing is best-effort for image-only or badly OCR'd statements
- ML fallback depends on enough labeled merchant history in the active dataset
- Insight recommendations are advisory, not financial guarantees
- This is a local analytics tool, not a regulated financial product

## Recent Parser Improvement

The ingestion layer now correctly handles compact Google Pay statement PDFs where transaction labels are merged into strings such as:

```text
01Jan,2026 PaidtoKESHARWANIBROTHERS ₹100
02Jan,2026 ReceivedfromNaveenSharma ₹1,000
```

This reduced false transfer classification and brought PDF-based trend views back in line with the equivalent tabular representation.

## Screenshots

Suggested sections for repository screenshots:

- Overview dashboard
- Anomaly explorer
- Spending trends
- Behavioral profile

If you add screenshots later, a clean structure would be:

```text
assets/
  screenshots/
    overview.png
    anomalies.png
    trends.png
    profile.png
```

Then embed them like this:

```markdown
![Overview](assets/screenshots/overview.png)
![Anomalies](assets/screenshots/anomalies.png)
```

## Roadmap

- Improve merchant display formatting for compact PDF names
- Expand parser support for more wallet and bank statement layouts
- Add richer evaluation tooling for parser quality and alert quality
- Improve budgeting and recurring-payment workflows

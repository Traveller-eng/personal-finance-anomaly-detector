# Personal Finance Anomaly Detector (PFAD)

**Local-first financial behavior analysis system for detecting, explaining, and correcting anomalous spending patterns.**

---

## Overview

PFAD is a behavior-aware financial analysis system that transforms raw transaction logs into structured insights and actionable recommendations.

Unlike traditional expense trackers that focus on aggregation and visualization, PFAD focuses on:

```text
behavior -> deviation -> explanation -> action
```

It processes historical transactions, models baseline spending behavior, detects statistically significant deviations using an ensemble approach, and generates structured recommendations to improve financial stability.

---

## System Architecture

```text
Raw Input
-> Parser Layer (CSV / Excel / PDF / Text)
-> Normalization (schema alignment)
-> Entity Resolution (person vs business)
-> Classification (rule-based + adaptive memory)
-> Feature Engineering (rolling statistics)
-> Signal Engine (Isolation Forest + Z-score)
-> Insight Engine (behavioral analysis)
-> Action Layer (decision generation)
-> UI
```

---

## Core Components

### 1. Multi-Format Ingestion

PFAD supports:

- CSV / TSV exports
- Excel exports
- Semi-structured PDFs such as GPay, PhonePe, and bank statements
- Text-like financial documents such as `.txt`, `.json`, and `.log`

Key techniques:

- fuzzy column mapping
- layered parser routing
- regex-based extraction pipelines
- compact-statement parsing for merged labels
- validation and row filtering

### 2. Data Normalization

All inputs are converted into a unified schema:

```python
{
    "date": datetime,
    "amount": float,
    "type": "debit|credit",
    "merchant": str,
    "merchant_normalized": str,
    "entity_type": "person|business|unknown",
    "category": str,
    "category_confidence": "high|low|manual",
    "source": "rule|ml|user",
    "is_transfer": bool
}
```

This ensures downstream consistency across all modules.

### 3. Entity Resolution

A critical step to avoid semantic errors in analysis.

```text
merchant -> entity_type
```

- person -> transfer
- business -> categorized normally
- unknown -> retained for cautious downstream handling

This prevents misclassification such as:

```text
"Naveen Sharma" -> shopping          x
"Naveen Sharma" -> income_transfer   check
```

It also handles compact merchant strings from PDFs such as:

```text
NaveenSharma -> naveen sharma
JioPrepaidRecharges -> jio prepaid recharges
```

### 4. Category Classification (Hybrid System)

Classification priority:

1. User-defined mapping (persistent memory)
2. Entity-based rules (person -> transfer)
3. Keyword-based rules (for example `zomato -> food`)
4. Existing provided category
5. ML fallback
6. Fallback -> `others`

The system is adaptive:

- user overrides are stored locally
- mappings are reused across sessions
- low-confidence rows can be corrected in the UI

### 5. Feature Engineering

PFAD models behavior using rolling statistical features:

- rolling mean (30-day baseline)
- rolling standard deviation
- transaction frequency
- merchant frequency
- category share
- velocity (transactions/day)
- weekend indicator
- new merchant detection

These features represent user-specific financial behavior rather than static thresholds.

### 6. Signal Engine (Ensemble Detection)

PFAD combines two approaches:

#### Isolation Forest

- detects multi-dimensional anomalies
- effective for irregular spending patterns

#### Z-score

- detects statistical deviations within spending patterns
- effective for magnitude spikes relative to baseline

#### Ensemble Logic

```text
Both agree -> critical
One agrees  -> warning
None        -> normal
```

This reduces false positives compared to a single-model detector.

### 7. Insight Engine

Instead of exposing only raw anomaly scores, PFAD generates structured insights.

Example:

```text
Food spending increased 4.6x relative to baseline
```

Types of insights include:

- behavioral drift
- category spikes
- recurring expense detection
- spending concentration
- volatility patterns

### 8. Action Layer

Each insight is converted into a structured recommendation:

```text
Problem:
Transport spending increased 54%

Cause:
High frequency short trips

Impact:
1200 INR per month additional spend

Action:
Limit ride frequency to 2 per day

Expected Outcome:
Stabilizes average discretionary spend
```

This layer differentiates PFAD from descriptive analytics dashboards.

### 9. Adaptive Financial Memory

User corrections are persisted locally:

```json
{
  "komal store": "groceries"
}
```

- applied before classification
- improves accuracy over time
- preserves personalization without cloud storage

### 10. Model Evaluation

PFAD includes internal evaluation surfaces such as:

- anomaly density
- average anomaly score
- precision
- recall
- false alert rate

For real datasets without labels, heuristic confidence and behavioral consistency checks are used instead.

## Evaluation Methodology

Metrics are evaluated on controlled synthetic datasets using:

- approximately 5% injected anomalies
- category-balanced sampling
- controlled variance distributions

This allows controlled evaluation in the absence of real-world anomaly labels.

---

## Example End-to-End Flow

### Input

```text
1000 INR received from Naveen Sharma
```

### Output

```text
Entity: Person
Category: Income Transfer
Insight: Excluded from spending analysis
```

### Input

```text
2147 INR at Zomato
```

### Output

```text
Category: Food
Deviation: Significant increase vs baseline
Cause: High recent category velocity
Action: Reduce discretionary food spending this week
```

## Example Insight

### Input

```text
4800 INR spent on food in 1 day
```

### Output

- Insight: Food spending increased 4.2x versus baseline
- Cause: High transaction frequency and weekend effect
- Impact: 1600 INR projected weekly overspend
- Action: Reduce discretionary food spend by 300 INR per day for the next 5 days

---

## Design Decisions

- **Unsupervised learning**: real transaction data rarely has labeled anomalies
- **Ensemble detection**: reduces variance and over-reliance on one signal
- **Local-first architecture**: improves privacy and avoids unnecessary external dependencies
- **Hybrid classification**: balances interpretability, control, and adaptability
- **Transfer suppression**: prevents person-to-person cash movement from polluting spend analytics

---

## Engineering Challenges

### 1. Unstructured Data Ingestion

Financial data often lacks consistent schema, especially in PDFs.

### 2. No Ground Truth Labels

Real-world financial data does not provide labeled anomalies.

### 3. High False Positive Risk

Over-alerting reduces trust. Ensemble detection and transfer suppression mitigate this.

### 4. Non-Stationary Behavior

User spending patterns shift over time, requiring adaptive baselines rather than fixed thresholds.

---

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

### Permanent Account Setup

PFAD supports both guest mode and permanent local accounts.

Generate a bcrypt password hash using the included utility:

```bash
python src/utils/hash_gen.py
```

Then add the user to `config.yaml`:

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
      password: "$2b$12$paste_hash_here..."
```

Important:

- log in with the `username` key such as `admin`
- do not log in with `name` or `email`
- store only hashed passwords in `config.yaml`

### Run

```bash
streamlit run app.py
```

## Docker Usage

PFAD can be run in a container for consistent local deployment.

### Option 1: Docker Compose (recommended)

Build and run:

```bash
docker compose up --build
```

Default access URL:

```text
http://localhost:8502
```

The compose setup maps:

- container `8501` -> host `8502`
- `./data` -> `/app/data` for persistent local app data
- `./config.yaml` -> `/app/config.yaml` for local authentication config

Stop services:

```bash
docker compose down
```

### Option 2: Docker CLI

Build image:

```bash
docker build -t pfad:latest .
```

Run container:

```bash
docker run --rm -p 8502:8501 -v ${PWD}/data:/app/data -v ${PWD}/config.yaml:/app/config.yaml pfad:latest
```

On Windows PowerShell, if `${PWD}` path expansion causes issues, use absolute paths:

```bash
docker run --rm -p 8502:8501 -v C:\path\to\repo\data:/app/data -v C:\path\to\repo\config.yaml:/app/config.yaml pfad:latest
```

### Notes

- The container starts Streamlit with `--server.address=0.0.0.0 --server.port=8501`
- `config.yaml` must exist on host before starting container
- `PFAD_TEST_DB` is set in compose to store SQLite data under `/app/data/pfad.db`

---

## Technology Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **ML**: Scikit-learn
- **Persistence**: SQLite + JSON
- **Parsing**: pdfplumber, openpyxl, custom parser layer

---

## Complexity

- Parsing: `O(n)`
- Feature computation: `O(n)`
- Isolation Forest: approximately `O(n log n)`
- End-to-end pipeline: near-linear for typical workloads

---

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

---

## Limitations

- Isolation Forest may still produce false positives on sparse datasets
- Z-score logic is sensitive to thin history and unstable variance
- PDF parsing depends on text-based extraction; OCR is not included
- Entity resolution remains heuristic for ambiguous counterparties
- Impact projections are advisory heuristics, not forecasts

## Failure Modes

- Person names that resemble business entities may be misclassified
- Sparse transaction history reduces statistical reliability
- Sudden lifestyle changes may temporarily trigger false positives
- Merchant normalization may fail on heavily obfuscated text inputs

---

## Key Takeaway

PFAD is not primarily a budgeting tool.

It is a system designed to:

```text
identify behavioral deviations
explain causal drivers
guide corrective financial decisions
```

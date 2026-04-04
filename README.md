# Personal Finance Anomaly Detector (PFAD)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-UI%2FUX-FF4B4B?style=for-the-badge&logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikitlearn)
![SQLite](https://img.shields.io/badge/SQLite-Persistence-003B57?style=for-the-badge&logo=sqlite)
![Coverage](https://img.shields.io/badge/Coverage-94%25%20(pytest--cov)-brightgreen?style=for-the-badge)

**PFAD** is a behavior-aware financial intelligence system — a machine-learning-powered financial operations center. Moving beyond rudimentary budgeting apps, PFAD acts as an autonomous, proactive financial assistant. It ingests your typical transaction history, learns your unique spending fingerprint, extracts causal drivers for unusual behavior, and creates definitive action plans to stabilize your Financial Health.

Built with a stunning, low-noise **Premium Fintech UI** (inspired by platforms like Stripe and Apple Wallet), PFAD surfaces intelligence over data density, allowing you to manage your personal finances with absolute clarity and peace of mind.

---

## 🌟 Core Features for Everyday Life

### 1. The Intelligence Action Layer
PFAD doesn't just display charts — it acts as a decision engine. Our new Intelligence Layer transforms raw analytics into definitive **Suggested Adjustments**. Instead of just telling you that you overspent on Transport, PFAD tells you exactly *how* to fix it, providing:

*   **Target:** `Reduce Transport by ₹306`
*   **Timeframe & Strategy:** `over the next 7 days — by halting impulse buys.`
*   **Impact Projection:** `Estimated Impact: +3 to +6 (approximate improvement)`
*   **Risk if Ignored:** `If this trend continues, you may exceed your monthly budget by ₹4,200.`


### 2. Dual-Signal Anomaly Detection & Causal Deconstruction
PFAD refuses to be a black box. It leverages an **Ensemble Reasoning Engine**—combining an Isolation Forest with rolling Z-score statistical bounds (30-day deviation)—to detect both multi-dimensional and volume-based outliers. Using custom feature attribution logic mapped by our core `explainer.py` engine, it assigns a High/Medium confidence rating to every flag, and deconstructs the anomaly into 4 components:
*   **Confidence Rating:** `High (Isolation Forest + Statistical agreement)`
*   **What Happened:** The actual impact of the transaction.
*   **Baseline & Deviation:** Proportional variance against rolling 30-day benchmarks.
*   **Cause:** The multi-factor driver (e.g. *Weekend Spike + New Vendor*).

### 3. Native Ground Truth Evaluation Engine
How do you know the ML system works? We built an onboard performance engine that evaluates itself. By injecting synthetic anomalies into the mock dataset with true labels, PFAD calculates rigorous ML metrics (**Precision**, **Recall**, and **F1-Score**) in real-time, proving its accuracy directly in the diagnostic panel. (When using real-world data without ground truth, it gracefully falls back to generating Model Confidence heuristics).

### 4. Native Multi-Format Ingestion Engine
Stop wrestling with generic CSV mapping or manual data entry. PFAD's robust `unified_parser` natively supports:
*   **Bank Exports**: Natively maps standard/messy headers (Mint, Chase) using fuzzy match resolution.
*   **Excel Spreadsheets**: Parses `.xlsx` and `.xls` files natively, intelligently identifying primary transaction sheets.
*   **PDF Statements & Receipts**: A layered 5-stage Regex pipeline processes messy Indian transaction formats — easily digesting PDFs from GPay, PhonePe, and structured bank statements with built-in noise reduction and confidence scoring.

### 5. Adaptive Financial Memory
To provide true personalization, PFAD includes a local `Adaptive Memory Subsystem`. If the default category classification (`Rule-Based` + `Machine Learning`) fails or misses a nuanced merchant, you can manually categorize it directly through the dashboard. The system persists this into `merchant_category_map.json`, ensuring the framework intelligently auto-applies your override mappings on all future uploads—learning specifically for you over time.

### 5. Smart Subscription tracking & Financial Health Score
Your financial pulse is quantified into a 0-100 score, embedded in a seamless dark-mode radial dashboard. Algorithms actively scout and categorize invisible subscriptions prioritizing structural changes to cut down overhead.

## Why This Matters

Most finance apps track spending.
PFAD explains behavior.

Instead of showing where money went,
it identifies what changed and how to correct it.

This enables proactive financial control rather than reactive tracking.

---

## 🛠️ Technology Stack & Architecture

PFAD is optimized for local execution—meaning it is fast, private, and deeply secure by design.

*   **Frontend Engine**: [Streamlit](https://streamlit.io/) with deep CSS/HTML injection for glass-metal cards, grid-flexbox architectures, dynamic flex margins, and responsive Plotly integrations.
*   **Compute & Analytics**: [Pandas](https://pandas.pydata.org/) and NumPy for instantaneous rolling window calculations (EWMAs) and sub-second matrix operations.
*   **Machine Learning State**: [Scikit-Learn](https://scikit-learn.org/) handling the `IsolationForest` pipeline and internal performance evaluation.
*   **Persistence Layer**: Native SQLite3 implementation via `src/database.py`, locking your metrics, goals, and logic locally so your data never hits the cloud.

### 📂 Directory Structure

```text
├── app.py                      # Core interactive Streamlit dashboard & UI Engine
├── requirements.txt            # Environment mappings
├── data/                       # Local store for sample data and adaptive memory (merchant_category_map.json)
├── src/                        # Core backend capabilities
│   ├── anomaly_detector.py     # Isolation Forest & Performance Evaluation Engine
│   ├── category_classifier.py  # Adaptive rule-based and ML categorization logic
│   ├── data_loader.py          # Fuzzy matching & column normalization
│   ├── database.py             # SQLite wrapper for rules, constraints & budgets
│   ├── explainer.py            # Structural causal deconstruction for anomalies
│   ├── feature_engine.py       # ML vector & feature extraction
│   ├── health_scorer.py        # Composite 0-100 logic processing
│   ├── insights.py             # Action Layer generator for predictive financial adjustments
│   ├── preprocessor.py         # Null handling and data sanitization
│   ├── parsers/                # Extensible parsing subsystem (CSV, Excel, PDF)
│   └── user_profiler.py        # Baseline behavioral mapping
└── tests/                      
    ├── test_parsers.py         # Multi-format parser logic integration tests
    ├── test_stress_suite.py    # Master end-to-end regression & stress suite
```

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and spin up an isolated virtual environment:
```bash
git clone https://github.com/Traveller-eng/personal-finance-anomaly-detector.git
cd personal-finance-anomaly-detector

python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Authentication Setup

PFAD uses local authentication for security.

1. Create a `config.yaml` file in the root directory
2. Add your credentials using hashed passwords
3. Restart the app

Example configuration is provided in `config_example.yaml`. You can quickly generate your own secure credentials natively by executing our onboard hashing utility:
```bash
python src/utils/hash_gen.py
```

> **Guest Mode**: PFAD supports a fully sandboxed "Quick Evaluation" Guest Mode. Click "Continue as Guest" at the login portal to bypass authentication and safely explore the engine in a Read-Only state where structural database saves are automatically disabled.

### 3. Launching the Hub
Execute the engine locally. All routing and data manipulation runs strictly on-device out of the box.
```bash
streamlit run app.py
```

> **Note:** If you run the app without uploading a file, an onboard autonomous generator will synthesize 1-year of hyper-realistic test transactions for immediate visualization—fully equipped with ground truth anomaly labels to demonstrate model precision!

---

## 🗺️ Feature Roadmap

- **Multi-currency support**: Expand architectural boundaries via native Exchange Rate API integrations to automatically normalize multi-national bank exports.

---

## Limitations

- Synthetic data may not fully represent real-world financial behavior
- Isolation Forest may produce false positives in sparse or irregular categories
- Z-score assumes normal distribution, which may not hold across all spending patterns
- Impact projections are heuristic and not predictive models
- Image-only (scanned) PDFs currently require external OCR (text-based PDF statements are fully supported natively)

---

## 🔒 Security & Privacy First
Financial data should never be arbitrary. 
PFAD is deliberately built without external API routing. All SQL tracking (`pfad.db`), model caching, budget configurations, and anomaly logs are maintained **strictly locally on your hard drive**.

## 🧪 Rigorous Testing
PFAD maintains high code coverage backed by a robust testing suite (`pytest tests/`). It enforces assertions simulating unexpected data inputs, verifying that the Action Layer constructs safely and anomalous pipelines fail gracefully under edge cases. 

---
### License
PFAD is open-source software, built with passion and provided freely under the **MIT License**. Elevate your standard of living today.

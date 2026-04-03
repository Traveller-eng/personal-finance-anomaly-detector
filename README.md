# Personal Finance Anomaly Detector (PFAD)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-UI%2FUX-FF4B4B?style=for-the-badge&logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikitlearn)
![SQLite](https://img.shields.io/badge/SQLite-Persistence-003B57?style=for-the-badge&logo=sqlite)
![Coverage](https://img.shields.io/badge/Coverage-94%25%20(pytest--cov)-brightgreen?style=for-the-badge)

**PFAD** is a behavior-aware financial intelligence system — a machine-learning-powered financial operations center. Moving beyond rudimentary budgeting apps, PFAD acts as an autonomous, proactive financial assistant It ingests your typical transaction history, learns your unique spending fingerprint, extracts causal drivers for unusual behavior, and creates definitive action plans to stabilize your Financial Health.

Built with a stunning, low-noise **Premium Fintech UI** (inspired by platforms like Stripe and Apple Wallet), PFAD surfaces intelligence over data density, allowing you to manage your personal finances with absolute clarity and peace of mind.

---

## 🌟 Core Features for Everyday Life

### 1. The Intelligence Action Layer
PFAD doesn't just display charts — it acts as a decision engine. Our new Intelligence Layer transforms raw analytics into definitive **Suggested Adjustments**. Instead of just telling you that you overspent on Transport, PFAD tells you exactly *how* to fix it, providing:

*   **Target:** `Reduce Transport by ₹306`
*   **Timeframe & Strategy:** `over the next 7 days — by halting impulse buys.`
*   **Impact Projection:** `Estimated Impact: +3 to +6 (approximate improvement)`
*   **Risk if Ignored:** `If this trend continues, you may exceed your monthly budget by ₹4,200.`


### 2. Deep Causal Explanations 
PFAD refuses to be a black box. Using an Isolation Forest model, PFAD identifies and explains unusual spending patterns. It deconstructs every flagged anomaly into 3 components:
*   **What Happened:** The actual impact of the transaction.
*   **Baseline & Deviation:** Proportional variance against rolling 30-day benchmarks.
*   **Cause:** The multi-factor driver (e.g. *Weekend Spike + New Vendor*).

### 3. Native Ground Truth Evaluation Engine
How do you know the ML system works? We built an onboard performance engine that evaluates itself. By injecting synthetic anomalies into the mock dataset with true labels, PFAD calculates rigorous ML metrics (**Precision**, **Recall**, and **F1-Score**) in real-time, proving its accuracy directly in the diagnostic panel. (When using real-world data without ground truth, it gracefully falls back to generating Model Confidence heuristics).

### 4. Automatic Bank Integrations
Stop wrestling with generic CSV mapping. PFAD natively parses exports from the industry's top platforms:
*   **Mint, Chase, and YNAB** formats are supported natively. Feed the tool your export file and the routing architecture parses merchant strings, timestamps, and classifications instantly.

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

PFAD is built to be relentlessly fast, infinitely scalable, and thoroughly secure.

*   **Frontend Engine**: [Streamlit](https://streamlit.io/) with deep CSS/HTML injection for glass-metal cards, grid-flexbox architectures, dynamic flex margins, and responsive Plotly integrations.
*   **Compute & Analytics**: [Pandas](https://pandas.pydata.org/) and NumPy for instantaneous rolling window calculations (EWMAs) and sub-second matrix operations.
*   **Machine Learning State**: [Scikit-Learn](https://scikit-learn.org/) handling the `IsolationForest` pipeline and internal performance evaluation.
*   **Persistence Layer**: Native SQLite3 implementation via `src/database.py`, locking your metrics, goals, and logic locally so your data never hits the cloud.

### 📂 Directory Structure

```text
├── app.py                      # Core interactive Streamlit dashboard & UI Engine
├── requirements.txt            # Environment mappings
├── src/                        # Core backend capabilities
│   ├── anomaly_detector.py     # Isolation Forest & Performance Evaluation Engine
│   ├── data_loader.py          # Native format parsing (Mint, Chase, YNAB)
│   ├── database.py             # SQLite wrapper for rules, constraints & budgets
│   ├── explainer.py            # Structural causal deconstruction for anomalies
│   ├── feature_engine.py       # ML vector & feature extraction
│   ├── health_scorer.py        # Composite 0-100 logic processing
│   ├── insights.py             # Action Layer generator for predictive financial adjustments
│   ├── preprocessor.py         # Null handling and data sanitization
│   └── user_profiler.py        # Baseline behavioral mapping
└── tests/                      
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

Example configuration is provided in `config_example.yaml`

### 3. Launching the Hub
Execute the engine locally. All routing and data manipulation runs strictly on-device out of the box.
```bash
streamlit run app.py
```

> **Note:** If you run the app without uploading a file, an onboard autonomous generator will synthesize 1-year of hyper-realistic test transactions for immediate visualization—fully equipped with ground truth anomaly labels to demonstrate model precision!

---

## Limitations

- Synthetic data may not fully represent real-world financial behavior
- Isolation Forest may produce false positives in sparse or irregular categories
- Z-score assumes normal distribution, which may not hold across all spending patterns
- Impact projections are heuristic and not predictive models
- No real-time bank API integration (CSV-based ingestion only)
- Single-currency assumption (no FX handling)

---

## 🔒 Security & Privacy First
Financial data should never be arbitrary. 
PFAD is deliberately built without external API routing. All SQL tracking (`pfad.db`), model caching, budget configurations, and anomaly logs are maintained **strictly locally on your hard drive**.

## 🧪 Comprehensive Stress Testing
PFAD runs behind an impenetrable 14-Module Testing Suite (`pytest tests/`). It enforces adversarial assertions simulating everything from catastrophic null-data inputs to missing databases, verifying that the Action Layer constructs safely and anomalous pipelines fail gracefully under extreme stress. 

---
### License
PFAD is open-source software, built with passion and provided freely under the **MIT License**. Elevate your standard of living today.

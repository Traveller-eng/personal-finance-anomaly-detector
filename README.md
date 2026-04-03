# Personal Finance Anomaly Detector (PFAD)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Framework-FF4B4B?style=for-the-badge&logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikitlearn)
![SQLite](https://img.shields.io/badge/SQLite-Persistence-003B57?style=for-the-badge&logo=sqlite)
![Coverage](https://img.shields.io/badge/Coverage-100%25-brightgreen?style=for-the-badge)

A premium, machine-learning-powered financial operations center designed to automatically monitor spending behavior, detect irregular transactions, and provide actionable insights. Moving far beyond traditional budgeting tools that simply categorize expenses, PFAD utilizes unsupervised machine learning (Isolation Forest) paired with deterministic behavioral algorithms to learn your unique spending fingerprint, track recurring overhead, and alert you strictly when significant deviations occur.

## 🌟 Key Features

*   **Stateful Feedback Loop:** Don't agree with an anomaly? A single click ("Mark as Expected") caches merchant allowances into a local **SQLite database**, suppressing false positives instantly on all future scans without retraining headaches.
*   **Intelligent Anomaly Detection:** Uses an Unsupervised Isolation Forest model equipped with Model Snapshots. Once your baseline is learned, the ML execution state is cached to disk, meaning lightning-fast reload times across sessions.
*   **Native Bank Integrations:** Forget manual spreadsheet wrangling. PFAD natively parses transaction formats out-of-the-box from **Mint, Chase, and YNAB**, powered by an autonomous matching router.
*   **Recurring Overhead Tracking:** Engineered algorithms scan for monthly-periodicity and tight spending-variance to intelligently track hidden subscriptions and predict fixed-cost overheads without manual tagging.
*   **Actionable Insights & Budgets:** Tell PFAD your monthly constraints. Our Behavioral Insights engine cross-verifies live burn rates to fire off warnings when you surpass projected category budgets ahead of schedule.
*   **Premium Interactive UI:** Built entirely on Streamlit utilizing custom CSS, glassmorphism elements, dynamic Month-Over-Month delta tables, and Plotly interactive data graphics.

## 🛠️ Technology Architecture

*   **Frontend / UI:** [Streamlit](https://streamlit.io/) + Custom Behavioral CSS Framework
*   **State & Persistence:** SQLite3 (Local `pfad.db` for budgets, feedback, & ML Model Blobs)
*   **Machine Learning:** [Scikit-Learn](https://scikit-learn.org/) (Isolation Forest)
*   **Data Pipelines:** [Pandas](https://pandas.pydata.org/), NumPy
*   **Testing Infrastructure:** Exhaustive, 14-module Pytest Suite integrating adversarial stress tests and isolated DB fixtures.

## 📂 Project Structure

```text
├── app.py                      # Core interactive Streamlit dashboard
├── requirements.txt            # Python dependencies
├── src/                        # Core backend modules
│   ├── anomaly_detector.py     # Isolation Forest & SQLite Model Snapshots
│   ├── data_loader.py          # Native Mint, Chase, YNAB, Generic parsing routes
│   ├── database.py             # SQLite wrapper for rules, constraints, & states
│   ├── explainer.py            # Natural Language generation for anomalies
│   ├── feature_engine.py       # ML vector engineering
│   ├── health_scorer.py        # Composite 0-100 behavior scoring
│   ├── insights.py             # Live burn-rate tracking & behavioral warnings
│   ├── preprocessor.py         # Null/Garbage cleansing
│   ├── recurring.py            # Periodicity tracking for subscriptions
│   └── user_profiler.py        # Baseline behavioral mapping
└── tests/                      # 14-Module Testing Infrastructure
    ├── test_stress_suite.py    # Master end-to-end regression & stress suite
    ...
```

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Traveller-eng/personal-finance-anomaly-detector.git
cd personal-finance-anomaly-detector
```

### 2. Install Dependencies
It is highly recommended to isolate environments utilizing `venv`:
```bash
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Launch the Hub
The local UI fires securely via standard Streamlit execution:
```bash
streamlit run app.py
```

*Note: If no custom file is provided, an onboard algorithmic generator simulates 1-year of hyper-realistic test transactions completely locally.*

## 🔒 Your Data is Private
All SQL tracking (`pfad.db`), model caching, and execution traces remain **strictly local**. Nothing leaves your machine. Your budget constraints, anomaly dismissals, and financial metrics are scoped securely onto your own file system.

## 🧪 Testing Parity
PFAD is heavily fortified against nulls, data corruption, and adversarial inputs. The core suite spans 40 explicit assertions mapping across Performance Benchmarks, Data Loader pipelines, and isolated SQLite testing environments.
```bash
pytest tests/ -v
```

## 📄 License
This project is open-source and available under the terms of the **MIT License**. Build, modify, and improve freely.

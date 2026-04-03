# Personal Finance Anomaly Detector (PFAD)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-UI%2FUX-FF4B4B?style=for-the-badge&logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikitlearn)
![SQLite](https://img.shields.io/badge/SQLite-Persistence-003B57?style=for-the-badge&logo=sqlite)
![Coverage](https://img.shields.io/badge/Test%20Coverage-14%20Modules-brightgreen?style=for-the-badge)

**PFAD** is a state-of-the-art, machine-learning-powered financial operations center. Moving beyond rudimentary budgeting apps, PFAD acts as a proactive financial assistant. It ingests your typical transaction history, learns your unique spending fingerprint, and automatically alerts you of unusual behaviors, hidden subscriptions, or structural deviations in your burn rate.

Built with a stunning, low-noise **Premium Fintech UI** (inspired by platforms like Stripe and Apple Wallet), PFAD surfaces intelligence over data density, allowing you to manage your personal finances with absolute clarity and peace of mind.

---

## 🌟 Core Features for Everyday Life

### 1. Intelligent Anomaly Detection ("Calm Intelligence")
PFAD doesn't scream at you for buying coffee. Using an **Unsupervised Isolation Forest** algorithm, it maps out a multi-dimensional baseline of your spending. 
*   **Deviation Mapping:** It scans for combinations of anomalies (e.g., spending 30% more than usual on Transport specifically on a Sunday).
*   **Feedback Loops:** If the system flags something you expect, a single click ("Mark as Expected") stores the rule in a local **SQLite database**, suppressing future false positives instantly.

### 2. Automatic Bank Integrations
Stop wrestling with generic CSV mapping. PFAD natively parses exports from the industry's top platforms:
*   **Mint, Chase, and YNAB** formats are supported natively. Simply feed the tool your export file and the routing architecture parses merchant strings, timestamps, and classifications instantly.

### 3. Financial Health & "What Changed" Index
Your financial pulse is quantified into a 0-100 core score, rendered in a dynamic semantic radial UI. 
*   Rather than making you dig through charts, the **What Changed Dashboard** summarizes exactly *why* your score moved (e.g., *"Wait, you've spent ₹852 more on Shopping this week than your 30-day average."*).

### 4. Smart Subscription & Overhead Tracking
Leveraging complex algorithmic periodicity detection (tracking 30-day payment rhythms alongside low variance coefficients), PFAD actively scouts and categorizes invisible subscriptions that are secretly inflating your monthly overhead, giving you ultimate control over fixed costs.

### 5. Premium "Dark-Mode" Architecture
Aesthetically tuned to output maximum contrast and minimal noise. Gone are flashy emojis and unreadable charts. You get monochrome Plotly graphs, muted charcoal backgrounds (`#0E1117`), and targeted amber/ruby glow effects focusing your eyes exclusively on what requires attention. 

---

## 🛠️ Technology Stack & Architecture

PFAD is built to be relentlessly fast, infinitely scalable, and thoroughly secure.

*   **Frontend Engine**: [Streamlit](https://streamlit.io/) with deep CSS/HTML injection for glass-metal cards, grid-flexbox architectures, and micro-interactions.
*   **Compute & Analytics**: [Pandas](https://pandas.pydata.org/) and NumPy for instantaneous rolling window calculations (EWMAs) and sub-second matrix operations.
*   **Machine Learning State**: [Scikit-Learn](https://scikit-learn.org/) handling the `IsolationForest`, leveraging SQLite-backed Model Snapshots so your custom models persist locally across sessions.
*   **Persistence Layer**: Native SQLite3 implementation via `src/database.py`, locking your metrics, goals, and logic locally so your data never hits the cloud.

### 📂 Directory Structure

```text
├── app.py                      # Core interactive Streamlit dashboard & UI Engine
├── requirements.txt            # Environment mappings
├── src/                        # Core backend capabilities
│   ├── anomaly_detector.py     # Isolation Forest & SQLite Model Snapshots
│   ├── data_loader.py          # Native format parsing (Mint, Chase, YNAB)
│   ├── database.py             # SQLite wrapper for rules, constraints & budgets
│   ├── explainer.py            # Natural Language generation for anomalies
│   ├── feature_engine.py       # ML vector & feature extraction
│   ├── health_scorer.py        # Composite 0-100 logic processing
│   ├── insights.py             # Live burn-rate tracking & behavioral warnings
│   ├── preprocessor.py         # Null handling and data sanitization
│   ├── recurring.py            # Periodicity tracking for subscriptions
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

### 2. Launching the Hub
Execute the engine locally. All routing and data manipulation runs strictly on-device out of the box.
```bash
streamlit run app.py
```

> **Note:** If you run the app without uploading a file, an onboard autonomous generator will synthesize 1-year of hyper-realistic test transactions for immediate visualization.

---

## 🔒 Security & Privacy First
Financial data should never be arbitrary. 
PFAD is deliberately built without external API routing. All SQL tracking (`pfad.db`), model caching, budget configurations, and anomaly logs are maintained **strictly locally on your hard drive**.

## 🧪 Comprehensive Stress Testing
PFAD runs behind an impenetrable 14-Module Testing Suite (`pytest tests/test_stress_suite.py`). It enforces **40+ adversarial assertions** simulating everything from catastrophic null-data inputs to missing databases, verifying that the engine fails gracefully or handles anomalies smoothly under stress. 

---
### License
PFAD is open-source software, built with passion and provided freely under the **MIT License**. Elevate your standard of living today.

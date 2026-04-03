# Personal Finance Anomaly Detector (PFAD)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Framework-FF4B4B)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-F7931E)
![Plotly](https://img.shields.io/badge/Plotly-Data%20Visualization-3F4F75)
![License](https://img.shields.io/badge/License-MIT-green)

A premium, machine-learning-powered financial dashboard designed to automatically monitor spending behavior, detect irregular transactions, and provide actionable insights. Moving beyond traditional budgeting tools that simply categorize expenses, PFAD utilizes unsupervised machine learning (Isolation Forest) to learn your unique spending fingerprint and alert you exclusively when significant behavioral deviations occur.

## 🌟 Key Features

*   **Intelligent Anomaly Detection:** Uses an Unsupervised Isolation Forest model to detect multi-dimensional anomalies (e.g., amount, category timing, merchant novelty) without relying on rigid, pre-defined rules.
*   **Structured Explanations:** Explains *why* a transaction was flagged by explicitly breaking down your "Typical Behavior" vs. the "Deviation", avoiding confusing black-box ML outputs.
*   **Actionable Insights Engine:** Scans your spending profile to generate plain-text advice, flagging issues like "Upcoming Subscription Spikes," "Category Budget Risks," and "Weekend Spending Surges."
*   **Smart Trends & EWMA Forecasting:** Visualizes weekly and monthly trends, utilizing Exponentially Weighted Moving Averages (EWMA) to forecast upcoming 7-day spending trajectories with resilience against single-day outliers.
*   **Financial Health Scoring:** Calculates a composite 0-100 score evaluating budget adherence, spending consistency, anomaly frequencies, and transaction necessity.
*   **Premium Interactive UI:** Built on Streamlit and entirely redesigned with custom CSS and Plotly graphics, offering a sleek, dark-navy fintech aesthetic modeled after modern high-end banking apps.

## 🛠️ Technology Stack

*   **Frontend / UI:** [Streamlit](https://streamlit.io/) + Custom CSS Injection
*   **Data Visualization:** [Plotly Express / Graph Objects](https://plotly.com/python/)
*   **Machine Learning:** [Scikit-Learn](https://scikit-learn.org/) (Isolation Forest)
*   **Data Processing:** [Pandas](https://pandas.pydata.org/), NumPy
*   **Testing:** Pytest

## 📂 Project Structure

```text
├── app.py                      # Main Streamlit dashboard and UI routing
├── generate_sample_data.py     # Script to generate realistic dummy transactions
├── requirements.txt            # Python dependencies
├── src/                        # Core backend modules
│   ├── anomaly_detector.py     # Isolation Forest ML implementation
│   ├── data_loader.py          # CSV parsing and validation logic
│   ├── explainer.py            # Reason-generation for flagged transactions
│   ├── feature_engine.py       # Time-series and categorical feature extraction
│   ├── health_scorer.py        # Composite 0-100 grade scoring algorithms
│   ├── insights.py             # Behavioral insight & advice generator
│   ├── preprocessor.py         # Data cleaning and missing value handling
│   └── user_profiler.py        # Spends calculation (averages, distributions)
└── tests/                      # Unit tests for ML and data components
    ├── test_anomaly_detector.py
    ├── test_explainer.py
    └── test_preprocessor.py
```

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Traveller-eng/personal-finance-anomaly-detector.git
cd personal-finance-anomaly-detector
```

### 2. Install Dependencies
It is recommended to use a virtual environment:
```bash
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the Dashboard
Launch the interface locally via Streamlit:
```bash
streamlit run app.py
```
*Note: A realistic 1-year sample dataset will automatically generate if you do not upload your own CSV.*

## 📊 Using Your Own Data

You can upload your own bank statements directly through the dashboard sidebar. Your CSV file must contain the following required columns (case-insensitive mapping supported):
*   `date`: Transaction date (e.g., YYYY-MM-DD)
*   `amount`: Transaction value (numeric)
*   `category`: Type of expense (e.g., Food, Transport, Bills)
*   `merchant`: Name of the vendor

## 🧪 Running Tests
The project boasts a suite of unit tests to verify the integrity of the data processors and the anomaly detector. To run the test suite:
```bash
pytest tests/ -v
```

## 📄 License
This project is open-source and available under the terms of the MIT License.

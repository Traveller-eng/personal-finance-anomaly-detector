"""
app.py — Personal Finance Anomaly Detector — Streamlit Dashboard
=================================================================
PURPOSE:
    The front-end of the system. Connects all backend modules into an
    interactive, visual experience. Built with Streamlit for rapid iteration
    and Python-native UI composition.

    5 Pages:
    1. Overview    — KPIs, totals, anomaly summary
    2. Anomalies   — Detailed anomaly explorer with explanations
    3. Trends      — Spending over time (daily, weekly, monthly)
    4. Profile     — Behavioral fingerprint visualization
    5. Health      — Financial health score breakdown + recommendations

PIPELINE (Every page shares this):
    Upload (CSV/Excel/PDF) → parse → map columns → preprocess →
    classify categories → feature engineer → profile → anomaly detect →
    explain → health score → insights → display
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import sys
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

# Add project root to path so src/ imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_from_dataframe, load_csv
from src.category_classifier import CATEGORIES, classify_categories, get_classification_summary
from src.parsers import parse_file, ParseResult
from src.parsers.unified_parser import get_supported_extensions
from src.preprocessor import clean_data
from src.feature_engine import engineer_features
from src.user_profiler import UserProfile
from src.anomaly_detector import AnomalyDetector
from src.explainer import explain_anomalies
from src.health_scorer import HealthScorer
from src.insights import InsightGenerator
from src.database import add_expected_transaction, set_budget, get_budgets

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="PFAD — Personal Finance Anomaly Detector",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Core Structural */
.stApp { background-color: #0E1117; }
[data-testid="stSidebar"] { background-color: #0E1117; opacity: 0.85; border-right: 1px solid rgba(255,255,255,0.05); }

/* Typography */
* { letter-spacing: 0.3px; }
h1, h2, h3, h4, .section-header { font-weight: 500; color: #F3F4F6; }
.section-header { font-size: 1.1rem; padding-bottom: 0.5rem; margin-bottom: 32px; border-bottom: 1px solid rgba(255,255,255,0.04); }
.text-muted { color: #9CA3AF; font-size: 0.9rem; }
.fintech-score { font-size: 3rem; font-weight: 600; color: #F3F4F6; text-shadow: 0 2px 10px rgba(0,0,0,0.5); }

/* The New Card System */
.kpi-card, .anomaly-card, .insight-panel, .card {
    background: linear-gradient(145deg, #18191E, #0E1117);
    border-radius: 16px;
    padding: 20px;
    border: 1px solid rgba(255,255,255,0.04);
    box-shadow: 0 4px 20px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.03);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    margin-bottom: 1rem;
}
.kpi-card:hover, .anomaly-card:hover, .insight-panel:hover, .card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.05);
}

/* Hero Score Ring */
.score-ring {
    width: 220px;
    height: 220px;
    margin: 0;
    border-radius: 50%;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    position: relative;
    box-shadow: inset 0 4px 20px rgba(0,0,0,0.5);
    background: radial-gradient(circle, #0E1117 50%, transparent 60%);
}
.score-ring::before {
    content: "";
    position: absolute;
    inset: -2px;
    border-radius: 50%;
    z-index: -1;
    background: conic-gradient(#34D399 0% var(--score-pct, 0%), #1F2937 var(--score-pct, 0%));
}
.score-caption { font-size: 0.85rem; color: #9CA3AF; margin-top: 5px; }

/* Anomaly Alert Cards */
.anomaly-critical { border-left: 4px solid #F87171; box-shadow: 0 0 0 1px rgba(248,113,113,0.1), 0 4px 15px rgba(0,0,0,0.4); }
.anomaly-warning { border-left: 4px solid #FBBF24; box-shadow: 0 0 0 1px rgba(251,191,36,0.1), 0 4px 15px rgba(0,0,0,0.4); }
.anomaly-mild { border-left: 4px solid #6B7280; box-shadow: 0 0 0 1px rgba(107,114,128,0.1), 0 4px 15px rgba(0,0,0,0.4); }

.anomaly-header { display: flex; align-items: center; justify-content: flex-start; gap: 12px; margin-bottom: 0.8rem; flex-wrap: wrap; }
.anomaly-title { font-weight: 500; font-size: 1rem; color: #E5E7EB; }
.anomaly-amount { font-weight: 600; font-size: 1.05rem; color: #E5E7EB; background: rgba(255,255,255,0.05); padding: 4px 8px; border-radius: 6px; }

.anomaly-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 12px;
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(255,255,255,0.03);
}
.grid-item { display: flex; flex-direction: column; gap: 4px; }
.grid-label { font-size: 0.75rem; color: #9CA3AF; text-transform: uppercase; letter-spacing: 0.3px; }
.grid-val { font-size: 0.9rem; font-weight: 500; color: #E5E7EB; }

.anomaly-reason {
    margin-top: 1rem;
    font-size: 0.85rem;
    color: #9CA3AF;
    line-height: 1.6;
}

/* What Changed Insights */
.insight-panel-title { font-weight: 500; font-size: 1rem; color: #E5E7EB; margin-bottom: 12px; display: flex; align-items: center; gap: 8px; }
.insight-listItem { margin-bottom: 8px; display: flex; align-items: start; gap: 8px; font-size: 0.85rem; color: #E5E7EB; }
.insight-bullet { color: #34D399; font-weight: bold; }

/* Subtext & Tones */
.subtle-emphasis { color: #34D399; font-weight: 500; }
.alert-text { color: #FBBF24; font-weight: 500; }

div[data-testid="stMetricValue"] { font-size: 1.4rem !important; font-weight: 600; color: #E5E7EB; }
</style>
""", unsafe_allow_html=True)


# ─── Pipeline Runner (cached) ─────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_pipeline(df_raw: pd.DataFrame, currency_symbol: str, contamination: float):
    """
    Full ML pipeline: load → clean → classify → engineer → profile → detect → explain → score.
    Cached so rerunning the same data doesn't repeat computation.
    Returns (df, profile, scorer, insights, detector, all_warnings)
    """
    all_warnings = []
    # Cache buster for new data model separation
    df_raw = df_raw.copy()

    # 1. Load + validate (schema-adaptive, never crashes)
    df, load_warnings = load_from_dataframe(df_raw.copy())
    all_warnings.extend(load_warnings)

    # 2. Clean & preprocess into strict schema
    df = clean_data(df)

    # 3. Auto-classify categories (hybrid: memory + rules + ML)
    df, class_warnings = classify_categories(df)
    all_warnings.extend(class_warnings)

    # 4. Feature engineering
    df = engineer_features(df)

    # 5. User profiling
    profile = UserProfile(df, currency_symbol=currency_symbol)

    # 6. Anomaly detection
    detector = AnomalyDetector(config={"contamination": contamination})
    df = detector.fit_predict(df)

    # 7. Explanation generation
    df = explain_anomalies(df, profile)

    # 8. Health scoring
    scorer = HealthScorer(df, profile)

    # 9. Insights
    insights = InsightGenerator(df, profile, currency_symbol=currency_symbol)

    return df, profile, scorer, insights, detector, all_warnings


# ─── Sidebar ─────────────────────────────────────────────────────────────────
def render_sidebar() -> tuple:
    """Render sidebar with upload, settings, and navigation."""
    with st.sidebar:
        st.markdown('<div class="sidebar-logo">💳 PFAD</div>', unsafe_allow_html=True)
        st.markdown("**Personal Finance Anomaly Detector**")
        st.divider()

        # ── Data Source
        st.markdown("### 📂 Data Source")
        data_source = st.radio(
            "Choose data",
            ["Use Sample Data", "Upload Your File"],
            index=0,
            label_visibility="collapsed",
        )

        uploaded_file = None
        if data_source == "Upload Your File":
            uploaded_file = st.file_uploader(
                "Upload Transaction File",
                type=get_supported_extensions(),
                help="Supports CSV, Excel (.xlsx), and PDF (GPay, PhonePe, bank statements)",
            )
            st.caption("📌 **Smart parsing**: CSV, Excel, and PDF files are auto-detected and parsed")
            st.caption("🔄 Columns like `amount (inr)`, `party`, `txn date` are auto-mapped")

        st.divider()

        # ── Settings
        st.markdown("### ⚙️ Settings")

        currency_options = {
            "₹ Indian Rupee": "₹",
            "$ US Dollar": "$",
            "€ Euro": "€",
            "£ British Pound": "£",
        }
        currency_label = st.selectbox("Currency", list(currency_options.keys()), index=0)
        currency_symbol = currency_options[currency_label]

        contamination_pct = st.slider(
            "Anomaly Sensitivity (%)",
            min_value=1,
            max_value=15,
            value=5,
            step=1,
            help="Expected fraction of anomalies. Higher = more transactions flagged.",
        )
        contamination = contamination_pct / 100.0

        # ── Budget Configuration
        st.markdown("### 🎯 Goals & Budgets")
        with st.expander("Set Monthly Budgets"):
            with st.form("budget_form"):
                cat_input = st.text_input("Category (e.g. Food, Transport)")
                limit_input = st.number_input("Monthly Limit", min_value=0.0, step=100.0)
                if st.form_submit_button("Save Budget"):
                    if cat_input:
                        if st.session_state.get("guest_mode", False):
                            st.warning(f"Guest Mode Active: Database saves are disabled.")
                        else:
                            set_budget(cat_input, limit_input)
                            st.success(f"Saved {cat_input} limit: {currency_symbol}{limit_input:,.0f}")
                        
        st.divider()

        # ── Navigation
        st.markdown("### 🗂️ Navigation")
        page = st.radio(
            "Go to",
            ["📊 Overview", "🔴 Anomalies", "📈 Trends", "👤 Profile", "💚 Health"],
            index=0,
            label_visibility="collapsed",
        )

        st.divider()
        st.caption("Built with Isolation Forest + Streamlit")

    return page, uploaded_file, currency_symbol, contamination


# ─── Data Loading ─────────────────────────────────────────────────────────────
def load_data(uploaded_file, currency_symbol, contamination):
    """
    Load data from uploaded file (any format) or sample data.
    Returns (raw_dataframe, parser_warnings).
    """
    sample_path = os.path.join(os.path.dirname(__file__), "data", "transactions.csv")
    parser_warnings = []

    # Generate sample data if it doesn't exist
    if not os.path.exists(sample_path) and uploaded_file is None:
        with st.spinner("Generating sample transaction data..."):
            from generate_sample_data import generate_dataset, save_dataset
            df_gen = generate_dataset()
            save_dataset(df_gen)

    # Load data via unified parser or direct CSV
    if uploaded_file is not None:
        with st.spinner(f"📄 Parsing {uploaded_file.name}..."):
            parse_result = parse_file(uploaded_file)
            parser_warnings.extend(parse_result.warnings)

            if not parse_result.success or len(parse_result.df) == 0:
                st.error("❌ Could not extract transactions from the uploaded file.")
                with st.expander("Parser Details", expanded=True):
                    for w in parse_result.warnings:
                        st.write(w)
                st.info(
                    "**Supported formats:** CSV, Excel (.xlsx), PDF (GPay, PhonePe, bank statements)\n\n"
                    "Ensure your file has at least date and amount data."
                )
                st.stop()

            df_raw = parse_result.df
            df_raw["parse_confidence"] = parse_result.parse_confidence
    else:
        if not os.path.exists(sample_path):
            st.error("Sample data not found. Please generate it or upload a file.")
            st.stop()
        try:
            df_raw = pd.read_csv(sample_path, encoding="utf-8")
        except UnicodeDecodeError:
            df_raw = pd.read_csv(sample_path, encoding="latin-1")
        df_raw["parse_confidence"] = 1.0

    return df_raw, parser_warnings


# ─── Helper Functions ─────────────────────────────────────────────────────────
def fmt(amount: float, symbol: str) -> str:
    """Format currency amount."""
    if amount is None or (isinstance(amount, (float, np.floating)) and np.isnan(amount)):
        return f"{symbol}0"
    return f"{symbol}{amount:,.0f}"


def plot_theme() -> dict:
    """Standard Plotly dark theme config."""
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#9CA3AF", family="Inter"),
        xaxis=dict(gridcolor="#1F2937", showgrid=True, zeroline=False),
        yaxis=dict(gridcolor="#1F2937", showgrid=True, zeroline=False),
        margin=dict(l=10, r=10, t=40, b=10),
    )


def behavioral_spend_view(df: pd.DataFrame) -> pd.DataFrame:
    """Return the debit-only, noise-filtered spending view used for behavior metrics."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=df.columns if df is not None else [])

    filtered = df.copy()
    if "type" in filtered.columns:
        filtered = filtered[filtered["type"].astype(str).str.lower() == "debit"]
    if "is_transfer" in filtered.columns:
        filtered = filtered[~filtered["is_transfer"].fillna(False).astype(bool)]
    if "category" in filtered.columns:
        filtered = filtered[~filtered["category"].astype(str).str.lower().isin(["income_transfer", "personal_transfer"])]
    return filtered


def render_behavior_insight_cards(insights: list[dict], limit: int = 3):
    """Render structured top insights with problem/cause/impact/action layout."""
    shown = insights[:limit]
    if not shown:
        st.info("No significant behavioral changes detected in the current period.")
        return

    for insight in shown:
        priority_color = "#F87171" if insight["priority"] == "critical" else ("#FBBF24" if insight["priority"] == "high" else "#94A3B8")
        st.markdown(
            f"""
<div class="card" style="border-left: 4px solid {priority_color};">
    <div style="display:flex; justify-content:space-between; align-items:center; gap:12px; margin-bottom:12px;">
        <div style="font-weight:600; font-size:1rem; color:#F8FAFC;">{insight['title']}</div>
        <div style="font-size:0.75rem; color:{priority_color}; text-transform:uppercase; letter-spacing:1px;">{insight['priority']}</div>
    </div>
    <div style="color:#E5E7EB; margin-bottom:8px;"><strong>What changed</strong><br>{insight['what_changed']}</div>
    <div style="color:#CBD5E1; margin-bottom:8px;"><strong>Why</strong><br>{insight['cause']}</div>
    <div style="color:#CBD5E1; margin-bottom:8px;"><strong>Impact</strong><br>{insight['impact']}</div>
    <div style="color:#E5E7EB; margin-bottom:8px;"><strong>Action</strong><br>{insight['action']}</div>
    <div style="color:#34D399;"><strong>Expected gain</strong><br>{insight['expected_gain']}</div>
</div>
            """,
            unsafe_allow_html=True,
        )


CATEGORY_COLORS = {
    "food": "#F87171",
    "shopping": "#60A5FA",
    "rent": "#A78BFA",
    "transport": "#34D399",
    "bills": "#FBBF24",
    "entertainment": "#F472B6",
    "health": "#22C55E",
    "education": "#93C5FD",
    "investment": "#38BDF8",
    "income_transfer": "#10B981",
    "personal_transfer": "#94A3B8",
    "others": "#64748B",
    "uncategorized": "#9CA3AF",
}

SEVERITY_COLORS = {
    "critical": "#F87171",
    "warning": "#FBBF24",
    "normal": "#334155",
}


# ─── PAGE 1: Overview ─────────────────────────────────────────────────────────
def page_overview(df: pd.DataFrame, profile: UserProfile, scorer: HealthScorer, insights_gen, currency: str):
    st.markdown("## 📊 Overview")
    st.divider()
    spend_df = behavioral_spend_view(df)
    health_score = scorer.total_score
    grade = scorer._get_grade()

    hero_col_left, hero_col_right = st.columns([1, 1.2], gap="large")

    with hero_col_left:
        st.markdown("<h3 style='margin-bottom: 1.5rem;'>Financial Health Index</h3>", unsafe_allow_html=True)
        color = "#34D399" if health_score >= 75 else ("#FBBF24" if health_score >= 50 else "#F87171")

        st.markdown(
            f"""
        <div style="
            width: 260px; height: 260px; margin: 0 auto;
            border-radius: 50%;
            display: flex; flex-direction: column; justify-content: center; align-items: center;
            background: conic-gradient({color} {health_score}%, #1F2937 {health_score}%);
            padding: 4px; box-shadow: inset 0 4px 15px rgba(0,0,0,0.5);
        ">
            <div style="background: #0E1117; width: 100%; height: 100%; border-radius: 50%; display: flex; flex-direction: column; justify-content: center; align-items: center; box-shadow: inset 0 4px 20px rgba(0,0,0,0.5);">
                <div class="fintech-score" style="color: {color};">{health_score:.0f}</div>
                <div class="score-caption" style="margin-top: 2px;">Out of 100 ({grade})</div>
            </div>
        </div>
            """,
            unsafe_allow_html=True,
        )

        top_insight = insights_gen.get_top_insights(1)
        if top_insight:
            st.markdown(
                f'<div style="text-align:center; margin-top:15px;"><span class="subtle-emphasis">{top_insight[0]["title"]}</span><br><span style="color:#E5E7EB; font-size:0.95rem;">{top_insight[0]["action"]}</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="text-align:center; margin-top:15px; color:#9CA3AF; font-size:0.95rem;">Spending patterns are stable.</div>',
                unsafe_allow_html=True,
            )

    with hero_col_right:
        st.markdown("<h3 style='margin-bottom: 1.5rem;'>Top Behavioral Insights</h3>", unsafe_allow_html=True)
        render_behavior_insight_cards(insights_gen.get_insights(), limit=3)

    anomalies = df[df["is_anomaly"] == 1]
    if len(anomalies) > 0:
        st.markdown('<p class="section-header" style="margin-top: 3.5rem;">Requires Attention</p>', unsafe_allow_html=True)
        recent_anomalies = anomalies.sort_values(by=["anomaly_score", "date"], ascending=[True, False]).head(5)

        cards_html = []
        for _, row in recent_anomalies.iterrows():
            sev = row.get("anomaly_severity", "warning").lower()
            if sev == "critical":
                sev_class = "anomaly-critical"
                icon = "Requires Attention"
            elif sev == "warning":
                sev_class = "anomaly-warning"
                icon = "Unusual Activity"
            else:
                sev_class = "anomaly-mild"
                icon = "Notable Change"

            struct_exp = row.get("structured_explanation", {})
            if isinstance(struct_exp, dict) and struct_exp:
                typical = struct_exp.get("baseline", "N/A")
                deviation = struct_exp.get("deviation", "N/A")
                reason = struct_exp.get("cause", "Multi-factor anomaly")
            else:
                typical = "N/A"
                deviation = "N/A"
                reason = row.get("explanation", "")

            cards_html.append(
                f"""<div class="anomaly-card {sev_class}">
    <div class="anomaly-header">
        <div class="anomaly-title"><span style="color:#9CA3AF; font-size:0.85rem; font-weight:500;">{icon} &nbsp;·&nbsp;</span> {str(row.get('category', '')).title()} at {row.get('merchant', 'Unknown')}</div>
        <div class="anomaly-amount">{fmt(row['amount'], currency)}</div>
    </div>
    <div style="color:#9CA3AF; font-size:0.8rem; margin-top:-8px; margin-bottom:8px;">{pd.to_datetime(row['date']).strftime('%b %d, %Y')}</div>
    <div class="anomaly-grid">
        <div class="grid-item">
            <span class="grid-label">Typical</span>
            <span class="grid-val">{typical}</span>
        </div>
        <div class="grid-item">
            <span class="grid-label">Observed</span>
            <span class="grid-val alert-text">{deviation}</span>
        </div>
    </div>
    <div class="anomaly-reason" style="margin-top:1rem; padding-top:0.8rem; border-top:1px solid rgba(255,255,255,0.05);">
        <strong>Reason:</strong> {reason}
    </div>
</div>"""
            )

        st.markdown("".join(cards_html), unsafe_allow_html=True)

    col_trends, col_donut, col_cashflow = st.columns([1.4, 1, 1])

    with col_trends:
        st.markdown('<p class="section-header" style="margin-top: 1rem;">📈 Spending Trend</p>', unsafe_allow_html=True)
        if len(spend_df) > 0:
            daily_spend = spend_df.groupby("date")["amount"].sum().reset_index()
            daily_spend = daily_spend.sort_values("date").set_index("date")
            full_idx = pd.date_range(daily_spend.index.min(), daily_spend.index.max())
            daily_spend = daily_spend.reindex(full_idx, fill_value=0).reset_index()
            daily_spend.columns = ["date", "amount"]
            daily_spend["ewma"] = daily_spend["amount"].ewm(alpha=0.25, adjust=False).mean()

            fig_trend = go.Figure()
            fig_trend.add_trace(go.Bar(
                x=daily_spend["date"],
                y=daily_spend["amount"],
                name="Daily Spend",
                marker_color="rgba(148, 163, 184, 0.45)",
                hovertemplate=f"<b>%{{x}}</b><br>{currency}%{{y:,.0f}}<extra></extra>",
            ))
            fig_trend.add_trace(go.Scatter(
                x=daily_spend["date"],
                y=daily_spend["ewma"],
                name="Smoothed Trend",
                line=dict(color="#34D399", width=3),
                mode="lines",
                hovertemplate=f"<b>%{{x}}</b><br>Trend: {currency}%{{y:,.0f}}<extra></extra>",
            ))
            fig_trend.update_layout(height=400, legend=dict(orientation="h", y=1.08), **plot_theme())
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("No spending transactions available after transfer filtering.")

    with col_donut:
        st.markdown('<p class="section-header" style="margin-top: 1rem;">💿 Category Mix</p>', unsafe_allow_html=True)
        debit_df = spend_df[spend_df["type"] == "debit"] if len(spend_df) > 0 else spend_df
        cat_totals = debit_df.groupby("category")["amount"].sum().reset_index() if len(debit_df) > 0 else pd.DataFrame(columns=["category", "amount"])
        total_spend = cat_totals["amount"].sum() if len(cat_totals) > 0 else 0

        fig_donut = go.Figure(go.Pie(
            labels=cat_totals["category"].str.title() if len(cat_totals) > 0 else [],
            values=cat_totals["amount"] if len(cat_totals) > 0 else [],
            hole=0.68,
            marker=dict(colors=[CATEGORY_COLORS.get(c.lower(), "#94A3B8") for c in cat_totals["category"]]),
            textinfo="none",
            hovertemplate=f"<b>%{{label}}</b><br>{currency}%{{value:,.0f}}<br>%{{percent}}<extra></extra>",
        ))
        fig_donut.update_layout(
            annotations=[dict(text=f"Spend<br><span style='font-size:22px; color:#F8FAFC; font-weight:bold;'>{currency}{total_spend:,.0f}</span>", x=0.5, y=0.5, showarrow=False)],
            showlegend=True,
            legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
            height=400,
            **{k: v for k, v in plot_theme().items() if k not in ["xaxis", "yaxis", "margin"]},
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_cashflow:
        st.markdown('<p class="section-header" style="margin-top: 1rem;">💸 Cashflow by Type</p>', unsafe_allow_html=True)
        flow_df = (
            df.groupby("type")["amount"].sum().reindex(["credit", "debit"], fill_value=0).reset_index()
            if len(df) > 0 else pd.DataFrame({"type": ["credit", "debit"], "amount": [0, 0]})
        )
        fig_flow = go.Figure(go.Bar(
            x=flow_df["type"].str.title(),
            y=flow_df["amount"],
            marker_color=["#34D399", "#FBBF24"],
            hovertemplate=f"<b>%{{x}}</b><br>{currency}%{{y:,.0f}}<extra></extra>",
        ))
        fig_flow.update_layout(height=400, showlegend=False, **plot_theme())
        st.plotly_chart(fig_flow, use_container_width=True)


# ─── PAGE 2: Anomaly Explorer ─────────────────────────────────────────────────
def build_progress_bar(pct: float, width: int = 10) -> str:
    """Helper to build ASCII art progress bars."""
    filled = int((pct) * width)
    return "█" * filled + "░" * (width - filled)

def page_anomalies(df: pd.DataFrame, profile: UserProfile, currency: str, detector):
    st.markdown("## 🔴 Anomaly Explorer")
    st.caption("Investigate unusual patterns and spending deviations detected by the system.")
    
    # --- MODEL DIAGNOSTICS PANEL ---
    perf = detector.evaluate_performance(df)
    
    st.markdown('<div class="card" style="padding: 15px; margin-top: 15px; margin-bottom: 30px;">', unsafe_allow_html=True)
    if perf["has_ground_truth"]:
        col_m1, col_m2, col_m3, col_m4 = st.columns([1.5, 1.5, 1.5, 2])
        p_pct, r_pct, f_pct = perf["precision"], perf["recall"], perf["false_alert_rate"]
        
        with col_m1:
            st.markdown(f"**Precision**<br><span style='color:#34D399; font-family:monospace;'>{build_progress_bar(p_pct)}</span> {p_pct:.0%}", unsafe_allow_html=True)
            st.markdown(f"<span style='font-size:0.75rem; color:#9CA3AF;'>{'Low false alerts' if p_pct>0.8 else 'High false alerts'}</span>", unsafe_allow_html=True)
        with col_m2:
            st.markdown(f"**Recall**<br><span style='color:#34D399; font-family:monospace;'>{build_progress_bar(r_pct)}</span> {r_pct:.0%}", unsafe_allow_html=True)
            st.markdown(f"<span style='font-size:0.75rem; color:#9CA3AF;'>{'Good coverage' if r_pct>0.7 else 'Missing anomalies'}</span>", unsafe_allow_html=True)
        with col_m3:
            color_fa = "#F87171" if f_pct > 0.2 else "#FBBF24"
            st.markdown(f"**False Alerts**<br><span style='color:{color_fa}; font-family:monospace;'>{build_progress_bar(f_pct)}</span> {f_pct:.0%}", unsafe_allow_html=True)
        with col_m4:
            st.markdown(f"**Engine Mode**<br><span style='color:#E5E7EB; font-weight:500;'>{perf['mode']}</span>", unsafe_allow_html=True)
            st.markdown(f"<span style='font-size:0.75rem; color:#9CA3AF;'>System is measuring accuracy internally based on synthetic flags.</span>", unsafe_allow_html=True)
    else:
        # Degraded Mode (Real Data)
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown(f"**Model Confidence** (No Ground Truth)<br><span style='color:#9CA3AF;'>Active</span>", unsafe_allow_html=True)
        with col_m2:
            st.markdown(f"**Flag Density**<br><span style='color:#E5E7EB;'>{perf['flag_density']} ({perf['anomaly_rate']:.1%})</span>", unsafe_allow_html=True)
        with col_m3:
            st.markdown(f"**Avg Anomaly Score**<br><span style='color:#E5E7EB;'>{perf['avg_score']:.2f}</span>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    anomalies = df[df["is_anomaly"] == 1].copy()

    if len(anomalies) == 0:
        st.success("✅ No anomalies detected! Your spending looks healthy.")
        return

    # ── Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Anomalies", len(anomalies))
    with col2:
        critical = len(anomalies[anomalies["anomaly_severity"] == "critical"])
        st.metric("Requires Attention", critical, delta="Urgent", delta_color="inverse")
    with col3:
        warning = len(anomalies[anomalies["anomaly_severity"] == "warning"])
        st.metric("Unusual Activity", warning)
    with col4:
        st.metric("Total Anomalous Spend", fmt(anomalies["amount"].sum(), currency))

    st.divider()

    # ── Scatter Plot: Amount vs Anomaly Score
    col_chart, col_bar = st.columns([3, 2])

    with col_chart:
        st.markdown("**Anomaly Score vs. Amount**")
        st.caption("Points further left (lower score) are more anomalous.")

        fig_scatter = go.Figure()

        # Normal transactions (sample for perf)
        normal = df[df["is_anomaly"] == 0].sample(min(500, len(df[df["is_anomaly"] == 0])), random_state=42)
        fig_scatter.add_trace(go.Scatter(
            x=normal["anomaly_score"],
            y=normal["amount"],
            mode="markers",
            name="Typical",
            marker=dict(color="#1F2937", size=5),
            hoverinfo="skip",
        ))

        # Anomalies — color by severity
        for severity, color in [("critical", "#F87171"), ("warning", "#FBBF24")]:
            sev_df = anomalies[anomalies["anomaly_severity"] == severity]
            if len(sev_df) > 0:
                fig_scatter.add_trace(go.Scatter(
                    x=sev_df["anomaly_score"],
                    y=sev_df["amount"],
                    mode="markers",
                    name=severity.title(),
                    marker=dict(color=color, size=9, symbol="circle",
                                line=dict(color="white", width=1)),
                    hovertemplate=(
                        f"<b>%{{customdata[0]}}</b><br>"
                        f"Amount: {currency}%{{y:,.0f}}<br>"
                        f"Category: %{{customdata[1]}}<br>"
                        f"Score: %{{x:.3f}}<extra></extra>"
                    ),
                    customdata=sev_df[["merchant", "category"]].values,
                ))

        fig_scatter.add_vline(
            x=df["anomaly_score"].quantile(0.05),
            line_dash="dash", line_color="#334155",
            annotation_text="Anomaly threshold", annotation_font_size=11,
            annotation_font_color="#9CA3AF"
        )
        fig_scatter.update_layout(title="Transaction Anomaly Map", **plot_theme())
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_bar:
        st.markdown("**Anomalies by Category**")
        cat_counts = anomalies["category"].value_counts().reset_index()
        cat_counts.columns = ["category", "count"]
        colors = [CATEGORY_COLORS.get(c, "#94A3B8") for c in cat_counts["category"]]

        fig_bar = go.Figure(go.Bar(
            x=cat_counts["count"],
            y=cat_counts["category"].str.title(),
            orientation="h",
            marker_color=colors,
            hovertemplate="<b>%{y}</b>: %{x} anomalies<extra></extra>",
        ))
        fig_bar.update_layout(
            title="Anomaly Distribution",
            yaxis=dict(autorange="reversed"),
            **{k: v for k, v in plot_theme().items() if k != "yaxis"}
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Filters
    st.divider()
    st.markdown("### 🔍 Filter & Explore")

    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        categories = ["All"] + sorted(anomalies["category"].unique().tolist())
        cat_filter = st.selectbox("Category", categories, key="anom_cat")
    with f_col2:
        sev_filter = st.selectbox("Severity", ["All", "Critical", "Warning"], key="anom_sev")
    with f_col3:
        sort_by = st.selectbox("Sort by", ["Date (Newest)", "Amount (Highest)", "Score (Most Anomalous)"])

    filtered = anomalies.copy()
    if cat_filter != "All":
        filtered = filtered[filtered["category"] == cat_filter]
    if sev_filter != "All":
        filtered = filtered[filtered["anomaly_severity"] == sev_filter.lower()]

    sort_map = {
        "Date (Newest)": ("date", False),
        "Amount (Highest)": ("amount", False),
        "Score (Most Anomalous)": ("anomaly_score", True),
    }
    sort_col, sort_asc = sort_map[sort_by]
    filtered = filtered.sort_values(sort_col, ascending=sort_asc)

    st.markdown(f"**{len(filtered)} transactions** match your filters")

    # ── Expandable anomaly cards
    for _, row in filtered.iterrows():
        sev = row["anomaly_severity"]
        sev_icon = "🔴" if sev == "critical" else "🟡"
        date_str = row["date"].strftime("%b %d, %Y") if hasattr(row["date"], "strftime") else str(row["date"])

        with st.expander(
            f"{sev_icon} {date_str} — {currency}{row['amount']:,.0f} on {row['category'].title()} at {row['merchant']}"
        ):
            exp_col1, exp_col2 = st.columns([3, 1])
            with exp_col1:
                st.markdown("**Why was this flagged?**")
                struct = row.get("structured_explanation", {})
                if isinstance(struct, dict) and struct.get("cause"):
                    st.markdown(f"**What happened:**<br><span style='color:#E5E7EB;'>{struct.get('what_happened', 'N/A')}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Baseline & Deviation:**<br><span style='color:#9CA3AF;'>{struct.get('baseline', 'N/A')}</span> <span style='color:#F87171;'>({struct.get('deviation', 'N/A')})</span>", unsafe_allow_html=True)
                    st.markdown(f"**Cause:**<br><span style='color:#E5E7EB;'>{struct.get('cause', 'Multi-factor')}</span>", unsafe_allow_html=True)
                    
                    if row.get("explanation"):
                        with st.expander("Show detailed breakdown"):
                            st.caption(row.get("explanation"))
                else:
                    st.info("Multi-factor anomaly: combination of amount, category, and timing.")
            with exp_col2:
                st.metric("Amount", fmt(row["amount"], currency))
                st.metric("Severity", sev.upper())
                
                conf = str(row.get("anomaly_confidence", "low")).lower()
                if conf == "high":
                    st.markdown("**Confidence: High**<br><span style='color:#34D399; font-size:0.75rem'>(Statistical + Model agreement)</span>", unsafe_allow_html=True)
                elif conf == "medium":
                    st.markdown("**Confidence: Medium**<br><span style='color:#FBBF24; font-size:0.75rem'>(Model isolation only)</span>", unsafe_allow_html=True)
                else:
                    st.markdown("**Confidence: Low**<br><span style='color:#94A3B8; font-size:0.75rem'>(Weak statistical support)</span>", unsafe_allow_html=True)
                    
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Mark as Expected", key=f"btn_exp_{row.name}"):
                    if st.session_state.get("guest_mode", False):
                        st.toast("Guest Mode Active: Database saves are disabled.")
                    else:
                        add_expected_transaction(row['merchant'].lower(), row['amount'])
                        st.toast(f"Rule added to ignore {row['merchant']} around {currency}{row['amount']:,.0f}!")
                        st.rerun()


# ─── PAGE 3: Trends ───────────────────────────────────────────────────────────
def page_trends(df: pd.DataFrame, currency: str):
    st.markdown("## 📈 Spending Trends")
    st.divider()
    df = behavioral_spend_view(df)
    if len(df) == 0:
        st.info("No spending data available after transfer filtering.")
        return

    # ── Month-over-Month Delta Table
    st.markdown('<p class="section-header">🗓️ Month-over-Month Delta</p>', unsafe_allow_html=True)
    try:
        df_dates = df.copy()
        df_dates["date"] = pd.to_datetime(df_dates["date"])
        current_month = df_dates['date'].max().to_period('M')
        last_month = current_month - 1
        
        curr_df = df_dates[df_dates['date'].dt.to_period('M') == current_month]
        last_df = df_dates[df_dates['date'].dt.to_period('M') == last_month]
        
        curr_cat = curr_df.groupby('category')['amount'].sum()
        last_cat = last_df.groupby('category')['amount'].sum()
        
        delta_df = pd.DataFrame({"This Month": curr_cat, "Last Month": last_cat}).fillna(0)
        delta_df["Delta (%)"] = np.where(delta_df["Last Month"] == 0, 0, 
                                         (delta_df["This Month"] - delta_df["Last Month"]) / delta_df["Last Month"] * 100)
        
        def format_delta(val, amount_diff):
            if val == 0 and amount_diff == 0: return "-"
            icon = "▲" if val > 0 else "▼"
            color = "#FF5252" if val > 0 else "#4CAF50" # Red if spend more, Green if spend less
            return f'<span style="color: {color};">{icon} {abs(val):.0f}%</span>'

        display_df = delta_df.copy()
        display_df["Category"] = display_df.index.str.title()
        amount_diffs = display_df["This Month"] - display_df["Last Month"]
        display_df["Δ"] = [format_delta(v, d) for v, d in zip(display_df["Delta (%)"], amount_diffs)]
        display_df["Last Month"] = display_df["Last Month"].apply(lambda x: fmt(x, currency))
        display_df["This Month"] = display_df["This Month"].apply(lambda x: fmt(x, currency))
        
        st.write(display_df[["Category", "Last Month", "This Month", "Δ"]].to_html(escape=False, index=False), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
    except Exception as e:
        st.warning("Insufficient data for Month-over-Month comparison.")

    # ── Time granularity selector
    granularity = st.radio(
        "View by", ["Daily", "Weekly", "Monthly"], horizontal=True, index=1
    )

    if granularity == "Daily":
        grouped = df.groupby("date")["amount"].sum().reset_index()
        x_col = "date"
        tick_format = "%b %d"
    elif granularity == "Weekly":
        df["week_start"] = df["date"] - pd.to_timedelta(df["date"].dt.dayofweek, unit="d")
        grouped = df.groupby("week_start")["amount"].sum().reset_index()
        grouped.columns = ["date", "amount"]
        x_col = "date"
        tick_format = "Week of %b %d"
    else:
        grouped = df.groupby(df["date"].dt.to_period("M"))["amount"].sum().reset_index()
        grouped["date"] = grouped["date"].astype(str)
        x_col = "date"
        tick_format = None

    # Rolling average
    if granularity == "Daily":
        grouped["rolling_avg"] = grouped["amount"].rolling(7, min_periods=1).mean()
        rolling_label = "7-day Rolling Avg"
    else:
        grouped["rolling_avg"] = grouped["amount"].rolling(3, min_periods=1).mean()
        rolling_label = "3-period Rolling Avg"

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Bar(
        x=grouped[x_col],
        y=grouped["amount"],
        name="Spending",
        marker_color="rgba(108,99,255,0.5)",
        hovertemplate=f"<b>%{{x}}</b><br>{currency}%{{y:,.0f}}<extra></extra>",
    ))
    fig_trend.add_trace(go.Scatter(
        x=grouped[x_col],
        y=grouped["rolling_avg"],
        name=rolling_label,
        line=dict(color="#34D399", width=2.5),
        hovertemplate=f"Avg: {currency}%{{y:,.0f}}<extra></extra>",
    ))
    fig_trend.update_layout(
        title=f"{granularity} Spending Trend",
        legend=dict(orientation="h", y=1.1),
        **plot_theme()
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # ── Category-wise trend lines
    st.markdown('<p class="section-header">Category Trends Over Time</p>', unsafe_allow_html=True)

    categories = sorted(df["category"].unique())
    selected_cats = st.multiselect(
        "Select categories to compare",
        options=categories,
        default=categories[:4],
        format_func=str.title,
    )

    if selected_cats:
        fig_cat_trend = go.Figure()
        for cat in selected_cats:
            cat_df = df[df["category"] == cat]
            if granularity == "Monthly":
                cat_grouped = cat_df.groupby(cat_df["date"].dt.to_period("M"))["amount"].sum().reset_index()
                cat_grouped["date"] = cat_grouped["date"].astype(str)
            elif granularity == "Weekly":
                cat_df = cat_df.copy()
                cat_df["week_start"] = cat_df["date"] - pd.to_timedelta(cat_df["date"].dt.dayofweek, unit="d")
                cat_grouped = cat_df.groupby("week_start")["amount"].sum().reset_index()
                cat_grouped.columns = ["date", "amount"]
            else:
                cat_grouped = cat_df.groupby("date")["amount"].sum().reset_index()

            fig_cat_trend.add_trace(go.Scatter(
                x=cat_grouped["date"],
                y=cat_grouped["amount"],
                name=cat.title(),
                mode="lines+markers",
                line=dict(color=CATEGORY_COLORS.get(cat, "#94A3B8"), width=2),
                marker=dict(size=5),
                hovertemplate=f"<b>{cat.title()}</b>: {currency}%{{y:,.0f}}<extra></extra>",
            ))

        fig_cat_trend.update_layout(
            title="Category-wise Spending Over Time",
            legend=dict(orientation="h", y=1.1),
            **plot_theme()
        )
        st.plotly_chart(fig_cat_trend, use_container_width=True)

    # ── Spending Heatmap
    st.markdown('<p class="section-header">Spending Heatmap (Day of Week × Month)</p>', unsafe_allow_html=True)
    heatmap_data = df.groupby(["day_name", "month"])["amount"].sum().reset_index()
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heatmap_pivot = heatmap_data.pivot_table(
        index="day_name", columns="month", values="amount", fill_value=0
    )
    heatmap_pivot = heatmap_pivot.reindex([d for d in day_order if d in heatmap_pivot.index])

    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    col_labels = [month_names.get(c, str(c)) for c in heatmap_pivot.columns]

    fig_heat = go.Figure(go.Heatmap(
        z=heatmap_pivot.values,
        x=col_labels,
        y=heatmap_pivot.index.tolist(),
        colorscale=[[0, "#1A1D29"], [0.5, "#34D399"], [1, "#F87171"]],
        hovertemplate=f"<b>%{{y}} - %{{x}}</b><br>{currency}%{{z:,.0f}}<extra></extra>",
    ))
    fig_heat.update_layout(title="Spending Intensity Heatmap", **{
        k: v for k, v in plot_theme().items() if k not in ["xaxis", "yaxis"]
    })
    st.plotly_chart(fig_heat, use_container_width=True)


# ─── PAGE 4: Behavioral Profile ───────────────────────────────────────────────
def page_profile(df: pd.DataFrame, profile: UserProfile, currency: str):
    st.markdown("## 👤 Your Spending Profile")
    st.caption("Your financial fingerprint — how you typically spend money.")
    st.divider()
    spend_df = behavioral_spend_view(df)

    # ── Row 1: Category breakdown
    st.markdown('<p class="section-header">Category Spending Breakdown</p>', unsafe_allow_html=True)

    prof_rows = []
    for cat, p in profile.category_profiles.items():
        prof_rows.append({
            "Category": cat.title(),
            "Avg Transaction": fmt(p["mean"], currency),
            "Normal Range": f"{currency}{p['normal_range'][0]:,.0f} – {currency}{p['normal_range'][1]:,.0f}",
            "Transactions": p["transaction_count"],
            "Total Spend": fmt(p["total_spend"], currency),
            "Share": f"{p['share_of_total']}%",
            "_sort_total": p["total_spend"],
        })

    prof_df = pd.DataFrame(prof_rows).sort_values("_sort_total", ascending=False) if prof_rows else pd.DataFrame()
    if "_sort_total" in prof_df.columns:
        prof_df = prof_df.drop(columns=["_sort_total"])
    with st.expander("Show category detail table", expanded=False):
        st.dataframe(prof_df, use_container_width=True, hide_index=True)

    # ── Row 2: Category bars + weekday pattern
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Category Totals**")
        cat_totals = spend_df.groupby("category")["amount"].sum().reset_index().sort_values("amount")
        colors = [CATEGORY_COLORS.get(c, "#94A3B8") for c in cat_totals["category"]]

        fig_h = go.Figure(go.Bar(
            x=cat_totals["amount"],
            y=cat_totals["category"].str.title(),
            orientation="h",
            marker_color=colors,
            hovertemplate=f"<b>%{{y}}</b>: {currency}%{{x:,.0f}}<extra></extra>",
        ))
        fig_h.update_layout(title="Total Spend by Category", **plot_theme())
        st.plotly_chart(fig_h, use_container_width=True)

    with col_right:
        st.markdown("**Day-of-Week Spending Pattern**")
        dow = profile.temporal_profile["day_of_week_avg_spend"]
        days = list(dow.keys())
        amounts = list(dow.values())
        colors_dow = ["#FBBF24" if d in ("Saturday", "Sunday") else "#34D399" for d in days]

        fig_dow = go.Figure(go.Bar(
            x=days,
            y=amounts,
            marker_color=colors_dow,
            hovertemplate=f"<b>%{{x}}</b>: {currency}%{{y:,.0f}} avg<extra></extra>",
        ))
        fig_dow.update_layout(title="Avg Spend per Day of Week", **plot_theme())
        st.plotly_chart(fig_dow, use_container_width=True)

    # ── Row 3: Top merchants
    st.markdown('<p class="section-header">Top Merchants by Category</p>', unsafe_allow_html=True)

    cat_list = sorted(profile.merchant_profile["by_category"].keys())
    if not cat_list:
        st.info("No merchant profile available after transfer filtering.")
        return
    sel_cat = st.selectbox("Select a category", cat_list, format_func=str.title, key="prof_cat")

    merch_data = profile.merchant_profile["by_category"][sel_cat]["top_merchants"]
    if merch_data:
        merch_df = pd.DataFrame(merch_data)
        rename_map = {
            "name": "Merchant",
            "count": "Transactions",
            "total_spend": "Total Spend",
            "avg_spend": "Avg Spend",
        }
        merch_df = merch_df.rename(columns=rename_map)
        display_cols = [col for col in ["Merchant", "Transactions", "Total Spend", "Avg Spend"] if col in merch_df.columns]
        merch_df = merch_df[display_cols]
        if "Total Spend" in merch_df.columns:
            merch_df["Total Spend"] = merch_df["Total Spend"].apply(lambda x: fmt(x, currency))
        if "Avg Spend" in merch_df.columns:
            merch_df["Avg Spend"] = merch_df["Avg Spend"].apply(lambda x: fmt(x, currency))
        with st.expander("Show merchant detail table", expanded=False):
            st.dataframe(merch_df, use_container_width=True, hide_index=True)

    # ── Velocity stats
    st.markdown('<p class="section-header">Spending Velocity</p>', unsafe_allow_html=True)
    vcol1, vcol2, vcol3 = st.columns(3)
    vel = profile.velocity_profile
    with vcol1:
        st.metric("Avg Daily Spend", fmt(vel["avg_daily_spend"], currency))
    with vcol2:
        st.metric("Avg Weekly Spend", fmt(vel["avg_weekly_spend"], currency))
    with vcol3:
        st.metric("Avg Monthly Spend", fmt(vel["avg_monthly_spend"], currency))


# ─── PAGE 5: Financial Health ─────────────────────────────────────────────────
def page_health(scorer: HealthScorer, insights_gen: InsightGenerator, currency: str):
    st.markdown("## 💚 Financial Health")
    st.caption("Your overall financial health score and actionable recommendations.")
    st.divider()

    breakdown = scorer.get_score_breakdown()
    total = breakdown["total_score"]
    grade = breakdown["grade"]

    # ── Score Gauge
    col_gauge, col_components = st.columns([1, 2])

    with col_gauge:
        # Color based on score
        if total >= 75:
            gauge_color = "#34D399"
        elif total >= 50:
            gauge_color = "#FBBF24"
        else:
            gauge_color = "#FF5252"

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=total,
            delta={"reference": 70, "valueformat": ".1f"},
            title={"text": f"Health Score<br><span style='font-size:1.3rem'>{grade}</span>",
                   "font": {"size": 15, "color": "#FAFAFA"}},
            number={"font": {"size": 36, "color": gauge_color}, "suffix": "/100"},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#8B8FA8"},
                "bar": {"color": gauge_color, "thickness": 0.25},
                "bgcolor": "#1A1D29",
                "steps": [
                    {"range": [0, 40], "color": "rgba(255,82,82,0.15)"},
                    {"range": [40, 70], "color": "rgba(255,167,38,0.15)"},
                    {"range": [70, 100], "color": "rgba(102,187,106,0.15)"},
                ],
                "threshold": {
                    "line": {"color": "#34D399", "width": 3},
                    "thickness": 0.75,
                    "value": 70,
                },
            },
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#FAFAFA"),
            height=320,
            margin=dict(l=20, r=20, t=50, b=30),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_components:
        st.markdown("**Score Components (20 pts each)**")

        for key, comp in breakdown["components"].items():
            pct = comp["percentage"]
            bar_color = "#66BB6A" if pct >= 70 else ("#FFA726" if pct >= 50 else "#FF5252")

            st.markdown(f"**{comp['label']}** — {comp['score']:.1f}/20")
            st.progress(pct / 100, text=comp["description"])

    st.divider()

    st.markdown('<p class="section-header">💡 Top 3 Actions</p>', unsafe_allow_html=True)
    render_behavior_insight_cards(insights_gen.get_insights(), limit=3)


# ─── Main App ─────────────────────────────────────────────────────────────────
def main():
    if not os.path.exists("config.yaml"):
        st.error("🔒 No authentication config found.")
        st.markdown("""
        **Security Engine Active:** PFAD requires local authentication.
        
        Please create a `config.yaml` file with your credentials. Example configuration is provided in `config_example.yaml`.
        """)
        st.stop()
        
    with open("config.yaml", "r", encoding="utf-8") as file:
        config = yaml.load(file, Loader=SafeLoader)
        
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
    )
    
    if "guest_mode" not in st.session_state:
        st.session_state["guest_mode"] = False

    if not st.session_state["guest_mode"]:
        # Render login via auth module (it handles UI automatically)
        authenticator.login()
        
        if st.session_state["authentication_status"] is False:
            st.error('Username/password is incorrect')
            st.stop()
        elif st.session_state["authentication_status"] is None:
            st.warning('Please enter your username and password')
            st.markdown("---")
            st.markdown("### 🏃‍♂️ Quick Evaluation Mode")
            st.caption("Evaluate the UI and architecture without saving any database changes.")
            if st.button("Continue as Guest ➝"):
                st.session_state["guest_mode"] = True
                st.rerun()
            st.stop()

    page, uploaded_file, currency_symbol, contamination = render_sidebar()
    with st.sidebar:
        if st.session_state.get("guest_mode", False):
            if st.button("Exit Guest Mode / Log In"):
                st.session_state["guest_mode"] = False
                st.rerun()
        else:
            authenticator.logout("Logout", "sidebar")

    # Load data (multi-format: CSV, Excel, PDF)
    df_raw, parser_warnings = load_data(uploaded_file, currency_symbol, contamination)

    # Run pipeline (cached) — NEVER crashes
    with st.spinner("🧠 Running anomaly detection pipeline..."):
        try:
            df, profile, scorer, insights_gen, detector, pipeline_warnings = run_pipeline(
                df_raw, currency_symbol, contamination
            )
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.info("Your CSV may be in an unsupported format. Ensure it has at least a date and amount column.")
            st.stop()

    # ── Show Data Quality & Parser Warnings ────────────────────────────
    all_system_warnings = parser_warnings + pipeline_warnings
    if all_system_warnings:
        with st.sidebar:
            st.divider()
            with st.expander(f"📋 Data Quality ({len(all_system_warnings)} notes)", expanded=False):
                for w in all_system_warnings:
                    if w.startswith("❌"):
                        st.error(w)
                    elif w.startswith("⚠️"):
                        st.warning(w)
                    elif w.startswith("✅"):
                        st.success(w)
                    elif w.startswith("📊") or w.startswith("📄") or w.startswith("📂"):
                        st.info(w)
                    elif w.startswith("🔄") or w.startswith("🤖"):
                        st.info(w)
                    else:
                        st.info(w)

    # ── Show Classification Summary ───────────────────────────────────
    if uploaded_file is not None:
        cls_summary = get_classification_summary(df)
        if cls_summary.get("rule", 0) > 0 or cls_summary.get("ml", 0) > 0 or cls_summary.get("user", 0) > 0:
            with st.sidebar:
                with st.expander("🏷️ Category Classification", expanded=False):
                    st.markdown(f"""
                    | Source | Count |
                    |--------|-------|
                    | 👤 User Memory / Provided | {cls_summary.get('user', 0)} |
                    | ✅ Rule Engine | {cls_summary.get('rule', 0)} |
                    | 🤖 ML Model | {cls_summary.get('ml', 0)} |
                    | 🔁 Transfer Rows | {cls_summary.get('transfer_rows', 0)} |
                    """)
                    st.caption(
                        f"Confidence: {cls_summary.get('manual_confidence', 0)} Manual · "
                        f"{cls_summary.get('high_confidence', 0)} High · "
                        f"{cls_summary.get('low_confidence', 0)} Low"
                    )
                    
    # ── User Override UI (Adaptive System) ────────────────────────────
    if uploaded_file is not None:
        # Get merchants that are uncategorized, 'others', or have low confidence
        low_conf_mask = df["category_confidence"].astype(str).str.lower().eq("low")
        low_conf_df = df[low_conf_mask]
        
        if len(low_conf_df) > 0:
            with st.sidebar:
                with st.expander("⚠️ Needs Categorization", expanded=True):
                    st.caption("Teach the system mapping for unknown merchants.")
                    from src.category_classifier import load_mapping, save_mapping, normalize_merchant
                    user_mapping = load_mapping()
                    unique_merchants = low_conf_df["merchant"].unique()
                    
                    with st.form("categorize_form"):
                        updates = {}
                        for merchant in unique_merchants[:15]: # Cap at 15 for UI layout
                            current_cat = df[df["merchant"] == merchant]["category"].iloc[0].lower()
                            opts = CATEGORIES
                            idx = opts.index(current_cat) if current_cat in opts else len(opts) - 1
                            
                            new_cat = st.selectbox(
                                f"💳 {merchant}",
                                opts,
                                index=idx,
                                key=f"cat_ui_{merchant}"
                            )
                            updates[merchant] = new_cat
                        
                        if len(unique_merchants) > 15:
                            st.caption(f"... and {len(unique_merchants) - 15} more.")
                            
                        if st.form_submit_button("Save Preferences"):
                            if st.session_state.get("guest_mode", False):
                                st.warning("Database saves disabled in Guest Mode.")
                            else:
                                for m, c in updates.items():
                                    user_mapping[normalize_merchant(m)] = c
                                save_mapping(user_mapping)
                                st.success("Saved! Reload to see changes.")

    # Route to page
    if "Overview" in page:
        page_overview(df, profile, scorer, insights_gen, currency_symbol)
    elif "Anomalies" in page:
        page_anomalies(df, profile, currency_symbol, detector)
    elif "Trends" in page:
        page_trends(df, currency_symbol)
    elif "Profile" in page:
        page_profile(df, profile, currency_symbol)
    elif "Health" in page:
        page_health(scorer, insights_gen, currency_symbol)


if __name__ == "__main__":
    main()

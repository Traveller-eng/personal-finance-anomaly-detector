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
    Upload CSV → load → preprocess → feature engineer → profile →
    anomaly detect → explain → health score → insights → display
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import sys

# Add project root to path so src/ imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_from_dataframe, load_csv
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
.fintech-score { font-size: 3rem; font-weight: 600; background: linear-gradient(145deg, #F3F4F6, #9CA3AF); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

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
    background: conic-gradient(#7C9CFF 0% var(--score-pct, 0%), #1F2937 var(--score-pct, 0%));
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
.insight-bullet { color: #7C9CFF; font-weight: bold; }

/* Subtext & Tones */
.subtle-emphasis { color: #7C9CFF; font-weight: 500; }
.alert-text { color: #FBBF24; font-weight: 500; }

div[data-testid="stMetricValue"] { font-size: 1.4rem !important; font-weight: 600; color: #E5E7EB; }
</style>
""", unsafe_allow_html=True)


# ─── Pipeline Runner (cached) ─────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_pipeline(df_raw: pd.DataFrame, currency_symbol: str, contamination: float):
    """
    Full ML pipeline: load → clean → engineer → profile → detect → explain → score.
    Cached so rerunning the same data doesn't repeat computation.


    """
    # 1. Load + validate
    df = load_from_dataframe(df_raw.copy())

    # 2. Clean & preprocess
    df = clean_data(df)

    # 3. Feature engineering
    df = engineer_features(df)

    # 4. User profiling
    profile = UserProfile(df, currency_symbol=currency_symbol)

    # 5. Anomaly detection
    detector = AnomalyDetector(config={"contamination": contamination})
    df = detector.fit_predict(df)

    # 6. Explanation generation
    df = explain_anomalies(df, profile)

    # 7. Health scoring
    scorer = HealthScorer(df, profile)

    # 8. Insights
    insights = InsightGenerator(df, profile, currency_symbol=currency_symbol)

    return df, profile, scorer, insights, detector


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
            ["Use Sample Data", "Upload Your CSV"],
            index=0,
            label_visibility="collapsed",
        )

        uploaded_file = None
        parser_type = st.selectbox(
            "Bank Format",
            ["Generic (Default)", "Mint", "YNAB", "Chase"],
        )
        if data_source == "Upload Your CSV":
            uploaded_file = st.file_uploader(
                "Upload CSV",
                type=["csv"],
                help="Required columns: date, amount, category, merchant",
            )
            st.caption("Required columns for Generic: `date`, `amount`, `category`, `merchant`")

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

    return page, uploaded_file, currency_symbol, contamination, parser_type


# ─── Data Loading ─────────────────────────────────────────────────────────────
def load_data(uploaded_file, currency_symbol, contamination, parser_type):
    """Load data and run the full pipeline."""
    sample_path = os.path.join(os.path.dirname(__file__), "data", "transactions.csv")

    # Generate sample data if it doesn't exist
    if not os.path.exists(sample_path) and uploaded_file is None:
        with st.spinner("Generating sample transaction data..."):
            from generate_sample_data import generate_dataset, save_dataset
            df_gen = generate_dataset()
            save_dataset(df_gen)

    # Load data
    if uploaded_file is not None:
        df_raw = load_from_dataframe(pd.read_csv(uploaded_file), parser_type=parser_type)
    else:
        if not os.path.exists(sample_path):
            st.error("Sample data not found. Please generate it or upload a CSV.")
            st.stop()
        df_raw = load_csv(sample_path, parser_type="Generic (Default)")

    return df_raw


# ─── Helper Functions ─────────────────────────────────────────────────────────
def fmt(amount: float, symbol: str) -> str:
    """Format currency amount."""
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


CATEGORY_COLORS = {
    "food": "#F87171",
    "shopping": "#7C9CFF",
    "transport": "#34D399",
    "bills": "#FBBF24",
    "entertainment": "#F472B6",
    "health": "#A7F3D0",
    "education": "#93C5FD",
    "rent": "#C084FC",
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

    # 1. Hero Section & What Changed Integration
    health_score = scorer.total_score
    grade = scorer._get_grade()
    
    # We bring What Changed directly to the top so there is no empty right-side layout.
    hero_col_left, hero_col_right = st.columns([1, 1], gap="large")
    
    with hero_col_left:
        st.markdown(f"<h3 style='margin-bottom: 1.5rem;'>Financial Health Index</h3>", unsafe_allow_html=True)
        # Re-introduce semantic colors:
        color = "#34D399" if health_score >= 75 else ("#FBBF24" if health_score >= 50 else "#F87171")
        
        st.markdown(f"""
        <div style="
            width: 220px; height: 220px; margin: 0 auto; 
            border-radius: 50%; 
            display: flex; flex-direction: column; justify-content: center; align-items: center;
            background: conic-gradient({color} {health_score}%, #1F2937 {health_score}%);
            padding: 4px; box-shadow: inset 0 4px 15px rgba(0,0,0,0.5);
        ">
            <div style="background: #0E1117; width: 100%; height: 100%; border-radius: 50%; display: flex; flex-direction: column; justify-content: center; align-items: center; box-shadow: inset 0 4px 20px rgba(0,0,0,0.5);">
                <div class="fintech-score" style="background: linear-gradient(145deg, #F3F4F6, {color}); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{health_score:.0f}</div>
                <div class="score-caption" style="margin-top: 2px;">Out of 100 ({grade})</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        top_insight = insights_gen.get_top_insights(1)
        if top_insight:
            st.markdown(f'<div style="text-align: center; margin-top: 15px;"><span class="subtle-emphasis">{top_insight[0]["title"]}</span><br><span style="color: #E5E7EB; font-size: 0.95rem;">{top_insight[0]["suggestion"]}</span></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="text-align: center; margin-top: 15px; color: #9CA3AF; font-size: 0.95rem;">Spending patterns are stable</div>', unsafe_allow_html=True)
            
    with hero_col_right:
        st.markdown(f"<h3 style='margin-bottom: 1.5rem;'>What Changed This Week</h3>", unsafe_allow_html=True)
        insights_list = insights_gen.get_insights()
        if insights_list:
            st.markdown('<div class="insight-panel" style="margin-bottom: 0;">', unsafe_allow_html=True)
            for ins in insights_list[:4]:
                st.markdown(f'''<div class="insight-listItem">
        <span class="insight-bullet">→</span> 
        <span style="color: #E5E7EB; font-size: 0.95rem; line-height: 1.5;">
            <strong>{ins["title"]}</strong><br>
            <span class="text-muted">{ins["message"]}</span>
        </span>
    </div>''', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No significant behavioral changes detected this period.")

    # 2. Priority Section (Anomaly Alerts) - Dominant if exist
    anomalies = df[df["is_anomaly"] == 1]
    if len(anomalies) > 0:
        st.markdown('<p class="section-header" style="margin-top: 3.5rem;">Requires Attention</p>', unsafe_allow_html=True)
        recent_anomalies = anomalies.sort_values(by=["anomaly_score", "date"], ascending=[False, False]).head(5)
        
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
                typical = struct_exp.get("typical", "N/A")
                deviation = struct_exp.get("deviation", "N/A")
                reason = struct_exp.get("reason", "Multi-factor anomaly")
            else:
                typical = "N/A"
                deviation = "N/A"
                reason = row.get("explanation", "")

            amount_fmt = fmt(row['amount'], currency)
            merch = row.get('merchant', 'Unknown')
            cat = row.get('category', '').title()
            date_str = pd.to_datetime(row['date']).strftime('%b %d, %Y')

            cards_html.append(f"""<div class="anomaly-card {sev_class}">
    <div class="anomaly-header">
        <div class="anomaly-title"><span style="color:#9CA3AF; font-size:0.85rem; font-weight: 500;">{icon} &nbsp;·&nbsp;</span> {cat} at {merch}</div>
        <div class="anomaly-amount">{amount_fmt}</div>
    </div>
    <div style="color: #9CA3AF; font-size: 0.8rem; margin-top:-8px; margin-bottom: 8px;">{date_str}</div>
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
    <div class="anomaly-reason" style="margin-top: 1rem; padding-top: 0.8rem; border-top: 1px solid rgba(255,255,255,0.05);">
        <strong>Reason:</strong> {reason}
    </div>
</div>""")

        st.markdown("".join(cards_html), unsafe_allow_html=True)

    # 4 & 5. Trends and Breakdown (Side by Side)
    col_trends, col_donut = st.columns([1.5, 1])

    with col_trends:
        st.markdown('<p class="section-header" style="margin-top: 1rem;">📈 Trends & Smart Forecast</p>', unsafe_allow_html=True)
        
        # Exponential Smoothing
        daily_spend = df.groupby("date")["amount"].sum().reset_index()
        daily_spend = daily_spend.sort_values("date").set_index("date")
        
        # Fill missing days
        full_idx = pd.date_range(daily_spend.index.min(), daily_spend.index.max())
        daily_spend = daily_spend.reindex(full_idx, fill_value=0).reset_index()
        daily_spend.columns = ["date", "amount"]
        
        # EWMA
        daily_spend["ewma"] = daily_spend["amount"].ewm(alpha=0.25, adjust=False).mean()
        
        # Forecast naive
        last_ewma = daily_spend["ewma"].iloc[-1]
        last_date = daily_spend["date"].max()
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=7)
        forecast_df = pd.DataFrame({"date": future_dates, "forecast": [last_ewma]*7})

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Bar(
            x=daily_spend["date"], y=daily_spend["amount"], name="Actual Spend",
            marker_color="rgba(51, 65, 85, 0.4)",
            hovertemplate=f"<b>%{{x}}</b><br>{currency}%{{y:,.0f}}<extra></extra>",
        ))
        fig_trend.add_trace(go.Scatter(
            x=daily_spend["date"], y=daily_spend["ewma"], name="Smoothed Trend",
            line=dict(color="#7C9CFF", width=3), mode="lines",
            hovertemplate=f"<b>%{{x}}</b><br>Trend: {currency}%{{y:,.0f}}<extra></extra>",
        ))
        fig_trend.add_trace(go.Scatter(
            x=forecast_df["date"], y=forecast_df["forecast"], name="7-Day Forecast",
            line=dict(color="#FBBF24", width=2, dash="dot"), mode="lines",
            hovertemplate=f"<b>%{{x}}</b><br>Forecast: {currency}%{{y:,.0f}}<extra></extra>",
        ))
        
        fig_trend.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.1), **{k: v for k, v in plot_theme().items() if k != "margin"})
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_donut:
        st.markdown('<p class="section-header" style="margin-top: 1rem;">💿 Category Breakdown</p>', unsafe_allow_html=True)
        cat_totals = df.groupby("category")["amount"].sum().reset_index()
        total_spend = cat_totals["amount"].sum()
        
        colors = [CATEGORY_COLORS.get(c, "#6366F1") for c in cat_totals["category"]]
        
        # Intelligent Center text formatting
        center_text = f"Total Spend<br><span style='font-size: 24px; color: #F8FAFC; font-weight: bold;'>{currency}{total_spend:,.0f}</span>"
        
        fig_donut = go.Figure(go.Pie(
            labels=cat_totals["category"].str.title(),
            values=cat_totals["amount"],
            hole=0.7,
            marker=dict(colors=colors),
            textinfo="none",
            hovertemplate=f"<b>%{{label}}</b><br>{currency}%{{value:,.0f}}<br>%{{percent}}<extra></extra>",
        ))
        fig_donut.update_layout(
            annotations=[dict(text=center_text, x=0.5, y=0.5, showarrow=False)],
            showlegend=True,
            legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"),
            height=400, margin=dict(l=0, r=0, t=10, b=0),
            **{k: v for k, v in plot_theme().items() if k not in ["xaxis", "yaxis", "margin"]}
        )
        st.plotly_chart(fig_donut, use_container_width=True)


# ─── PAGE 2: Anomaly Explorer ─────────────────────────────────────────────────
def page_anomalies(df: pd.DataFrame, profile: UserProfile, currency: str):
    st.markdown("## 🔴 Anomaly Explorer")
    st.caption("Investigate unusual patterns and spending deviations detected by the system.")
    st.divider()

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
        colors = [CATEGORY_COLORS.get(c, "#6C63FF") for c in cat_counts["category"]]

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
                explanation = row.get("explanation", "")
                if explanation:
                    st.markdown(explanation)
                else:
                    st.info("Multi-factor anomaly: combination of amount, category, and timing.")
            with exp_col2:
                st.metric("Amount", fmt(row["amount"], currency))
                st.metric("Severity", sev.upper())
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Mark as Expected", key=f"btn_exp_{row.name}"):
                    add_expected_transaction(row['merchant'].lower(), row['amount'])
                    st.toast(f"Rule added to ignore {row['merchant']} around {currency}{row['amount']:,.0f}!")
                    st.rerun()


# ─── PAGE 3: Trends ───────────────────────────────────────────────────────────
def page_trends(df: pd.DataFrame, currency: str):
    st.markdown("## 📈 Spending Trends")
    st.divider()

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
        line=dict(color="#FF6B6B", width=2.5),
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
                line=dict(color=CATEGORY_COLORS.get(cat, "#6C63FF"), width=2),
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
        colorscale=[[0, "#1A1D29"], [0.5, "#6C63FF"], [1, "#FF6B6B"]],
        hovertemplate="<b>%{y} - %{x}</b><br>₹%{z:,.0f}<extra></extra>",
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
        })

    prof_df = pd.DataFrame(prof_rows).sort_values("Total Spend", ascending=False)
    st.dataframe(prof_df, use_container_width=True, hide_index=True)

    # ── Row 2: Category bars + weekday pattern
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Category Totals**")
        cat_totals = df.groupby("category")["amount"].sum().reset_index().sort_values("amount")
        colors = [CATEGORY_COLORS.get(c, "#6C63FF") for c in cat_totals["category"]]

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
        colors_dow = ["#EC407A" if d in ("Saturday", "Sunday") else "#6C63FF" for d in days]

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
    sel_cat = st.selectbox("Select a category", cat_list, format_func=str.title, key="prof_cat")

    merch_data = profile.merchant_profile["by_category"][sel_cat]["top_merchants"]
    if merch_data:
        merch_df = pd.DataFrame(merch_data)
        merch_df.columns = ["Merchant", "Transactions", "Total Spend", "Avg Spend"]
        merch_df["Total Spend"] = merch_df["Total Spend"].apply(lambda x: fmt(x, currency))
        merch_df["Avg Spend"] = merch_df["Avg Spend"].apply(lambda x: fmt(x, currency))
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
            gauge_color = "#66BB6A"
        elif total >= 50:
            gauge_color = "#FFA726"
        else:
            gauge_color = "#FF5252"

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=total,
            delta={"reference": 70, "valueformat": ".1f"},
            title={"text": f"Health Score<br><span style='font-size:1.5rem'>{grade}</span>",
                   "font": {"size": 16, "color": "#FAFAFA"}},
            number={"font": {"size": 48, "color": gauge_color}, "suffix": "/100"},
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
                    "line": {"color": "#6C63FF", "width": 3},
                    "thickness": 0.75,
                    "value": 70,
                },
            },
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#FAFAFA"),
            height=300,
            margin=dict(l=10, r=10, t=10, b=10),
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

    # ── Insights
    st.markdown('<p class="section-header">💡 Actionable Insights & Recommendations</p>',
                unsafe_allow_html=True)

    all_insights = insights_gen.get_insights()

    if not all_insights:
        st.success("🎉 No issues found! Your spending looks healthy. Keep it up!")
    else:
        type_icons = {
            "warning": ("🔴", "insight-warning"),
            "positive": ("🟢", "insight-positive"),
            "suggestion": ("💡", "insight-suggestion"),
            "prediction": ("🔮", "insight-prediction"),
        }

        for insight in all_insights:
            icon, css_class = type_icons.get(insight["type"], ("ℹ️", "insight-suggestion"))
            priority_badge = f"{'🔴' if insight['priority'] == 'high' else '🟡' if insight['priority'] == 'medium' else '🔵'}"

            st.markdown(f"""
            <div class="{css_class}">
                <div class="insight-title">{icon} {insight['title']} {priority_badge}</div>
                <div class="insight-msg">{insight['message']}</div>
                <div class="insight-tip">💡 {insight.get('suggestion', '')}</div>
            </div>
            """, unsafe_allow_html=True)


# ─── Main App ─────────────────────────────────────────────────────────────────
def main():
    page, uploaded_file, currency_symbol, contamination, parser_type = render_sidebar()

    # Load data
    df_raw = load_data(uploaded_file, currency_symbol, contamination, parser_type)

    # Run pipeline (cached)
    with st.spinner("🧠 Running anomaly detection pipeline..."):
        try:
            df, profile, scorer, insights_gen, detector = run_pipeline(
                df_raw, currency_symbol, contamination
            )
        except ValueError as e:
            st.error(f"Data error: {e}")
            st.info("Please ensure your CSV has columns: `date`, `amount`, `category`, `merchant`")
            st.stop()

    # Route to page
    if "Overview" in page:
        page_overview(df, profile, scorer, insights_gen, currency_symbol)
    elif "Anomalies" in page:
        page_anomalies(df, profile, currency_symbol)
    elif "Trends" in page:
        page_trends(df, currency_symbol)
    elif "Profile" in page:
        page_profile(df, profile, currency_symbol)
    elif "Health" in page:
        page_health(scorer, insights_gen, currency_symbol)


if __name__ == "__main__":
    main()

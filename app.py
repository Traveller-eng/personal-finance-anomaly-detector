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

/* CSS */
/* Core Structural */
.stApp { background-color: #0F172A; }
.css-1d391kg { background-color: #1E293B; } /* Sidebar background approximation */
.kpi-card {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
}

/* Anomaly Alert Cards - Hierarchy & Constraints */
.anomaly-card {
    background: #1E293B;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.8rem;
    display: flex;
    flex-direction: column;
    max-width: 100%;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.anomaly-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
}
.anomaly-critical { border-left: 5px solid #EF4444; border-top: 1px solid #334155; border-right: 1px solid #334155; border-bottom: 1px solid #334155; }
.anomaly-warning { border-left: 5px solid #F59E0B; border-top: 1px solid #334155; border-right: 1px solid #334155; border-bottom: 1px solid #334155; }
.anomaly-mild { border-left: 5px solid #94A3B8; border-top: 1px solid #334155; border-right: 1px solid #334155; border-bottom: 1px solid #334155; }

.anomaly-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}
.anomaly-title { font-weight: 600; font-size: 1.05rem; display: flex; align-items: center; gap: 8px;}
.anomaly-amount { font-weight: 700; font-size: 1.2rem; color: #F8FAFC; }
.anomaly-icon { font-size: 1.2rem; }

.anomaly-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 1rem;
    background: #0F172A;
    padding: 0.8rem;
    border-radius: 8px;
    margin-top: 0.5rem;
}
.grid-item { display: flex; flex-direction: column; }
.grid-label { font-size: 0.75rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 2px;}
.grid-val { font-size: 0.95rem; font-weight: 500; color: #E2E8F0; }
.deviation-bad { color: #EF4444; font-weight: 600; }

.anomaly-reason {
    margin-top: 0.8rem;
    font-size: 0.9rem;
    color: #CBD5E1;
    border-top: 1px solid #334155;
    padding-top: 0.8rem;
    line-height: 1.5;
}

/* What Changed Insights */
.insight-panel {
    background: linear-gradient(to right, #1E293B, #0F172A);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1rem;
}
.insight-panel-title {
    font-weight: 600; font-size: 1.1rem; color: #F8FAFC; margin-bottom: 1rem; display: flex; align-items: center; gap: 6px;
}
.insight-listItem { margin-bottom: 0.5rem; display: flex; align-items: start; gap: 8px;}
.insight-bullet { color: #6366F1; font-weight: bold; }

/* Existing insight cards */
.insight-warning { background: rgba(239, 68, 68, 0.1); border-left: 4px solid #EF4444; border-radius: 8px; padding: 1rem; margin-bottom: 0.8rem; }
.insight-positive { background: rgba(34, 197, 94, 0.1); border-left: 4px solid #22C55E; border-radius: 8px; padding: 1rem; margin-bottom: 0.8rem; }
.insight-suggestion { background: rgba(99, 102, 241, 0.1); border-left: 4px solid #6366F1; border-radius: 8px; padding: 1rem; margin-bottom: 0.8rem; }
.insight-prediction { background: rgba(245, 158, 11, 0.1); border-left: 4px solid #F59E0B; border-radius: 8px; padding: 1rem; margin-bottom: 0.8rem; }
.insight-title { font-weight: 600; font-size: 0.95rem; margin-bottom: 0.3rem; }
.insight-msg { font-size: 0.88rem; color: #CBD5E1; }
.insight-tip { font-size: 0.82rem; color: #94A3B8; margin-top: 0.3rem; font-style: italic; }

/* Section headers */
.section-header {
    font-size: 1.3rem; font-weight: 600; margin-bottom: 1rem; padding-bottom: 0.5rem;
    border-bottom: 1px solid #334155; color: #F8FAFC;
}

/* Sidebar */
.sidebar-logo { text-align: center; padding: 1rem 0 2rem 0; font-size: 1.4rem; font-weight: 700; color: #6366F1; }
div[data-testid="stMetricValue"] { font-size: 1.8rem !important; }
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
        if data_source == "Upload Your CSV":
            uploaded_file = st.file_uploader(
                "Upload CSV",
                type=["csv"],
                help="Required columns: date, amount, category, merchant",
            )
            st.caption("Required columns: `date`, `amount`, `category`, `merchant`")

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
        df_raw = pd.read_csv(uploaded_file)
    else:
        if not os.path.exists(sample_path):
            st.error("Sample data not found. Please generate it or upload a CSV.")
            st.stop()
        df_raw = pd.read_csv(sample_path)

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
        font=dict(color="#FAFAFA", family="Inter"),
        xaxis=dict(gridcolor="#2D3361", showgrid=True, zeroline=False),
        yaxis=dict(gridcolor="#2D3361", showgrid=True, zeroline=False),
        margin=dict(l=10, r=10, t=40, b=10),
    )


CATEGORY_COLORS = {
    "food": "#FF6B6B",
    "shopping": "#6C63FF",
    "transport": "#48C9B0",
    "bills": "#FFA726",
    "entertainment": "#EC407A",
    "health": "#66BB6A",
    "education": "#42A5F5",
    "rent": "#AB47BC",
    "uncategorized": "#78909C",
}

SEVERITY_COLORS = {
    "critical": "#FF5252",
    "warning": "#FFA726",
    "normal": "#66BB6A",
}


# ─── PAGE 1: Overview ─────────────────────────────────────────────────────────
def page_overview(df: pd.DataFrame, profile: UserProfile, scorer: HealthScorer, insights_gen, currency: str):
    st.markdown("## 📊 Overview")
    st.divider()

    # 1. Hero Section (Health Score)
    health_score = scorer.total_score
    grade = scorer._get_grade()
    color = "#22C55E" if health_score >= 75 else ("#F59E0B" if health_score >= 50 else "#EF4444")
    status_text = "Stable" if health_score >= 75 else ("Needs Attention" if health_score >= 50 else "At Risk")

    hero_col1, hero_col2 = st.columns([1, 4])
    with hero_col1:
        st.markdown(f"""
        <div style="background: {color}20; border: 2px solid {color}; border-radius: 50%; width: 120px; height: 120px; display: flex; align-items: center; justify-content: center; flex-direction: column;">
            <span style="font-size: 2.5rem; font-weight: 700; color: {color}; line-height: 1;">{health_score:.0f}</span>
            <span style="font-size: 0.9rem; font-weight: 600; color: {color}; text-transform: uppercase;">{grade}</span>
        </div>
        """, unsafe_allow_html=True)
    with hero_col2:
        st.markdown(f"<h3 style='margin-bottom: 0.2rem;'>Financial Health: {status_text}</h3>", unsafe_allow_html=True)
        # Fetch top insight string for quick context
        top_insight = insights_gen.get_top_insights(1)
        if top_insight:
            st.markdown(f"<p style='color: #94A3B8; font-size: 1.1rem;'>{top_insight[0]['title']} — {top_insight[0]['suggestion']}</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color: #94A3B8; font-size: 1.1rem;'>Your spending patterns are stable and within normal baseline.</p>", unsafe_allow_html=True)

    # 2. Priority Section (Anomaly Alerts) - Dominant if exist
    anomalies = df[df["is_anomaly"] == 1]
    if len(anomalies) > 0:
        st.markdown('<p class="section-header" style="margin-top: 2.5rem;">🔴 Anomalies & Alerts</p>', unsafe_allow_html=True)
        recent_anomalies = anomalies.sort_values(by=["anomaly_score", "date"], ascending=[False, False]).head(5)
        
        cards_html = []
        for _, row in recent_anomalies.iterrows():
            sev = row.get("anomaly_severity", "warning").lower()
            if sev == "critical":
                sev_class = "anomaly-critical"
                icon = "🚨 (Critical Deviation)"
            elif sev == "warning":
                sev_class = "anomaly-warning"
                icon = "⚠️ (Warning)"
            else:
                sev_class = "anomaly-mild"
                icon = "ℹ️ (Mild)"
            
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
        <div class="anomaly-title" style="color: #F8FAFC;">{icon} &nbsp; {cat} at {merch}</div>
        <div class="anomaly-amount">{amount_fmt}</div>
    </div>
    <div style="color: #94A3B8; font-size: 0.85rem;">{date_str}</div>
    <div class="anomaly-grid">
        <div class="grid-item">
            <span class="grid-label">Typical Behavior</span>
            <span class="grid-val">{typical}</span>
        </div>
        <div class="grid-item">
            <span class="grid-label">Deviation</span>
            <span class="grid-val deviation-bad">{deviation}</span>
        </div>
    </div>
    <div class="anomaly-reason">
        <strong>Reason:</strong> {reason}
    </div>
</div>""")

        st.markdown("".join(cards_html), unsafe_allow_html=True)

    # 3. What Changed (Insight Panel)
    st.markdown('<p class="section-header" style="margin-top: 2.5rem;">⚡ What Changed Recently</p>', unsafe_allow_html=True)
    insights_list = insights_gen.get_insights()
    if insights_list:
        st.markdown('<div class="insight-panel">', unsafe_allow_html=True)
        for ins in insights_list[:4]:  # Top 4 insights
            st.markdown(f'''<div class="insight-listItem">
    <span class="insight-bullet">→</span> 
    <span style="color: #E2E8F0; font-size: 1rem;">
        <strong>{ins["title"]}:</strong> {ins["message"]}
    </span>
</div>''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No significant behavioral changes detected.")

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
            marker_color="rgba(99, 102, 241, 0.25)",
            hovertemplate=f"<b>%{{x}}</b><br>{currency}%{{y:,.0f}}<extra></extra>",
        ))
        fig_trend.add_trace(go.Scatter(
            x=daily_spend["date"], y=daily_spend["ewma"], name="Smoothed Trend",
            line=dict(color="#6366F1", width=3), mode="lines",
            hovertemplate=f"<b>%{{x}}</b><br>Trend: {currency}%{{y:,.0f}}<extra></extra>",
        ))
        fig_trend.add_trace(go.Scatter(
            x=forecast_df["date"], y=forecast_df["forecast"], name="7-Day Forecast",
            line=dict(color="#F59E0B", width=2, dash="dot"), mode="lines",
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
    st.caption("Drill into every flagged transaction and understand exactly why it was detected.")
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
        st.metric("Critical", critical, delta="Needs attention", delta_color="inverse")
    with col3:
        warning = len(anomalies[anomalies["anomaly_severity"] == "warning"])
        st.metric("Warnings", warning)
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
            name="Normal",
            marker=dict(color="rgba(102,187,106,0.3)", size=5),
            hoverinfo="skip",
        ))

        # Anomalies — color by severity
        for severity, color in [("critical", "#FF5252"), ("warning", "#FFA726")]:
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
            line_dash="dash", line_color="#6C63FF",
            annotation_text="Anomaly threshold", annotation_font_size=11,
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
                st.metric("Anomaly Score", f"{row['anomaly_score']:.3f}")
                st.metric("Severity", sev.upper())


# ─── PAGE 3: Trends ───────────────────────────────────────────────────────────
def page_trends(df: pd.DataFrame, currency: str):
    st.markdown("## 📈 Spending Trends")
    st.divider()

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
    page, uploaded_file, currency_symbol, contamination = render_sidebar()

    # Load data
    df_raw = load_data(uploaded_file, currency_symbol, contamination)

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

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =============================================================================
# APP CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Quotation Pricing Decision Support",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_VERSION = "v2.0.0"
LAST_UPDATED = "2026-06-11"
DATA_PATH = Path(__file__).with_name("df_preprocessed.csv")

GMR_BINS = [-np.inf, 0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, np.inf]
GMR_LABELS = ["< 0%", "0–10%", "10–20%", "20–30%", "30–40%", "40–50%", "50–60%", "> 60%"]

MODEL_FEATURES = [
    "product",
    "kw",
    "qty",
    "unit_price",
    "gross_margin_rate",
    "energy_grant_amount",
    "estimated_cost",
    "grant_ratio_to_subtotal",
    "competitor_count_available",
    "avg_competitor_price",
    "min_competitor_price",
    "price_gap_avg_competitor_pct",
    "price_gap_min_competitor_pct",
]

NUMERIC_FEATURES = [feature for feature in MODEL_FEATURES if feature != "product"]
CATEGORICAL_FEATURES = ["product"]

# Color palette used consistently throughout the dashboard.
COLORS = {
    "navy": "#1D3557",
    "blue": "#2563EB",
    "sky": "#38BDF8",
    "teal": "#14B8A6",
    "green": "#22C55E",
    "amber": "#F59E0B",
    "orange": "#F97316",
    "red": "#EF4444",
    "purple": "#8B5CF6",
    "slate": "#64748B",
    "light_blue": "#EFF6FF",
    "light_slate": "#F8FAFC",
}

# =============================================================================
# GLOBAL CSS
# =============================================================================
st.markdown(
    """
    <style>
    :root {
        --navy: #1D3557;
        --blue: #2563EB;
        --teal: #14B8A6;
        --green: #22C55E;
        --amber: #F59E0B;
        --red: #EF4444;
        --purple: #8B5CF6;
        --slate: #64748B;
        --text-dark: #172033;
        --text-muted: #667085;
        --border: #E5E7EB;
        --soft-bg: #F7F9FC;
    }

    .stApp {
        background: #F5F7FB;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.15rem;
        padding-bottom: 2.5rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1D3557 0%, #274C77 100%);
    }

    section[data-testid="stSidebar"] * {
        color: #FFFFFF;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        border-radius: 10px;
        padding: 0.35rem 0.45rem;
        margin-bottom: 0.18rem;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: rgba(255, 255, 255, 0.11);
    }

    /* Main typography */
    .page-title {
        margin: 0;
        color: #172033;
        font-size: 2.35rem;
        font-weight: 800;
        line-height: 1.08;
        letter-spacing: -0.04em;
    }

    .page-subtitle {
        margin-top: 0.42rem;
        margin-bottom: 1rem;
        color: #667085;
        font-size: 0.98rem;
        line-height: 1.55;
    }

    .eyebrow {
        display: inline-block;
        margin-bottom: 0.32rem;
        color: #2563EB;
        font-size: 0.74rem;
        font-weight: 800;
        letter-spacing: 0.11em;
        text-transform: uppercase;
    }

    .section-title {
        color: #172033;
        font-size: 1.22rem;
        font-weight: 800;
        margin: 0.05rem 0 0.15rem 0;
    }

    .section-subtitle {
        color: #667085;
        font-size: 0.91rem;
        line-height: 1.48;
        margin-bottom: 0.62rem;
    }

    /* Streamlit bordered containers */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.98);
        border: 1px solid #E6EAF0 !important;
        border-radius: 18px !important;
        box-shadow: 0 8px 20px rgba(16, 24, 40, 0.045);
        padding: 0.35rem 0.4rem;
    }

    /* Custom KPI cards */
    .metric-card {
        min-height: 116px;
        border: 1px solid #E6EAF0;
        border-radius: 16px;
        padding: 0.95rem 1rem;
        background: linear-gradient(145deg, #FFFFFF 0%, #FBFDFF 100%);
        box-shadow: 0 5px 14px rgba(16, 24, 40, 0.035);
    }

    .metric-topline {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.4rem;
    }

    .metric-label {
        color: #667085;
        font-size: 0.82rem;
        font-weight: 700;
        line-height: 1.25;
    }

    .metric-icon {
        width: 30px;
        height: 30px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 10px;
        font-size: 1rem;
        background: #EFF6FF;
    }

    .metric-value {
        margin-top: 0.55rem;
        color: #172033;
        font-size: 1.8rem;
        font-weight: 850;
        letter-spacing: -0.035em;
        line-height: 1;
    }

    .metric-footnote {
        margin-top: 0.48rem;
        color: #667085;
        font-size: 0.76rem;
        line-height: 1.35;
    }

    .accent-blue {border-top: 4px solid #2563EB;}
    .accent-teal {border-top: 4px solid #14B8A6;}
    .accent-green {border-top: 4px solid #22C55E;}
    .accent-purple {border-top: 4px solid #8B5CF6;}
    .accent-amber {border-top: 4px solid #F59E0B;}
    .accent-red {border-top: 4px solid #EF4444;}
    .accent-slate {border-top: 4px solid #64748B;}

    .insight-box {
        border-left: 5px solid #2563EB;
        border-radius: 12px;
        background: #EFF6FF;
        color: #1E3A8A;
        padding: 0.88rem 1rem;
        font-size: 0.9rem;
        line-height: 1.48;
    }

    .note-card {
        border: 1px solid #E6EAF0;
        border-radius: 14px;
        background: #FBFCFE;
        padding: 0.86rem 0.92rem;
        color: #475467;
        font-size: 0.88rem;
        line-height: 1.5;
    }

    .note-card ul,
    .note-card ol {
        margin-bottom: 0.15rem;
        padding-left: 1.3rem;
    }

    .note-card li {
        margin-bottom: 0.42rem;
    }

    .review-flag {
        border-radius: 13px;
        padding: 0.78rem 0.9rem;
        margin-bottom: 0.56rem;
        background: #FFFFFF;
        border: 1px solid #E6EAF0;
        font-size: 0.88rem;
        line-height: 1.45;
    }

    .risk-high {border-left: 5px solid #EF4444;}
    .risk-medium {border-left: 5px solid #F59E0B;}
    .risk-low {border-left: 5px solid #22C55E;}

    .small-note {
        color: #667085;
        font-size: 0.82rem;
        line-height: 1.42;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.4rem;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding-left: 0.85rem;
        padding-right: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# DATA PREPARATION AND MODELLING
# =============================================================================
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Load the preprocessed data and apply modelling safeguards."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH.name}. Place df_preprocessed.csv in the same folder as app.py."
        )

    df = pd.read_csv(DATA_PATH)

    # Quantity <= 0 was confirmed as a system error during the interview.
    df = df[df["qty"] > 0].copy()

    # For the same Quote ID and product, the expert suggested keeping the
    # highest unit price when price-list updates create multiple records.
    if "is_highest_price" in df.columns:
        df = df[df["is_highest_price"] == 1].copy()

    df = df.drop_duplicates().copy()
    df["is_win"] = (df["convert_to_order"] == 0).astype(int)
    df["gmr_pct"] = df["gross_margin_rate"] * 100
    df["gmr_band"] = pd.cut(df["gross_margin_rate"], bins=GMR_BINS, labels=GMR_LABELS)
    df["estimated_cost_per_unit"] = df["estimated_cost"] / df["qty"]
    df["multi_product_quote"] = df.groupby("quote_id")["product"].transform("nunique") > 1
    return df


def create_pipeline() -> Pipeline:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, NUMERIC_FEATURES),
            ("categorical", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )
    classifier = LogisticRegression(
        max_iter=1500,
        class_weight="balanced",
        C=0.5,
        random_state=42,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


@st.cache_resource(show_spinner=False)
def train_model() -> Tuple[Pipeline, Dict[str, float], List[List[int]]]:
    """Train an explainable baseline using a Quote-ID grouped holdout split."""
    data = load_data()
    X = data[MODEL_FEATURES]
    y = data["is_win"]
    groups = data["quote_id"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    train_data = data.iloc[train_idx]
    train_quote_counts = train_data.groupby("quote_id").size()
    train_weights = train_data["quote_id"].map(lambda quote: 1.0 / train_quote_counts.loc[quote]).to_numpy()

    eval_model = create_pipeline()
    eval_model.fit(X.iloc[train_idx], y.iloc[train_idx], classifier__sample_weight=train_weights)
    probabilities = eval_model.predict_proba(X.iloc[test_idx])[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    metrics = {
        "ROC-AUC": roc_auc_score(y.iloc[test_idx], probabilities),
        "PR-AUC": average_precision_score(y.iloc[test_idx], probabilities),
        "Accuracy": accuracy_score(y.iloc[test_idx], predictions),
        "Precision (Win)": precision_score(y.iloc[test_idx], predictions, zero_division=0),
        "Recall (Win)": recall_score(y.iloc[test_idx], predictions, zero_division=0),
        "F1 (Win)": f1_score(y.iloc[test_idx], predictions, zero_division=0),
    }
    cm = confusion_matrix(y.iloc[test_idx], predictions).tolist()

    # Refit on all available records, weighting each Quote ID equally.
    all_quote_counts = data.groupby("quote_id").size()
    all_weights = data["quote_id"].map(lambda quote: 1.0 / all_quote_counts.loc[quote]).to_numpy()
    final_model = create_pipeline()
    final_model.fit(X, y, classifier__sample_weight=all_weights)
    return final_model, metrics, cm


def product_defaults(data: pd.DataFrame, product: str) -> Dict[str, float]:
    subset = data[data["product"] == product].copy()
    comp_a = subset["competitor_a"].dropna()
    comp_b = subset["competitor_b"].dropna()
    comp_c = subset["competitor_c"].dropna()
    return {
        "kw": float(subset["kw"].median()),
        "cost_per_unit": float(subset["estimated_cost_per_unit"].median()),
        "grant": float(subset["energy_grant_amount"].median()),
        "competitor_a": float(comp_a.median()) if not comp_a.empty else np.nan,
        "competitor_b": float(comp_b.median()) if not comp_b.empty else np.nan,
        "competitor_c": float(comp_c.median()) if not comp_c.empty else np.nan,
    }


def available_competitors(competitor_a: float, competitor_b: float, competitor_c: float) -> List[float]:
    values = [competitor_a, competitor_b, competitor_c]
    return [float(value) for value in values if pd.notna(value) and float(value) > 0]


def build_scenario(
    product: str,
    kw: float,
    qty: int,
    estimated_cost_per_unit: float,
    gross_margin_rate: float,
    energy_grant_amount: float,
    competitor_a: float = np.nan,
    competitor_b: float = np.nan,
    competitor_c: float = np.nan,
) -> pd.DataFrame:
    if gross_margin_rate >= 0.95:
        raise ValueError("Gross margin rate must be below 95%.")

    unit_price = estimated_cost_per_unit / (1.0 - gross_margin_rate)
    subtotal_price = unit_price * qty
    total_cost = estimated_cost_per_unit * qty
    competitors = available_competitors(competitor_a, competitor_b, competitor_c)
    count = len(competitors)
    avg_competitor = float(np.mean(competitors)) if competitors else np.nan
    min_competitor = float(np.min(competitors)) if competitors else np.nan

    price_gap_avg_pct = ((unit_price - avg_competitor) / avg_competitor) * 100 if competitors else np.nan
    price_gap_min_pct = ((unit_price - min_competitor) / min_competitor) * 100 if competitors else np.nan

    row = {
        "product": product,
        "kw": float(kw),
        "qty": int(qty),
        "unit_price": float(unit_price),
        "gross_margin_rate": float(gross_margin_rate),
        "energy_grant_amount": float(energy_grant_amount),
        "estimated_cost": float(total_cost),
        "grant_ratio_to_subtotal": float(energy_grant_amount / subtotal_price) if subtotal_price else np.nan,
        "competitor_count_available": int(count),
        "avg_competitor_price": avg_competitor,
        "min_competitor_price": min_competitor,
        "price_gap_avg_competitor_pct": price_gap_avg_pct,
        "price_gap_min_competitor_pct": price_gap_min_pct,
        "subtotal_price": float(subtotal_price),
        "gross_profit_amount": float(subtotal_price - total_cost),
    }
    return pd.DataFrame([row])


def score_scenario(model: Pipeline, scenario: pd.DataFrame) -> Dict[str, float]:
    probability = float(model.predict_proba(scenario[MODEL_FEATURES])[:, 1][0])
    gross_profit = float(scenario["gross_profit_amount"].iloc[0])
    expected_gross_profit = probability * gross_profit
    return {
        "win_probability": probability,
        "gross_profit_amount": gross_profit,
        "expected_gross_profit": expected_gross_profit,
    }


def recommend_margin(
    model: Pipeline,
    product: str,
    kw: float,
    qty: int,
    estimated_cost_per_unit: float,
    energy_grant_amount: float,
    competitor_a: float,
    competitor_b: float,
    competitor_c: float,
    minimum_margin: float,
    maximum_margin: float = 0.70,
) -> pd.DataFrame:
    records = []
    for gmr in np.arange(0.00, maximum_margin + 0.001, 0.01):
        scenario = build_scenario(
            product=product,
            kw=kw,
            qty=qty,
            estimated_cost_per_unit=estimated_cost_per_unit,
            gross_margin_rate=float(gmr),
            energy_grant_amount=energy_grant_amount,
            competitor_a=competitor_a,
            competitor_b=competitor_b,
            competitor_c=competitor_c,
        )
        scored = score_scenario(model, scenario)
        records.append(
            {
                "Gross Margin Rate": gmr,
                "Unit Price": float(scenario["unit_price"].iloc[0]),
                "Win Probability": scored["win_probability"],
                "Gross Profit if Won": scored["gross_profit_amount"],
                "Expected Gross Profit": scored["expected_gross_profit"],
                "Meets Minimum Margin": gmr >= minimum_margin,
            }
        )
    return pd.DataFrame(records)


def historical_benchmark(data: pd.DataFrame, product: str, gmr: float) -> Dict[str, float]:
    band = pd.cut(pd.Series([gmr]), bins=GMR_BINS, labels=GMR_LABELS).iloc[0]
    subset = data[(data["product"] == product) & (data["gmr_band"] == band)].copy()
    subset = subset.drop_duplicates(subset=["quote_id"])
    if subset.empty:
        return {"band": str(band), "n": 0, "historical_win_rate": np.nan}
    return {
        "band": str(band),
        "n": int(len(subset)),
        "historical_win_rate": float(subset["is_win"].mean()),
    }


def classify_price_position(unit_price: float, min_competitor: float) -> str:
    if pd.isna(min_competitor):
        return "Competitor information unavailable"
    ratio = (unit_price - min_competitor) / min_competitor
    if ratio < -0.05:
        return "Below competitor benchmark"
    if ratio <= 0.05:
        return "Approximately aligned (±5%)"
    return "Above competitor benchmark"


def risk_flags(
    scenario: pd.DataFrame,
    score: Dict[str, float],
    minimum_margin: float,
    minimum_win_probability: float,
) -> List[Tuple[str, str, str]]:
    row = scenario.iloc[0]
    flags: List[Tuple[str, str, str]] = []
    gmr = float(row["gross_margin_rate"])

    if gmr < 0:
        flags.append(("High", "Profitability risk", "Negative margin: manager approval is required before submission."))
    elif gmr < minimum_margin:
        flags.append(("Medium", "Margin review", f"Margin is below the selected minimum target of {minimum_margin:.0%}."))

    if score["win_probability"] < minimum_win_probability:
        flags.append(
            (
                "Medium",
                "Conversion risk",
                f"Estimated win probability is below the review threshold of {minimum_win_probability:.0%}.",
            )
        )

    if int(row["competitor_count_available"]) == 0:
        flags.append(
            (
                "Medium",
                "Information gap",
                "No competitor benchmark is available. Review the recommendation with additional sales information.",
            )
        )
    elif pd.notna(row["price_gap_min_competitor_pct"]) and float(row["price_gap_min_competitor_pct"]) > 5:
        flags.append(
            (
                "High",
                "Competitor positioning risk",
                f"Quoted unit price is {row['price_gap_min_competitor_pct']:.1f}% above the minimum available competitor benchmark.",
            )
        )

    if not flags:
        flags.append(("Low", "No major rule-based warning", "The quotation does not trigger the current review rules."))
    return flags


def potential_loss_signal(row: pd.Series, product_median: Dict[str, float]) -> str:
    if pd.notna(row.get("price_gap_min_competitor_pct")) and row["price_gap_min_competitor_pct"] > 5:
        return "Above competitor benchmark"
    if row["gross_margin_rate"] > max(0.40, product_median.get(row["product"], 0.40)):
        return "High margin for product"
    if row["competitor_count_available"] == 0:
        return "Competitor information unavailable"
    return "Other factors not captured"


# =============================================================================
# UI HELPERS
# =============================================================================
def currency(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:,.0f}"


def format_percent(value: float, decimals: int = 1) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}%}"


def render_page_header(title: str, subtitle: str) -> None:
    st.markdown("<span class='eyebrow'>Group E · Quantitative Analytics</span>", unsafe_allow_html=True)
    st.markdown(f"<h1 class='page-title'>{escape(title)}</h1>", unsafe_allow_html=True)
    st.markdown(f"<div class='page-subtitle'>{escape(subtitle)}</div>", unsafe_allow_html=True)


def render_section_heading(title: str, subtitle: str = "") -> None:
    st.markdown(f"<div class='section-title'>{escape(title)}</div>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<div class='section-subtitle'>{escape(subtitle)}</div>", unsafe_allow_html=True)


def render_metric_card(label: str, value: str, footnote: str, icon: str, accent: str = "blue") -> None:
    safe_accent = accent if accent in {"blue", "teal", "green", "purple", "amber", "red", "slate"} else "blue"
    st.markdown(
        f"""
        <div class="metric-card accent-{safe_accent}">
            <div class="metric-topline">
                <div class="metric-label">{escape(label)}</div>
                <div class="metric-icon">{escape(icon)}</div>
            </div>
            <div class="metric-value">{escape(value)}</div>
            <div class="metric-footnote">{escape(footnote)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_insight(text: str) -> None:
    st.markdown(f"<div class='insight-box'>{escape(text)}</div>", unsafe_allow_html=True)


def render_note_html(content: str) -> None:
    st.markdown(f"<div class='note-card'>{content}</div>", unsafe_allow_html=True)


def apply_chart_layout(fig: go.Figure, *, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(family="Arial, sans-serif", color="#344054", size=12),
        margin=dict(l=15, r=15, t=28, b=15),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(showgrid=False, linecolor="#E5E7EB")
    fig.update_yaxes(gridcolor="#EEF2F6", zerolinecolor="#E5E7EB")
    return fig


def render_title() -> None:
    render_page_header(
        "Quotation Pricing Decision Support Dashboard",
        "Interactive price-review support for gross-margin management, competitor positioning, win-probability estimation, and quotation risk review.",
    )


# =============================================================================
# LOAD SHARED RESOURCES
# =============================================================================
try:
    data = load_data()
    model, model_metrics, confusion = train_model()
except Exception as exc:
    st.error(f"The dashboard could not be loaded: {exc}")
    st.stop()


# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================
with st.sidebar:
    st.markdown("## 📊 QABD Dashboard")
    st.caption("Quotation Pricing Decision Support")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Executive Overview",
            "🎯 Quote Recommender",
            "📈 Margin Sweet Spot",
            "⚖️ Competitor Positioning",
            "🧪 Data Quality & Model Notes",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Decision-support scope**")
    st.caption(
        "This prototype supports quotation price review. It does not replace managerial approval, customer knowledge, or commercial judgement."
    )
    st.markdown("---")
    st.caption(f"Version: {APP_VERSION}")
    st.caption(f"Updated: {LAST_UPDATED}")


# =============================================================================
# EXECUTIVE OVERVIEW
# =============================================================================
if page == "🏠 Executive Overview":
    render_title()

    quote_header = data.drop_duplicates(subset=["quote_id"]).copy()

    with st.container(border=True):
        render_section_heading(
            "Executive Overview",
            "A high-level summary of historical quotation performance, product-level conversion patterns, and rule-based review signals.",
        )
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_metric_card(
                "Historical quotations",
                f"{quote_header['quote_id'].nunique():,}",
                "Unique quotation records",
                "🧾",
                "blue",
            )
        with c2:
            render_metric_card(
                "Product categories",
                f"{data['product'].nunique():,}",
                "Distinct anonymized product codes",
                "📦",
                "purple",
            )
        with c3:
            render_metric_card(
                "Historical win rate",
                f"{quote_header['is_win'].mean():.1%}",
                "Quotation-level conversion success",
                "🏆",
                "green",
            )
        with c4:
            render_metric_card(
                "Rows with competitor info",
                f"{(data['competitor_count_available'] > 0).mean():.1%}",
                "At least one competitor benchmark",
                "🔎",
                "amber",
            )
        st.markdown("")
        render_insight(
            "The dashboard balances quotation profitability with tender success while considering product characteristics, energy grants, and available competitor-price benchmarks."
        )

    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            render_section_heading(
                "Historical win rate by gross-margin band",
                "Quotation-level conversion performance across historical gross-margin ranges.",
            )
            margin_summary = (
                quote_header.groupby("gmr_band", observed=False)
                .agg(Quotations=("quote_id", "nunique"), Win_Rate=("is_win", "mean"))
                .reset_index()
            )
            fig = px.bar(
                margin_summary,
                x="gmr_band",
                y="Win_Rate",
                text=margin_summary["Win_Rate"].map(lambda value: f"{value:.1%}"),
                color="gmr_band",
                color_discrete_sequence=[
                    COLORS["blue"],
                    COLORS["sky"],
                    COLORS["teal"],
                    COLORS["green"],
                    COLORS["amber"],
                    COLORS["orange"],
                    COLORS["red"],
                    COLORS["purple"],
                ],
                hover_data={"Quotations": True, "Win_Rate": ":.1%"},
                labels={"gmr_band": "Gross Margin Rate", "Win_Rate": "Historical Win Rate"},
            )
            fig.update_traces(marker_line_color="#FFFFFF", marker_line_width=1.1, textposition="outside")
            fig.update_layout(showlegend=False, yaxis_tickformat=".0%")
            apply_chart_layout(fig, height=365)
            st.plotly_chart(fig, use_container_width=True)

    with right:
        with st.container(border=True):
            render_section_heading(
                "Product-level win rate",
                "Only products with a minimum of 20 historical observations are shown.",
            )
            product_summary = (
                data.drop_duplicates(subset=["quote_id", "product"])
                .groupby("product")
                .agg(Observations=("quote_id", "nunique"), Win_Rate=("is_win", "mean"))
                .reset_index()
            )
            product_summary = product_summary[product_summary["Observations"] >= 20].sort_values("Win_Rate")
            fig = px.bar(
                product_summary,
                x="Win_Rate",
                y="product",
                orientation="h",
                text=product_summary["Win_Rate"].map(lambda value: f"{value:.1%}"),
                color="Win_Rate",
                color_continuous_scale=["#DBEAFE", "#38BDF8", "#0F766E"],
                hover_data={"Observations": True, "Win_Rate": ":.1%"},
                labels={"product": "Product", "Win_Rate": "Historical Win Rate"},
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(coloraxis_showscale=False, xaxis_tickformat=".0%")
            apply_chart_layout(fig, height=365)
            st.plotly_chart(fig, use_container_width=True)

    with st.container(border=True):
        render_section_heading(
            "Potential lost-tender review signals",
            "These are descriptive rule-based signals for reviewing unsuccessful product lines. They are not confirmed causal reasons.",
        )
        product_median = data.groupby("product")["gross_margin_rate"].median().to_dict()
        failed = data[data["is_win"] == 0].copy()
        failed["Potential review signal"] = failed.apply(
            lambda row: potential_loss_signal(row, product_median), axis=1
        )
        signal_summary = (
            failed["Potential review signal"]
            .value_counts()
            .rename_axis("Potential review signal")
            .reset_index(name="Failed product lines")
        )
        fig = px.bar(
            signal_summary,
            x="Potential review signal",
            y="Failed product lines",
            text_auto=True,
            color="Potential review signal",
            color_discrete_map={
                "Above competitor benchmark": COLORS["red"],
                "High margin for product": COLORS["amber"],
                "Competitor information unavailable": COLORS["purple"],
                "Other factors not captured": COLORS["slate"],
            },
        )
        fig.update_traces(marker_line_color="#FFFFFF", marker_line_width=1.0, textposition="outside")
        fig.update_layout(showlegend=False)
        apply_chart_layout(fig, height=365)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Additional customer and sales-process data would be required before interpreting these signals as the reasons for tender losses."
        )


# =============================================================================
# QUOTE RECOMMENDER
# =============================================================================
elif page == "🎯 Quote Recommender":
    render_title()

    with st.container(border=True):
        render_section_heading(
            "Interactive Quote Recommender",
            "Enter a quotation scenario to estimate win probability, compare the proposed price with available competitor benchmarks, and identify a margin recommendation that balances conversion chance with expected gross profit.",
        )

        products = sorted(data["product"].astype(str).unique().tolist())
        input_left, input_mid, input_right = st.columns(3)

        with input_left:
            product = st.selectbox("Product", products, index=products.index("H1") if "H1" in products else 0)
            defaults = product_defaults(data, product)
            kw = st.number_input("Power rating (kW)", min_value=0.1, value=float(defaults["kw"]), step=0.5)
            qty = st.number_input("Quantity", min_value=1, value=1, step=1)
            cost_per_unit = st.number_input(
                "Estimated cost per unit",
                min_value=1.0,
                value=float(round(defaults["cost_per_unit"], 2)),
                step=1000.0,
            )

        with input_mid:
            selected_gmr_pct = st.slider("Proposed gross margin rate", min_value=-10, max_value=80, value=30, step=1)
            energy_grant = st.number_input(
                "Energy grant amount",
                min_value=0.0,
                value=float(round(defaults["grant"], 2)),
                step=1000.0,
                help="The historical data indicate product-related energy-grant values. Adjust this input when reviewing a specific quotation.",
            )
            minimum_margin_pct = st.slider(
                "Minimum target margin for recommendation",
                min_value=0,
                max_value=50,
                value=20,
                step=1,
                help="The default is 20% based on the interview discussion. Lower-margin cases may still be reviewed by a supervisor.",
            )
            review_win_threshold_pct = st.slider("Win-probability review threshold", 10, 80, 40, 5)

        with input_right:
            has_competitor_info = st.checkbox("Use competitor price information", value=pd.notna(defaults["competitor_a"]))
            if has_competitor_info:
                comp_a_default = 0.0 if pd.isna(defaults["competitor_a"]) else defaults["competitor_a"]
                comp_b_default = 0.0 if pd.isna(defaults["competitor_b"]) else defaults["competitor_b"]
                comp_c_default = 0.0 if pd.isna(defaults["competitor_c"]) else defaults["competitor_c"]
                competitor_a = st.number_input("Competitor A price (0 = unavailable)", min_value=0.0, value=float(comp_a_default), step=1000.0)
                competitor_b = st.number_input("Competitor B price (0 = unavailable)", min_value=0.0, value=float(comp_b_default), step=1000.0)
                competitor_c = st.number_input("Competitor C price (0 = unavailable)", min_value=0.0, value=float(comp_c_default), step=1000.0)
            else:
                competitor_a = competitor_b = competitor_c = np.nan
            st.caption("Competitor fields are optional because competitor information is frequently unavailable in the historical dataset.")

    selected_gmr = selected_gmr_pct / 100
    minimum_margin = minimum_margin_pct / 100
    review_win_threshold = review_win_threshold_pct / 100

    try:
        scenario = build_scenario(
            product=product,
            kw=kw,
            qty=int(qty),
            estimated_cost_per_unit=cost_per_unit,
            gross_margin_rate=selected_gmr,
            energy_grant_amount=energy_grant,
            competitor_a=competitor_a,
            competitor_b=competitor_b,
            competitor_c=competitor_c,
        )
        selected_score = score_scenario(model, scenario)
        grid = recommend_margin(
            model=model,
            product=product,
            kw=kw,
            qty=int(qty),
            estimated_cost_per_unit=cost_per_unit,
            energy_grant_amount=energy_grant,
            competitor_a=competitor_a,
            competitor_b=competitor_b,
            competitor_c=competitor_c,
            minimum_margin=minimum_margin,
        )
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    feasible_grid = grid[grid["Meets Minimum Margin"]]
    best_row = feasible_grid.loc[feasible_grid["Expected Gross Profit"].idxmax()]
    benchmark = historical_benchmark(data, product, selected_gmr)
    unit_price = float(scenario["unit_price"].iloc[0])
    min_competitor = (
        float(scenario["min_competitor_price"].iloc[0])
        if pd.notna(scenario["min_competitor_price"].iloc[0])
        else np.nan
    )

    with st.container(border=True):
        render_section_heading(
            "Scenario Results",
            "The first row reflects the proposed quotation. The second row shows the model-based recommendation within the selected minimum-margin constraint.",
        )
        row1 = st.columns(4)
        with row1[0]:
            render_metric_card("Proposed unit price", currency(unit_price), "Calculated from cost and proposed GMR", "💵", "blue")
        with row1[1]:
            render_metric_card("Estimated win probability", f"{selected_score['win_probability']:.1%}", "Model-based probability estimate", "🎯", "green")
        with row1[2]:
            render_metric_card("Gross profit if won", currency(selected_score["gross_profit_amount"]), "Gross profit conditional on winning", "📈", "teal")
        with row1[3]:
            render_metric_card("Expected gross profit", currency(selected_score["expected_gross_profit"]), "Probability-adjusted gross profit", "⚖️", "purple")

        st.markdown("")
        row2 = st.columns(4)
        with row2[0]:
            render_metric_card("Recommended GMR", f"{best_row['Gross Margin Rate']:.0%}", "Highest expected GP within constraint", "✅", "green")
        with row2[1]:
            render_metric_card("Win probability at recommendation", f"{best_row['Win Probability']:.1%}", "Estimated chance at recommended margin", "🏆", "teal")
        with row2[2]:
            render_metric_card("Recommended unit price", currency(best_row["Unit Price"]), "Price implied by recommended GMR", "🏷️", "amber")
        with row2[3]:
            render_metric_card("Expected GP at recommendation", currency(best_row["Expected Gross Profit"]), "Maximum model-estimated expected GP", "💡", "purple")

        st.markdown("")
        render_insight(
            "The recommendation maximizes model-estimated expected gross profit among scenarios that meet the selected minimum-margin requirement. It should still be reviewed alongside sales knowledge and customer-specific considerations."
        )

    sim_left, sim_right = st.columns([1.55, 1.0])
    with sim_left:
        with st.container(border=True):
            render_section_heading(
                "What-if Margin Simulation",
                "Observe how win probability and expected gross profit change when the gross-margin rate is adjusted.",
            )
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=grid["Gross Margin Rate"] * 100,
                    y=grid["Win Probability"],
                    name="Estimated win probability",
                    mode="lines",
                    line=dict(color=COLORS["blue"], width=4),
                    yaxis="y1",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=grid["Gross Margin Rate"] * 100,
                    y=grid["Expected Gross Profit"],
                    name="Expected gross profit",
                    mode="lines",
                    line=dict(color=COLORS["purple"], width=4),
                    fill="tozeroy",
                    fillcolor="rgba(139, 92, 246, 0.10)",
                    yaxis="y2",
                )
            )
            fig.add_vline(x=selected_gmr_pct, line_dash="dash", line_color=COLORS["amber"], annotation_text="Proposed GMR")
            fig.add_vline(
                x=float(best_row["Gross Margin Rate"] * 100),
                line_dash="dot",
                line_color=COLORS["green"],
                annotation_text="Recommended GMR",
            )
            fig.update_layout(
                xaxis_title="Gross Margin Rate (%)",
                yaxis=dict(title="Estimated Win Probability", tickformat=".0%"),
                yaxis2=dict(title="Expected Gross Profit", overlaying="y", side="right"),
            )
            apply_chart_layout(fig, height=420)
            st.plotly_chart(fig, use_container_width=True)

    with sim_right:
        with st.container(border=True):
            render_section_heading(
                "Historical Benchmark",
                "A descriptive comparison with the selected product and margin band in the historical dataset.",
            )
            historical_rate = format_percent(benchmark["historical_win_rate"])
            st.markdown(
                f"""
                <div class="note-card">
                    <b>Selected product:</b> {escape(str(product))}<br><br>
                    <b>Gross-margin band:</b> {escape(str(benchmark['band']))}<br><br>
                    <b>Historical records in band:</b> {benchmark['n']:,}<br><br>
                    <b>Historical win rate:</b> {historical_rate}<br><br>
                    <b>Price position:</b> {escape(classify_price_position(unit_price, min_competitor))}
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption("Historical win rates are descriptive benchmarks and should not be interpreted as causal effects.")

    flag_left, flag_right = st.columns([1.0, 1.0])
    with flag_left:
        with st.container(border=True):
            render_section_heading(
                "Quote Review Flags",
                "Rule-based alerts help the sales or pricing reviewer identify scenarios that require additional attention.",
            )
            for severity, title, explanation in risk_flags(scenario, selected_score, minimum_margin, review_win_threshold):
                css = "risk-low" if severity == "Low" else ("risk-medium" if severity == "Medium" else "risk-high")
                st.markdown(
                    f"<div class='review-flag {css}'><b>{escape(severity)}: {escape(title)}</b><br>{escape(explanation)}</div>",
                    unsafe_allow_html=True,
                )

    with flag_right:
        with st.container(border=True):
            render_section_heading(
                "Competitor Positioning",
                "Compare the proposed unit price with the available competitor-price benchmarks.",
            )
            competitors = available_competitors(competitor_a, competitor_b, competitor_c)
            if competitors:
                competitor_values = [competitor_a, competitor_b, competitor_c]
                competitor_labels = [
                    f"Competitor {chr(65 + idx)}"
                    for idx, value in enumerate(competitor_values)
                    if pd.notna(value) and value > 0
                ]
                chart_data = pd.DataFrame(
                    {
                        "Price reference": ["Proposed unit price"] + competitor_labels,
                        "Price": [unit_price] + competitors,
                    }
                )
                fig = px.bar(
                    chart_data,
                    x="Price reference",
                    y="Price",
                    text_auto=",.0f",
                    color="Price reference",
                    color_discrete_sequence=[COLORS["blue"], COLORS["amber"], COLORS["purple"], COLORS["red"]],
                )
                fig.update_layout(showlegend=False)
                fig.update_traces(textposition="outside")
                apply_chart_layout(fig, height=330)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(
                    "No competitor benchmark is available for this scenario. Review the recommendation with additional market information."
                )

    with st.expander("View top margin scenarios"):
        display_grid = grid.sort_values("Expected Gross Profit", ascending=False).head(10).copy()
        display_grid["Gross Margin Rate"] = display_grid["Gross Margin Rate"].map(lambda value: f"{value:.0%}")
        display_grid["Win Probability"] = display_grid["Win Probability"].map(lambda value: f"{value:.1%}")
        display_grid["Unit Price"] = display_grid["Unit Price"].map(currency)
        display_grid["Gross Profit if Won"] = display_grid["Gross Profit if Won"].map(currency)
        display_grid["Expected Gross Profit"] = display_grid["Expected Gross Profit"].map(currency)
        st.dataframe(display_grid, use_container_width=True, hide_index=True)


# =============================================================================
# MARGIN SWEET SPOT
# =============================================================================
elif page == "📈 Margin Sweet Spot":
    render_title()

    with st.container(border=True):
        render_section_heading(
            "Gross-Margin Sweet Spot Analysis",
            "Compare historical conversion performance and expected gross-profit proxy across gross-margin ranges. Product-specific analysis is recommended because one universal threshold may not fit every product.",
        )
        choices = ["All products"] + sorted(data["product"].astype(str).unique().tolist())
        selected_product = st.selectbox("Filter by product", choices)

    if selected_product == "All products":
        subset = data.drop_duplicates(subset=["quote_id"]).copy()
    else:
        subset = data[data["product"] == selected_product].drop_duplicates(subset=["quote_id"]).copy()

    summary = (
        subset.groupby("gmr_band", observed=False)
        .agg(
            Quotations=("quote_id", "nunique"),
            Historical_Win_Rate=("is_win", "mean"),
            Average_Gross_Profit=("estimated_gross_profit", "mean"),
        )
        .reset_index()
    )
    summary["Expected_GP_Proxy"] = summary["Historical_Win_Rate"] * summary["Average_Gross_Profit"]

    best_win_idx = summary["Historical_Win_Rate"].fillna(-1).idxmax()
    best_gp_idx = summary["Expected_GP_Proxy"].fillna(-np.inf).idxmax()

    with st.container(border=True):
        render_section_heading("Sweet-Spot Summary", "Key descriptive indicators for the selected product filter.")
        top1, top2, top3 = st.columns(3)
        with top1:
            render_metric_card(
                "Best historical win-rate band",
                str(summary.loc[best_win_idx, "gmr_band"]),
                f"Historical win rate: {summary.loc[best_win_idx, 'Historical_Win_Rate']:.1%}",
                "🏆",
                "green",
            )
        with top2:
            render_metric_card(
                "Best expected-GP proxy band",
                str(summary.loc[best_gp_idx, "gmr_band"]),
                f"Expected-GP proxy: {summary.loc[best_gp_idx, 'Expected_GP_Proxy']:,.0f}",
                "💡",
                "purple",
            )
        with top3:
            render_metric_card(
                "Selected product filter",
                selected_product,
                f"Historical quotations analyzed: {subset['quote_id'].nunique():,}",
                "📦",
                "blue",
            )

    chart_left, chart_right = st.columns(2)
    with chart_left:
        with st.container(border=True):
            render_section_heading(
                "Historical Win Rate by Margin Band",
                "The different shades emphasize relative conversion performance across gross-margin bands.",
            )
            fig = px.bar(
                summary,
                x="gmr_band",
                y="Historical_Win_Rate",
                text=summary["Historical_Win_Rate"].map(lambda value: f"{value:.1%}" if pd.notna(value) else "N/A"),
                color="Historical_Win_Rate",
                color_continuous_scale=["#DBEAFE", "#60A5FA", "#1D4ED8"],
                hover_data={"Quotations": True},
                labels={"gmr_band": "Gross Margin Rate", "Historical_Win_Rate": "Historical Win Rate"},
            )
            fig.update_layout(coloraxis_showscale=False, yaxis_tickformat=".0%")
            fig.update_traces(marker_line_color="#FFFFFF", marker_line_width=1.0, textposition="outside")
            apply_chart_layout(fig, height=380)
            st.plotly_chart(fig, use_container_width=True)

    with chart_right:
        with st.container(border=True):
            render_section_heading(
                "Historical Expected-GP Proxy",
                "This descriptive proxy combines historical win rate and average realized gross profit.",
            )
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=summary["gmr_band"].astype(str),
                    y=summary["Expected_GP_Proxy"],
                    mode="lines+markers",
                    line=dict(color=COLORS["purple"], width=4),
                    marker=dict(size=10, color="#A78BFA", line=dict(color="#FFFFFF", width=1.2)),
                    fill="tozeroy",
                    fillcolor="rgba(139, 92, 246, 0.14)",
                    name="Expected GP proxy",
                    text=summary["Expected_GP_Proxy"].map(lambda value: f"{value:,.0f}" if pd.notna(value) else "N/A"),
                    hovertemplate="GMR band: %{x}<br>Expected-GP proxy: %{text}<extra></extra>",
                )
            )
            fig.add_hline(y=0, line_color="#CBD5E1", line_width=1)
            fig.update_layout(xaxis_title="Gross Margin Rate", yaxis_title="Historical Expected-GP Proxy", showlegend=False)
            apply_chart_layout(fig, height=380)
            st.plotly_chart(fig, use_container_width=True)

    with st.container(border=True):
        render_section_heading(
            "Detailed Sweet-Spot Table",
            "Use this table to compare the quotation count, historical win rate, average realized gross profit, and expected-GP proxy for each margin band.",
        )
        display_summary = summary.copy()
        display_summary["Historical_Win_Rate"] = display_summary["Historical_Win_Rate"].map(
            lambda value: f"{value:.1%}" if pd.notna(value) else "N/A"
        )
        display_summary["Average_Gross_Profit"] = display_summary["Average_Gross_Profit"].map(
            lambda value: f"{value:,.0f}" if pd.notna(value) else "N/A"
        )
        display_summary["Expected_GP_Proxy"] = display_summary["Expected_GP_Proxy"].map(
            lambda value: f"{value:,.0f}" if pd.notna(value) else "N/A"
        )
        st.dataframe(display_summary, use_container_width=True, hide_index=True)
        st.markdown("")
        render_insight(
            "The expected-GP proxy is descriptive. For actionable quote review, use the Quote Recommender page, which simulates margin scenarios while holding estimated cost constant."
        )


# =============================================================================
# COMPETITOR POSITIONING
# =============================================================================
elif page == "⚖️ Competitor Positioning":
    render_title()

    with st.container(border=True):
        render_section_heading(
            "Competitor Positioning Analysis",
            "Explore how historical win rates vary when the company's unit price is below, approximately aligned with, or above the minimum available competitor benchmark.",
        )
        render_insight(
            "Competitor analysis is retrospective. Use competitor information in operational price review only when the benchmark was available before quotation submission."
        )

    comp_data = data[data["min_competitor_price"].notna()].copy()
    comp_data["price_position"] = np.select(
        [
            comp_data["price_gap_min_competitor_pct"] < -5,
            comp_data["price_gap_min_competitor_pct"].between(-5, 5, inclusive="both"),
        ],
        ["Below competitor benchmark", "Approximately aligned (±5%)"],
        default="Above competitor benchmark",
    )

    position_order = ["Below competitor benchmark", "Approximately aligned (±5%)", "Above competitor benchmark"]
    positioning_summary = (
        comp_data.groupby("price_position")
        .agg(Product_Lines=("quote_id", "size"), Historical_Win_Rate=("is_win", "mean"))
        .reindex(position_order)
        .reset_index()
    )

    left, right = st.columns([1.12, 1.0])
    with left:
        with st.container(border=True):
            render_section_heading(
                "Historical Win Rate by Price Position",
                "Compare historical conversion performance relative to the minimum available competitor price.",
            )
            fig = px.bar(
                positioning_summary,
                x="price_position",
                y="Historical_Win_Rate",
                text=positioning_summary["Historical_Win_Rate"].map(lambda value: f"{value:.1%}"),
                color="price_position",
                color_discrete_map={
                    "Below competitor benchmark": COLORS["green"],
                    "Approximately aligned (±5%)": COLORS["amber"],
                    "Above competitor benchmark": COLORS["red"],
                },
                hover_data={"Product_Lines": True},
                labels={"price_position": "Price position", "Historical_Win_Rate": "Historical Win Rate"},
            )
            fig.update_layout(showlegend=False, yaxis_tickformat=".0%")
            fig.update_traces(textposition="outside")
            apply_chart_layout(fig, height=365)
            st.plotly_chart(fig, use_container_width=True)

    with right:
        with st.container(border=True):
            render_section_heading(
                "Competitor-Information Availability",
                "Number of available competitor-price benchmarks at the product-line level.",
            )
            availability = (
                data["competitor_count_available"]
                .value_counts()
                .sort_index()
                .rename_axis("Available competitor prices")
                .reset_index(name="Product lines")
            )
            fig = px.bar(
                availability,
                x="Available competitor prices",
                y="Product lines",
                text_auto=True,
                color="Available competitor prices",
                color_continuous_scale=["#CBD5E1", "#60A5FA", "#1D4ED8"],
            )
            fig.update_layout(coloraxis_showscale=False)
            fig.update_traces(textposition="outside")
            apply_chart_layout(fig, height=365)
            st.plotly_chart(fig, use_container_width=True)

    with st.container(border=True):
        render_section_heading(
            "Product-Level Competitor Positioning",
            "Only product-position combinations with at least 10 historical observations are shown.",
        )
        product_position = (
            comp_data.groupby(["product", "price_position"])
            .agg(Observations=("quote_id", "size"), Win_Rate=("is_win", "mean"))
            .reset_index()
        )
        product_position = product_position[product_position["Observations"] >= 10]
        fig = px.bar(
            product_position,
            x="product",
            y="Win_Rate",
            color="price_position",
            barmode="group",
            color_discrete_map={
                "Below competitor benchmark": COLORS["green"],
                "Approximately aligned (±5%)": COLORS["amber"],
                "Above competitor benchmark": COLORS["red"],
            },
            hover_data={"Observations": True, "Win_Rate": ":.1%"},
            labels={"product": "Product", "Win_Rate": "Historical Win Rate", "price_position": "Price position"},
        )
        fig.update_layout(yaxis_tickformat=".0%")
        apply_chart_layout(fig, height=440)
        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# DATA QUALITY AND MODEL NOTES
# =============================================================================
else:
    render_title()
    quote_counts = data.groupby("quote_id").size()

    with st.container(border=True):
        render_section_heading(
            "Data Quality & Model Notes",
            "A structured overview of cleaned data, model validation, interpretation boundaries, and recommended next data improvements.",
        )
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            render_metric_card("Clean product-line records", f"{len(data):,}", "After preprocessing safeguards", "🧹", "blue")
        with r2:
            render_metric_card("Unique Quote IDs", f"{data['quote_id'].nunique():,}", "Quotation-level identifiers", "🧾", "teal")
        with r3:
            render_metric_card("Multi-line Quote IDs", f"{(quote_counts > 1).sum():,}", "Quotes with multiple retained lines", "🧩", "purple")
        with r4:
            render_metric_card("Negative-margin records", f"{(data['gross_margin_rate'] < 0).sum():,}", "Retained as valid business cases", "⚠️", "amber")

    left, right = st.columns([1.0, 1.12])

    with left:
        with st.container(border=True):
            render_section_heading(
                "Competitor-Information Completeness",
                "Missing competitor-price records indicate limited historical benchmark availability.",
            )
            missing_raw = pd.DataFrame(
                {
                    "Field": ["Competitor A", "Competitor B", "Competitor C"],
                    "Missing records": [
                        data["competitor_a"].isna().sum(),
                        data["competitor_b"].isna().sum(),
                        data["competitor_c"].isna().sum(),
                    ],
                    "Missing rate value": [
                        data["competitor_a"].isna().mean(),
                        data["competitor_b"].isna().mean(),
                        data["competitor_c"].isna().mean(),
                    ],
                }
            )
            missing_raw["Missing rate"] = missing_raw["Missing rate value"].map(lambda value: f"{value:.1%}")
            fig = px.bar(
                missing_raw,
                x="Missing rate value",
                y="Field",
                orientation="h",
                text="Missing rate",
                color="Field",
                color_discrete_map={
                    "Competitor A": COLORS["blue"],
                    "Competitor B": COLORS["purple"],
                    "Competitor C": COLORS["red"],
                },
                hover_data={"Missing records": True, "Missing rate value": ":.1%"},
                labels={"Missing rate value": "Missing rate"},
            )
            fig.update_layout(showlegend=False, xaxis_tickformat=".0%")
            fig.update_traces(textposition="outside")
            apply_chart_layout(fig, height=290)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(missing_raw[["Field", "Missing records", "Missing rate"]], use_container_width=True, hide_index=True)

    with right:
        with st.container(border=True):
            render_section_heading(
                "Model Validation Summary",
                "The explainable logistic-regression baseline is evaluated with a Quote-ID-aware holdout split. Product lines from the same quotation do not appear in both training and test sets.",
            )
            mcols = st.columns(3)
            metric_items = list(model_metrics.items())
            for idx, (name, value) in enumerate(metric_items):
                with mcols[idx % 3]:
                    render_metric_card(name, f"{value:.3f}", "Holdout evaluation metric", "📐", ["blue", "teal", "purple"][idx % 3])
                    st.markdown("")

    info_left, info_right = st.columns([1.0, 1.0])
    with info_left:
        with st.container(border=True):
            render_section_heading(
                "Confusion Matrix",
                "The heatmap shows how many historical holdout observations were classified correctly and incorrectly.",
            )
            cm_df = pd.DataFrame(confusion, index=["Actual Loss", "Actual Win"], columns=["Predicted Loss", "Predicted Win"])
            heatmap = go.Figure(
                data=go.Heatmap(
                    z=cm_df.values,
                    x=cm_df.columns,
                    y=cm_df.index,
                    text=cm_df.values,
                    texttemplate="%{text}",
                    textfont={"size": 18},
                    colorscale=[[0, "#EFF6FF"], [1, "#2563EB"]],
                    showscale=False,
                    hovertemplate="%{y}<br>%{x}<br>Records: %{z}<extra></extra>",
                )
            )
            heatmap.update_layout(xaxis_title="Predicted class", yaxis_title="Actual class")
            apply_chart_layout(heatmap, height=310)
            st.plotly_chart(heatmap, use_container_width=True)

    with info_right:
        with st.container(border=True):
            render_section_heading(
                "Model Design Safeguards",
                "The current prototype includes safeguards based on the interview clarification and the exploratory data review.",
            )
            render_note_html(
                """
                <ul>
                    <li>Records are separated by <b>Quote ID</b> during holdout evaluation.</li>
                    <li>Each Quote ID is weighted equally during model training.</li>
                    <li>Quantity values less than or equal to zero are excluded.</li>
                    <li>Negative gross-margin records are retained as valid business cases.</li>
                    <li>Missing competitor prices are not replaced with zero.</li>
                    <li>The highest unit price is retained for repeated Quote ID–product combinations following the interview guidance on price-list updates.</li>
                </ul>
                """
            )

    note_left, note_right = st.columns(2)
    with note_left:
        with st.container(border=True):
            render_section_heading(
                "Important Interpretation Notes",
                "Use these boundaries when presenting and interpreting the dashboard output.",
            )
            render_note_html(
                """
                <ul>
                    <li>The dashboard estimates <b>associations</b>, not causal effects.</li>
                    <li>The primary use case is <b>during price review</b>, before quotation submission.</li>
                    <li>The dashboard operates at the <b>product-line price-review level</b>.</li>
                    <li>A quotation containing multiple products still requires managerial review at the full quotation level.</li>
                    <li>Competitor-price analysis should be used operationally only when the information was available before the tender outcome.</li>
                    <li>The recommendation supports managerial judgement; it is not an automated pricing decision.</li>
                </ul>
                """
            )

    with note_right:
        with st.container(border=True):
            render_section_heading(
                "Recommended Next Data Improvements",
                "These additions would strengthen the accuracy, interpretability, and business usefulness of future dashboard versions.",
            )
            render_note_html(
                """
                <ol>
                    <li>Add quotation timestamps and revision numbers to distinguish price updates from product-line records.</li>
                    <li>Record whether competitor prices were known before quotation submission or learned after the tender result.</li>
                    <li>Add customer segment, salesperson, region, and customer-history variables.</li>
                    <li>Add full quotation-level identifiers and product-line identifiers.</li>
                    <li>Record the final commercial reason for each won or lost tender when available.</li>
                </ol>
                """
            )

from __future__ import annotations

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

APP_VERSION = "v2.1.0"
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

# Color system
NAVY = "#172033"
BLUE = "#2563EB"
SKY = "#0EA5E9"
TEAL = "#14B8A6"
GREEN = "#22C55E"
AMBER = "#F59E0B"
ORANGE = "#F97316"
RED = "#EF4444"
PURPLE = "#7C3AED"
SLATE = "#64748B"
LIGHT_BG = "#F5F7FB"
CARD_BG = "#FFFFFF"


# =============================================================================
# GLOBAL UI STYLE
# =============================================================================
st.markdown(
    """
    <style>
    :root {
        --navy: #172033;
        --blue: #2563EB;
        --sky: #0EA5E9;
        --teal: #14B8A6;
        --green: #22C55E;
        --amber: #F59E0B;
        --orange: #F97316;
        --red: #EF4444;
        --purple: #7C3AED;
        --slate: #64748B;
        --light-bg: #F5F7FB;
        --card-bg: #FFFFFF;
        --soft-border: #E4E7EC;
    }

    .stApp {
        background: var(--light-bg);
    }

    .block-container {
        padding-top: 1.25rem;
        padding-bottom: 2.25rem;
        max-width: 1480px;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #172554 0%, #1E3A8A 100%);
        border-right: 0;
    }

    section[data-testid="stSidebar"] * {
        color: #F8FAFC;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        padding: 0.3rem 0.2rem;
    }

    .page-title {
        color: #172033;
        font-size: 2.45rem;
        font-weight: 800;
        line-height: 1.1;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.035rem;
    }

    .page-subtitle {
        color: #667085;
        font-size: 0.98rem;
        margin-bottom: 1rem;
    }

    .section-title {
        color: #172033;
        font-size: 1.2rem;
        font-weight: 800;
        margin: 0 0 0.15rem 0;
    }

    .section-caption {
        color: #667085;
        font-size: 0.9rem;
        margin: 0 0 0.7rem 0;
    }

    .metric-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
        border: 1px solid #E4E7EC;
        border-top: 4px solid var(--accent, #2563EB);
        border-radius: 18px;
        box-shadow: 0 4px 14px rgba(16, 24, 40, 0.045);
        padding: 0.9rem 1rem 0.85rem 1rem;
        min-height: 126px;
        margin-bottom: 0.3rem;
    }

    .metric-card.compact {
        min-height: 108px;
    }

    .metric-topline {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.35rem;
    }

    .metric-label {
        color: #667085;
        font-size: 0.84rem;
        font-weight: 700;
    }

    .metric-icon {
        background: #EEF4FF;
        border-radius: 11px;
        padding: 0.35rem 0.5rem;
        font-size: 1rem;
    }

    .metric-value {
        color: #101828;
        font-size: 1.8rem;
        font-weight: 800;
        line-height: 1.12;
    }

    .metric-footnote {
        color: #667085;
        font-size: 0.78rem;
        line-height: 1.35;
        margin-top: 0.45rem;
    }

    .group-title {
        display: inline-block;
        font-size: 1.02rem;
        font-weight: 800;
        color: #172033;
        margin-bottom: 0.1rem;
    }

    .group-caption {
        color: #667085;
        font-size: 0.84rem;
        margin-bottom: 0.5rem;
    }

    .pill {
        display: inline-block;
        border-radius: 999px;
        padding: 0.22rem 0.62rem;
        font-size: 0.74rem;
        font-weight: 700;
        margin-bottom: 0.45rem;
    }

    .pill-blue { background: #DBEAFE; color: #1D4ED8; }
    .pill-green { background: #DCFCE7; color: #15803D; }
    .pill-purple { background: #EDE9FE; color: #6D28D9; }
    .pill-amber { background: #FEF3C7; color: #B45309; }

    .insight-box {
        background: #EFF6FF;
        border-left: 5px solid #2563EB;
        border-radius: 14px;
        color: #1D4ED8;
        font-size: 0.91rem;
        line-height: 1.5;
        padding: 0.85rem 1rem;
        margin: 0.65rem 0;
    }

    .soft-box {
        background: #F8FAFC;
        border: 1px solid #E4E7EC;
        border-radius: 16px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.55rem;
    }

    .soft-box ul, .soft-box ol {
        padding-left: 1.1rem;
        margin: 0.2rem 0 0.1rem 0;
    }

    .soft-box li {
        color: #475467;
        font-size: 0.88rem;
        line-height: 1.45;
        margin-bottom: 0.3rem;
    }

    .risk-card {
        background: #FFFFFF;
        border: 1px solid #E4E7EC;
        border-left: 5px solid var(--risk-color, #64748B);
        border-radius: 14px;
        padding: 0.7rem 0.9rem;
        margin-bottom: 0.55rem;
    }

    .risk-card b { color: #172033; }
    .risk-card span { color: #667085; font-size: 0.86rem; }

    .sidebar-badge {
        background: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 12px;
        padding: 0.55rem 0.7rem;
        color: #F8FAFC;
        font-size: 0.78rem;
        line-height: 1.45;
    }

    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E4E7EC;
        border-radius: 14px;
        padding: 0.7rem 0.8rem;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #FFFFFF;
        border-radius: 18px;
        box-shadow: 0 4px 14px rgba(16, 24, 40, 0.04);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# DATA AND MODELLING
# =============================================================================
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Load the prepared dataset and apply safeguards confirmed during interviews."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH.name}")

    df = pd.read_csv(DATA_PATH)

    # Quantity <= 0 was confirmed as a system error.
    df = df[df["qty"] > 0].copy()

    # The expert recommended retaining the highest unit price when price-list
    # updates create repeated Quote ID + Product records.
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
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
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
    """Train a Quote-ID-aware explainable baseline and return validation metrics."""
    dataset = load_data()
    X = dataset[MODEL_FEATURES]
    y = dataset["is_win"]
    groups = dataset["quote_id"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    train_dataset = dataset.iloc[train_idx]
    train_quote_counts = train_dataset.groupby("quote_id").size()
    train_weights = train_dataset["quote_id"].map(lambda quote: 1.0 / train_quote_counts.loc[quote]).to_numpy()

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

    # Final model is fitted using all records with equal weighting per Quote ID.
    all_quote_counts = dataset.groupby("quote_id").size()
    all_weights = dataset["quote_id"].map(lambda quote: 1.0 / all_quote_counts.loc[quote]).to_numpy()

    final_model = create_pipeline()
    final_model.fit(X, y, classifier__sample_weight=all_weights)
    return final_model, metrics, cm


def product_defaults(dataset: pd.DataFrame, product: str) -> Dict[str, float]:
    subset = dataset[dataset["product"] == product].copy()
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

    competitor_values = available_competitors(competitor_a, competitor_b, competitor_c)
    competitor_count = len(competitor_values)
    avg_competitor = float(np.mean(competitor_values)) if competitor_values else np.nan
    min_competitor = float(np.min(competitor_values)) if competitor_values else np.nan

    avg_gap_pct = ((unit_price - avg_competitor) / avg_competitor) * 100 if competitor_values else np.nan
    min_gap_pct = ((unit_price - min_competitor) / min_competitor) * 100 if competitor_values else np.nan

    row = {
        "product": product,
        "kw": float(kw),
        "qty": int(qty),
        "unit_price": float(unit_price),
        "gross_margin_rate": float(gross_margin_rate),
        "energy_grant_amount": float(energy_grant_amount),
        "estimated_cost": float(total_cost),
        "grant_ratio_to_subtotal": float(energy_grant_amount / subtotal_price) if subtotal_price else np.nan,
        "competitor_count_available": int(competitor_count),
        "avg_competitor_price": avg_competitor,
        "min_competitor_price": min_competitor,
        "price_gap_avg_competitor_pct": avg_gap_pct,
        "price_gap_min_competitor_pct": min_gap_pct,
        "subtotal_price": float(subtotal_price),
        "gross_profit_amount": float(subtotal_price - total_cost),
    }
    return pd.DataFrame([row])


def score_scenario(model: Pipeline, scenario: pd.DataFrame) -> Dict[str, float]:
    win_probability = float(model.predict_proba(scenario[MODEL_FEATURES])[:, 1][0])
    gross_profit = float(scenario["gross_profit_amount"].iloc[0])
    expected_gross_profit = win_probability * gross_profit
    return {
        "win_probability": win_probability,
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
    records: List[Dict[str, float]] = []
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
        score = score_scenario(model, scenario)
        records.append(
            {
                "Gross Margin Rate": float(gmr),
                "Unit Price": float(scenario["unit_price"].iloc[0]),
                "Win Probability": score["win_probability"],
                "Gross Profit if Won": score["gross_profit_amount"],
                "Expected Gross Profit": score["expected_gross_profit"],
                "Meets Minimum Margin": bool(gmr >= minimum_margin),
            }
        )
    return pd.DataFrame(records)


def historical_benchmark(dataset: pd.DataFrame, product: str, gmr: float) -> Dict[str, float]:
    band = pd.cut(pd.Series([gmr]), bins=GMR_BINS, labels=GMR_LABELS).iloc[0]
    subset = dataset[(dataset["product"] == product) & (dataset["gmr_band"] == band)].copy()
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
    price_gap_ratio = (unit_price - min_competitor) / min_competitor
    if price_gap_ratio < -0.05:
        return "Below competitor benchmark"
    if price_gap_ratio <= 0.05:
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
        flags.append(("Medium", "Conversion risk", f"Estimated win probability is below the review threshold of {minimum_win_probability:.0%}."))

    if int(row["competitor_count_available"]) == 0:
        flags.append(("Medium", "Information gap", "No competitor benchmark is available. Review with additional sales information."))
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


def signed_currency(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:+,.0f}"


def render_page_header(title: str, subtitle: str) -> None:
    st.markdown(f"<div class='page-title'>{title}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='page-subtitle'>{subtitle}</div>", unsafe_allow_html=True)


def render_title() -> None:
    render_page_header(
        "Quotation Pricing Decision Support Dashboard",
        "Interactive price-review support for gross-margin management, competitor positioning, win-probability estimation, and quotation risk review.",
    )


def render_section_header(title: str, caption: str = "") -> None:
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    if caption:
        st.markdown(f"<div class='section-caption'>{caption}</div>", unsafe_allow_html=True)


def render_metric_card(
    label: str,
    value: str,
    footnote: str,
    accent: str = BLUE,
    icon: str = "📌",
    compact: bool = False,
) -> None:
    compact_class = " compact" if compact else ""
    st.markdown(
        f"""
        <div class='metric-card{compact_class}' style='--accent:{accent};'>
            <div class='metric-topline'>
                <div class='metric-label'>{label}</div>
                <div class='metric-icon'>{icon}</div>
            </div>
            <div class='metric-value'>{value}</div>
            <div class='metric-footnote'>{footnote}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_insight(text: str) -> None:
    st.markdown(f"<div class='insight-box'>{text}</div>", unsafe_allow_html=True)


def render_risk_card(severity: str, title: str, explanation: str) -> None:
    color = GREEN if severity == "Low" else AMBER if severity == "Medium" else RED
    st.markdown(
        f"""
        <div class='risk-card' style='--risk-color:{color};'>
            <b>{severity}: {title}</b><br>
            <span>{explanation}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def standard_layout(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#475467"),
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(showgrid=False, linecolor="#E4E7EC")
    fig.update_yaxes(gridcolor="#EAECF0", linecolor="#E4E7EC")
    return fig


# =============================================================================
# SHARED RESOURCES
# =============================================================================
try:
    data = load_data()
    model, model_metrics, confusion = train_model()
except Exception as exc:
    st.error(f"The dashboard could not be initialized: {exc}")
    st.stop()


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("## 📊 Pricing Review")
    st.caption("Quotation Decision Support")
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Executive Overview",
            "Quote Recommender",
            "Margin Sweet Spot",
            "Competitor Positioning",
            "Data Quality & Model Notes",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**Decision-support scope**")
    st.caption(
        "This dashboard supports price review. It does not replace managerial approval, customer knowledge, or commercial judgement."
    )
    st.markdown(
        f"""
        <div class='sidebar-badge'>
            <b>Version:</b> {APP_VERSION}<br>
            <b>Last updated:</b> {LAST_UPDATED}
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# PAGE 1 — EXECUTIVE OVERVIEW
# =============================================================================
if page == "Executive Overview":
    render_title()
    quote_header = data.drop_duplicates(subset=["quote_id"]).copy()

    with st.container(border=True):
        render_section_header(
            "Executive Overview",
            "A high-level summary of historical quotation performance, product conversion patterns, and potential lost-tender review signals.",
        )
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            render_metric_card("Historical quotations", f"{quote_header['quote_id'].nunique():,}", "Unique quotation records", BLUE, "🧾")
        with col2:
            render_metric_card("Product categories", f"{data['product'].nunique():,}", "Distinct anonymized product codes", TEAL, "📦")
        with col3:
            render_metric_card("Historical win rate", f"{quote_header['is_win'].mean():.1%}", "Quotation-level conversion success", GREEN, "🏆")
        with col4:
            render_metric_card("Rows with competitor info", f"{(data['competitor_count_available'] > 0).mean():.1%}", "Product lines with benchmark availability", PURPLE, "🔎")

        render_insight(
            "The dashboard is designed around the project objective: balancing quotation profitability with tender success while considering product characteristics, energy grants, and competitor-price benchmarks."
        )

    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            render_section_header("Historical win rate by gross-margin band", "Quotation-level performance across gross-margin ranges.")
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
                color_discrete_sequence=[BLUE, SKY, TEAL, GREEN, AMBER, ORANGE, RED, PURPLE],
                hover_data={"Quotations": True, "Win_Rate": ":.1%"},
                labels={"gmr_band": "Gross Margin Rate", "Win_Rate": "Historical Win Rate"},
            )
            fig.update_layout(showlegend=False, yaxis_tickformat=".0%")
            fig.update_traces(marker_line_color="#FFFFFF", marker_line_width=1.1)
            st.plotly_chart(standard_layout(fig, 360), use_container_width=True)

    with right:
        with st.container(border=True):
            render_section_header("Product-level win rate", "Products with at least 20 quotation observations.")
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
                color_continuous_scale=["#CCFBF1", "#14B8A6", "#0F766E"],
                hover_data={"Observations": True, "Win_Rate": ":.1%"},
                labels={"product": "Product", "Win_Rate": "Historical Win Rate"},
            )
            fig.update_layout(coloraxis_showscale=False, xaxis_tickformat=".0%")
            st.plotly_chart(standard_layout(fig, 360), use_container_width=True)

    with st.container(border=True):
        render_section_header(
            "Potential lost-tender review signals",
            "Rule-based descriptive signals for failed product lines. These are not confirmed causal reasons.",
        )
        product_median = data.groupby("product")["gross_margin_rate"].median().to_dict()
        failed = data[data["is_win"] == 0].copy()
        failed["Potential review signal"] = failed.apply(lambda row: potential_loss_signal(row, product_median), axis=1)
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
                "Above competitor benchmark": RED,
                "Other factors not captured": SLATE,
                "Competitor information unavailable": PURPLE,
                "High margin for product": AMBER,
            },
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(standard_layout(fig, 370), use_container_width=True)
        st.caption(
            "Additional customer, salesperson, and sales-process information would be needed before interpreting these signals as causal reasons for tender losses."
        )


# =============================================================================
# PAGE 2 — QUOTE RECOMMENDER
# =============================================================================
elif page == "Quote Recommender":
    render_title()

    with st.container(border=True):
        render_section_header(
            "Interactive Quote Recommender",
            "Enter a product-line pricing scenario to compare the proposed quote with a model-based recommendation.",
        )

        products = sorted(data["product"].astype(str).unique().tolist())
        input_left, input_middle, input_right = st.columns(3)

        with input_left:
            st.markdown("<span class='pill pill-blue'>1. Product & Cost</span>", unsafe_allow_html=True)
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

        with input_middle:
            st.markdown("<span class='pill pill-purple'>2. Margin & Review Rules</span>", unsafe_allow_html=True)
            selected_gmr_pct = st.slider("Proposed gross-margin rate", min_value=-10, max_value=80, value=30, step=1)
            energy_grant = st.number_input(
                "Energy grant amount",
                min_value=0.0,
                value=float(round(defaults["grant"], 2)),
                step=1000.0,
                help="Historical data indicate that grant values are product-related. Adjust this when reviewing a specific quotation.",
            )
            minimum_margin_pct = st.slider(
                "Minimum target margin for recommendation",
                min_value=0,
                max_value=50,
                value=20,
                step=1,
                help="The default reflects the interview discussion. Lower-margin cases may still be reviewed by a supervisor.",
            )
            review_win_threshold_pct = st.slider("Win-probability review threshold", 10, 80, 40, 5)

        with input_right:
            st.markdown("<span class='pill pill-amber'>3. Competitor Benchmarks</span>", unsafe_allow_html=True)
            use_competitor_info = st.checkbox("Use competitor-price information", value=pd.notna(defaults["competitor_a"]))
            if use_competitor_info:
                comp_a_default = 0.0 if pd.isna(defaults["competitor_a"]) else defaults["competitor_a"]
                comp_b_default = 0.0 if pd.isna(defaults["competitor_b"]) else defaults["competitor_b"]
                comp_c_default = 0.0 if pd.isna(defaults["competitor_c"]) else defaults["competitor_c"]
                competitor_a = st.number_input("Competitor A price (0 = unavailable)", min_value=0.0, value=float(comp_a_default), step=1000.0)
                competitor_b = st.number_input("Competitor B price (0 = unavailable)", min_value=0.0, value=float(comp_b_default), step=1000.0)
                competitor_c = st.number_input("Competitor C price (0 = unavailable)", min_value=0.0, value=float(comp_c_default), step=1000.0)
            else:
                competitor_a = competitor_b = competitor_c = np.nan
            st.caption("Competitor fields remain optional because market information is frequently unavailable in historical records.")

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
    proposed_unit_price = float(scenario["unit_price"].iloc[0])
    proposed_win_probability = float(selected_score["win_probability"])
    proposed_gp_if_won = float(selected_score["gross_profit_amount"])
    proposed_expected_gp = float(selected_score["expected_gross_profit"])

    recommended_gmr = float(best_row["Gross Margin Rate"])
    recommended_unit_price = float(best_row["Unit Price"])
    recommended_win_probability = float(best_row["Win Probability"])
    recommended_gp_if_won = float(best_row["Gross Profit if Won"])
    recommended_expected_gp = float(best_row["Expected Gross Profit"])

    delta_gmr = recommended_gmr - selected_gmr
    delta_unit_price = recommended_unit_price - proposed_unit_price
    delta_win_probability = recommended_win_probability - proposed_win_probability
    delta_gp_if_won = recommended_gp_if_won - proposed_gp_if_won
    delta_expected_gp = recommended_expected_gp - proposed_expected_gp

    min_competitor = float(scenario["min_competitor_price"].iloc[0]) if pd.notna(scenario["min_competitor_price"].iloc[0]) else np.nan

    # -------------------------------------------------------------------------
    # Scenario results with category-based information architecture
    # -------------------------------------------------------------------------
    with st.container(border=True):
        render_section_header(
            "Scenario Results",
            "Pricing, probability, and profitability are separated below so the current quotation and model recommendation can be reviewed more clearly.",
        )

        proposed_col, recommendation_col = st.columns(2)

        with proposed_col:
            with st.container(border=True):
                st.markdown("<span class='pill pill-blue'>Current Scenario</span>", unsafe_allow_html=True)
                st.markdown("<div class='group-title'>Proposed Quote</div>", unsafe_allow_html=True)
                st.markdown("<div class='group-caption'>The quotation currently entered by the user.</div>", unsafe_allow_html=True)

                st.markdown("**Pricing**")
                p1, p2 = st.columns(2)
                with p1:
                    render_metric_card("Proposed GMR", f"{selected_gmr:.0%}", "Current gross-margin input", BLUE, "📌", compact=True)
                with p2:
                    render_metric_card("Proposed unit price", currency(proposed_unit_price), "Calculated from cost and proposed GMR", BLUE, "🏷️", compact=True)

                st.markdown("**Win Probability**")
                render_metric_card("Estimated win probability", f"{proposed_win_probability:.1%}", "Model-estimated chance for the proposed quote", SKY, "🎯", compact=True)

                st.markdown("**Profitability**")
                p3, p4 = st.columns(2)
                with p3:
                    render_metric_card("Gross profit if won", currency(proposed_gp_if_won), "Profit conditional on tender success", TEAL, "📈", compact=True)
                with p4:
                    render_metric_card("Expected gross profit", currency(proposed_expected_gp), "Probability-adjusted gross profit", PURPLE, "⚖️", compact=True)

        with recommendation_col:
            with st.container(border=True):
                st.markdown("<span class='pill pill-green'>Model Recommendation</span>", unsafe_allow_html=True)
                st.markdown("<div class='group-title'>Recommended Quote</div>", unsafe_allow_html=True)
                st.markdown("<div class='group-caption'>The scenario with the highest expected gross profit within the selected minimum-margin rule.</div>", unsafe_allow_html=True)

                st.markdown("**Pricing**")
                r1, r2 = st.columns(2)
                with r1:
                    render_metric_card("Recommended GMR", f"{recommended_gmr:.0%}", f"Change versus proposed: {delta_gmr:+.0%}", GREEN, "✅", compact=True)
                with r2:
                    render_metric_card("Recommended unit price", currency(recommended_unit_price), f"Change versus proposed: {signed_currency(delta_unit_price)}", GREEN, "🏷️", compact=True)

                st.markdown("**Win Probability**")
                render_metric_card("Win probability at recommendation", f"{recommended_win_probability:.1%}", f"Change versus proposed: {delta_win_probability:+.1%}", AMBER, "🏆", compact=True)

                st.markdown("**Profitability**")
                r3, r4 = st.columns(2)
                with r3:
                    render_metric_card("Gross profit if won", currency(recommended_gp_if_won), f"Change versus proposed: {signed_currency(delta_gp_if_won)}", TEAL, "📊", compact=True)
                with r4:
                    render_metric_card("Expected GP at recommendation", currency(recommended_expected_gp), f"Change versus proposed: {signed_currency(delta_expected_gp)}", PURPLE, "💡", compact=True)

        st.markdown("### Comparison Summary")
        comparison_table = pd.DataFrame(
            {
                "Category": ["Pricing", "Pricing", "Probability", "Profitability", "Profitability"],
                "Metric": ["Gross Margin Rate", "Unit Price", "Win Probability", "Gross Profit if Won", "Expected Gross Profit"],
                "Proposed Quote": [
                    f"{selected_gmr:.0%}",
                    currency(proposed_unit_price),
                    f"{proposed_win_probability:.1%}",
                    currency(proposed_gp_if_won),
                    currency(proposed_expected_gp),
                ],
                "Recommended Quote": [
                    f"{recommended_gmr:.0%}",
                    currency(recommended_unit_price),
                    f"{recommended_win_probability:.1%}",
                    currency(recommended_gp_if_won),
                    currency(recommended_expected_gp),
                ],
                "Difference": [
                    f"{delta_gmr:+.0%}",
                    signed_currency(delta_unit_price),
                    f"{delta_win_probability:+.1%}",
                    signed_currency(delta_gp_if_won),
                    signed_currency(delta_expected_gp),
                ],
            }
        )
        st.dataframe(comparison_table, use_container_width=True, hide_index=True)

        render_insight(
            f"The proposed quotation produces an estimated win probability of <b>{proposed_win_probability:.1%}</b> and expected gross profit of <b>{currency(proposed_expected_gp)}</b>. "
            f"Within the selected minimum-margin rule, the model recommends a GMR of <b>{recommended_gmr:.0%}</b>, resulting in model-estimated expected gross profit of <b>{currency(recommended_expected_gp)}</b>. "
            "The recommendation should still be reviewed alongside customer context, sales knowledge, and competitor-data availability."
        )

    simulation_col, benchmark_col = st.columns([1.55, 1.0])

    with simulation_col:
        with st.container(border=True):
            render_section_header("What-if margin simulation", "Explore how the margin level changes win probability and expected gross profit.")
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=grid["Gross Margin Rate"] * 100,
                    y=grid["Win Probability"],
                    name="Estimated win probability",
                    mode="lines",
                    line=dict(color=BLUE, width=4),
                    yaxis="y1",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=grid["Gross Margin Rate"] * 100,
                    y=grid["Expected Gross Profit"],
                    name="Expected gross profit",
                    mode="lines",
                    line=dict(color=PURPLE, width=4),
                    fill="tozeroy",
                    fillcolor="rgba(124, 58, 237, 0.10)",
                    yaxis="y2",
                )
            )
            fig.add_vline(x=selected_gmr_pct, line_dash="dash", line_color=BLUE, annotation_text="Proposed GMR")
            fig.add_vline(x=recommended_gmr * 100, line_dash="dot", line_color=GREEN, annotation_text="Recommended GMR")
            fig.update_layout(
                xaxis_title="Gross Margin Rate (%)",
                yaxis=dict(title="Estimated Win Probability", tickformat=".0%"),
                yaxis2=dict(title="Expected Gross Profit", overlaying="y", side="right", showgrid=False),
            )
            st.plotly_chart(standard_layout(fig, 420), use_container_width=True)

    with benchmark_col:
        with st.container(border=True):
            render_section_header("Historical benchmark", "Contextual benchmark for the proposed scenario.")
            benchmark_win_rate = "N/A" if pd.isna(benchmark["historical_win_rate"]) else f"{benchmark['historical_win_rate']:.1%}"
            b1, b2 = st.columns(2)
            with b1:
                render_metric_card("Selected product", product, "Current review scenario", BLUE, "📦", compact=True)
            with b2:
                render_metric_card("GMR band", benchmark["band"], "Historical margin interval", TEAL, "📐", compact=True)
            b3, b4 = st.columns(2)
            with b3:
                render_metric_card("Historical records", f"{benchmark['n']:,}", "Product records in selected band", AMBER, "🧾", compact=True)
            with b4:
                render_metric_card("Historical win rate", benchmark_win_rate, "Descriptive benchmark only", PURPLE, "🏅", compact=True)
            render_insight(f"<b>Price position:</b> {classify_price_position(proposed_unit_price, min_competitor)}")

    flags_col, competitor_col = st.columns([1.0, 1.2])

    with flags_col:
        with st.container(border=True):
            render_section_header("Quote review flags", "Rule-based warnings that may require sales or supervisor review.")
            for severity, title, explanation in risk_flags(scenario, selected_score, minimum_margin, review_win_threshold):
                render_risk_card(severity, title, explanation)

    with competitor_col:
        with st.container(border=True):
            render_section_header("Competitor positioning", "Compare the proposed unit price with available benchmark prices.")
            competitors = available_competitors(competitor_a, competitor_b, competitor_c)
            if competitors:
                competitor_labels = [
                    f"Competitor {chr(65 + index)}"
                    for index, value in enumerate([competitor_a, competitor_b, competitor_c])
                    if pd.notna(value) and value > 0
                ]
                chart_data = pd.DataFrame(
                    {
                        "Price reference": ["Proposed unit price"] + competitor_labels,
                        "Price": [proposed_unit_price] + competitors,
                    }
                )
                fig = px.bar(
                    chart_data,
                    x="Price reference",
                    y="Price",
                    text_auto=",.0f",
                    color="Price reference",
                    color_discrete_sequence=[BLUE, TEAL, AMBER, PURPLE],
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(standard_layout(fig, 320), use_container_width=True)
            else:
                st.warning("No competitor benchmark is available for this scenario. Review with additional market information.")

    with st.expander("View top margin scenarios"):
        display_grid = grid.sort_values("Expected Gross Profit", ascending=False).head(10).copy()
        display_grid["Gross Margin Rate"] = display_grid["Gross Margin Rate"].map(lambda value: f"{value:.0%}")
        display_grid["Win Probability"] = display_grid["Win Probability"].map(lambda value: f"{value:.1%}")
        for column in ["Unit Price", "Gross Profit if Won", "Expected Gross Profit"]:
            display_grid[column] = display_grid[column].map(currency)
        st.dataframe(display_grid, use_container_width=True, hide_index=True)


# =============================================================================
# PAGE 3 — MARGIN SWEET SPOT
# =============================================================================
elif page == "Margin Sweet Spot":
    render_title()

    with st.container(border=True):
        render_section_header(
            "Gross-Margin Sweet Spot Analysis",
            "Compare historical conversion performance and descriptive expected-gross-profit proxy across margin ranges.",
        )
        options = ["All products"] + sorted(data["product"].astype(str).unique().tolist())
        selected_product = st.selectbox("Filter by product", options)

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

        summary_for_metrics = summary.dropna(subset=["Historical_Win_Rate", "Expected_GP_Proxy"])
        best_win_row = summary_for_metrics.loc[summary_for_metrics["Historical_Win_Rate"].idxmax()]
        best_gp_row = summary_for_metrics.loc[summary_for_metrics["Expected_GP_Proxy"].idxmax()]

        m1, m2, m3 = st.columns(3)
        with m1:
            render_metric_card("Best historical win-rate band", str(best_win_row["gmr_band"]), f"Historical win rate: {best_win_row['Historical_Win_Rate']:.1%}", BLUE, "🎯")
        with m2:
            render_metric_card("Best expected-GP band", str(best_gp_row["gmr_band"]), f"Expected-GP proxy: {best_gp_row['Expected_GP_Proxy']:,.0f}", PURPLE, "💡")
        with m3:
            render_metric_card("Filtered product", selected_product, f"Quotation records: {int(subset['quote_id'].nunique()):,}", TEAL, "📦")

    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            render_section_header("Historical win rate by margin band", "Win-rate distribution across gross-margin intervals.")
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
            st.plotly_chart(standard_layout(fig, 390), use_container_width=True)

    with right:
        with st.container(border=True):
            render_section_header("Historical expected-GP proxy", "Descriptive pattern of win-rate-adjusted gross-profit outcomes.")
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=summary["gmr_band"],
                    y=summary["Expected_GP_Proxy"],
                    mode="lines+markers",
                    line=dict(color=PURPLE, width=4),
                    marker=dict(color="#A78BFA", size=10),
                    fill="tozeroy",
                    fillcolor="rgba(167, 139, 250, 0.18)",
                    name="Expected GP Proxy",
                )
            )
            st.plotly_chart(standard_layout(fig, 390), use_container_width=True)

    with st.container(border=True):
        render_section_header("Detailed sweet-spot table", "Use the table as a descriptive benchmark before opening the Quote Recommender page.")
        display_summary = summary.copy()
        display_summary["Historical_Win_Rate"] = display_summary["Historical_Win_Rate"].map(lambda value: f"{value:.1%}" if pd.notna(value) else "N/A")
        display_summary["Average_Gross_Profit"] = display_summary["Average_Gross_Profit"].map(lambda value: currency(value))
        display_summary["Expected_GP_Proxy"] = display_summary["Expected_GP_Proxy"].map(lambda value: currency(value))
        st.dataframe(display_summary, use_container_width=True, hide_index=True)
        render_insight(
            "The expected-GP proxy is descriptive. For actionable quote review, use the Quote Recommender page, which simulates margin scenarios while holding estimated cost constant."
        )


# =============================================================================
# PAGE 4 — COMPETITOR POSITIONING
# =============================================================================
elif page == "Competitor Positioning":
    render_title()

    with st.container(border=True):
        render_section_header(
            "Competitor Positioning Analysis",
            "Review how historical win rates vary when company pricing is below, aligned with, or above the minimum available competitor benchmark.",
        )
        render_insight(
            "Competitor analysis is retrospective. Some competitor prices may only become known through customer feedback, so operational use requires information available at the price-review stage."
        )

    competitor_data = data[data["min_competitor_price"].notna()].copy()
    competitor_data["price_position"] = np.select(
        [
            competitor_data["price_gap_min_competitor_pct"] < -5,
            competitor_data["price_gap_min_competitor_pct"].between(-5, 5, inclusive="both"),
        ],
        ["Below competitor benchmark", "Approximately aligned (±5%)"],
        default="Above competitor benchmark",
    )

    position_order = ["Below competitor benchmark", "Approximately aligned (±5%)", "Above competitor benchmark"]
    positioning_summary = (
        competitor_data.groupby("price_position")
        .agg(Product_Lines=("quote_id", "size"), Historical_Win_Rate=("is_win", "mean"))
        .reindex(position_order)
        .reset_index()
    )

    left, right = st.columns([1.2, 1.0])
    with left:
        with st.container(border=True):
            render_section_header("Win rate by price position", "Historical tender success by relative pricing position.")
            fig = px.bar(
                positioning_summary,
                x="price_position",
                y="Historical_Win_Rate",
                text=positioning_summary["Historical_Win_Rate"].map(lambda value: f"{value:.1%}"),
                color="price_position",
                color_discrete_map={
                    "Below competitor benchmark": GREEN,
                    "Approximately aligned (±5%)": AMBER,
                    "Above competitor benchmark": RED,
                },
                hover_data={"Product_Lines": True},
                labels={"price_position": "Price position", "Historical_Win_Rate": "Historical Win Rate"},
            )
            fig.update_layout(showlegend=False, yaxis_tickformat=".0%")
            st.plotly_chart(standard_layout(fig, 360), use_container_width=True)

    with right:
        with st.container(border=True):
            render_section_header("Competitor-price availability", "Number of benchmark prices available per product line.")
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
                color_continuous_scale=["#EDE9FE", "#8B5CF6", "#5B21B6"],
            )
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(standard_layout(fig, 360), use_container_width=True)

    with st.container(border=True):
        render_section_header("Product-level competitor positioning", "Only product-position combinations with at least 10 observations are shown.")
        product_position = (
            competitor_data.groupby(["product", "price_position"])
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
            hover_data={"Observations": True, "Win_Rate": ":.1%"},
            color_discrete_map={
                "Below competitor benchmark": GREEN,
                "Approximately aligned (±5%)": AMBER,
                "Above competitor benchmark": RED,
            },
            labels={"product": "Product", "Win_Rate": "Historical Win Rate", "price_position": "Price position"},
        )
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(standard_layout(fig, 390), use_container_width=True)


# =============================================================================
# PAGE 5 — DATA QUALITY AND MODEL NOTES
# =============================================================================
else:
    render_title()
    quote_counts = data.groupby("quote_id").size()

    with st.container(border=True):
        render_section_header(
            "Data Quality & Model Notes",
            "A structured overview of cleaned data, validation results, interpretation boundaries, and data-improvement priorities.",
        )
        q1, q2, q3, q4 = st.columns(4)
        with q1:
            render_metric_card("Clean product-line records", f"{len(data):,}", "Records after preprocessing safeguards", BLUE, "🧹")
        with q2:
            render_metric_card("Unique Quote IDs", f"{data['quote_id'].nunique():,}", "Quotation-level unique identifiers", TEAL, "🧾")
        with q3:
            render_metric_card("Multi-line Quote IDs", f"{(quote_counts > 1).sum():,}", "Quotes with multiple product-line records", PURPLE, "🧩")
        with q4:
            render_metric_card("Negative-margin records", f"{(data['gross_margin_rate'] < 0).sum():,}", "Retained as valid strategic cases", AMBER, "⚠️")

    left, right = st.columns([1.05, 1.0])

    with left:
        with st.container(border=True):
            render_section_header("Competitor-information completeness", "Historical missingness affects how competitor variables should be used operationally.")
            missing_raw = pd.DataFrame(
                {
                    "Field": ["Competitor A", "Competitor B", "Competitor C"],
                    "Missing records": [
                        int(data["competitor_a"].isna().sum()),
                        int(data["competitor_b"].isna().sum()),
                        int(data["competitor_c"].isna().sum()),
                    ],
                    "Missing rate": [
                        float(data["competitor_a"].isna().mean()),
                        float(data["competitor_b"].isna().mean()),
                        float(data["competitor_c"].isna().mean()),
                    ],
                }
            )
            fig = px.bar(
                missing_raw,
                x="Missing rate",
                y="Field",
                orientation="h",
                text=missing_raw["Missing rate"].map(lambda value: f"{value:.1%}"),
                color="Field",
                color_discrete_map={"Competitor A": BLUE, "Competitor B": PURPLE, "Competitor C": RED},
            )
            fig.update_layout(showlegend=False, xaxis_tickformat=".0%")
            st.plotly_chart(standard_layout(fig, 250), use_container_width=True)

            display_missing = missing_raw.copy()
            display_missing["Missing rate"] = display_missing["Missing rate"].map(lambda value: f"{value:.1%}")
            st.dataframe(display_missing, use_container_width=True, hide_index=True)

    with right:
        with st.container(border=True):
            render_section_header("Model validation summary", "Quote-ID-aware holdout evaluation for the explainable logistic-regression baseline.")
            metric_items = list(model_metrics.items())
            row1 = st.columns(3)
            row2 = st.columns(3)
            accents = [BLUE, PURPLE, TEAL, AMBER, GREEN, ORANGE]
            icons = ["📈", "📊", "✅", "🎯", "🔍", "⚖️"]
            for index, ((name, value), accent, icon) in enumerate(zip(metric_items, accents, icons)):
                target_column = row1[index] if index < 3 else row2[index - 3]
                with target_column:
                    render_metric_card(name, f"{value:.3f}", "Holdout evaluation metric", accent, icon, compact=True)

    lower_left, lower_right = st.columns([1.0, 1.0])

    with lower_left:
        with st.container(border=True):
            render_section_header("Confusion matrix", "Classification outcomes using a 0.50 decision threshold.")
            confusion_df = pd.DataFrame(confusion, index=["Actual Loss", "Actual Win"], columns=["Predicted Loss", "Predicted Win"])
            fig = go.Figure(
                data=go.Heatmap(
                    z=confusion_df.values,
                    x=confusion_df.columns,
                    y=confusion_df.index,
                    text=confusion_df.values,
                    texttemplate="%{text}",
                    textfont={"size": 18},
                    colorscale="Blues",
                    showscale=False,
                )
            )
            st.plotly_chart(standard_layout(fig, 285), use_container_width=True)

    with lower_right:
        with st.container(border=True):
            render_section_header("Model safeguards", "Safeguards applied before model training and dashboard use.")
            st.markdown(
                """
                <div class='soft-box'>
                    <ul>
                        <li>Training and holdout records are separated by <b>Quote ID</b>.</li>
                        <li>Each Quote ID is weighted equally during model training.</li>
                        <li>Quantity values less than or equal to zero are excluded as system errors.</li>
                        <li>Negative gross margins are retained as valid strategic business cases.</li>
                        <li>Missing competitor values are not replaced with zero.</li>
                        <li>For repeated Quote ID + Product records, the highest unit price is retained following interview guidance.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

    notes_left, notes_right = st.columns(2)

    with notes_left:
        with st.container(border=True):
            render_section_header("Important interpretation notes", "Boundaries for responsible interpretation of the dashboard output.")
            st.markdown(
                """
                <div class='soft-box'>
                    <ul>
                        <li>The dashboard estimates <b>associations</b>, not causal effects.</li>
                        <li>The primary use case is <b>during price review</b>, before quotation submission.</li>
                        <li>The recommender works at the <b>product-line price-review level</b>.</li>
                        <li>Multi-product quotations still require managerial review at the full quotation level.</li>
                        <li>Competitor-price fields should only be used operationally when available before the decision.</li>
                        <li>The recommendation supports judgement; it does not automate commercial approval.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with notes_right:
        with st.container(border=True):
            render_section_header("Recommended next data improvements", "Additional fields that would strengthen future model usefulness.")
            st.markdown(
                """
                <div class='soft-box'>
                    <ol>
                        <li>Add quotation timestamps and revision numbers.</li>
                        <li>Record whether competitor prices were known before submission or learned after the tender result.</li>
                        <li>Add customer segment, salesperson, region, and customer-history variables.</li>
                        <li>Add explicit full-quotation and product-line identifiers.</li>
                        <li>Record the final commercial reason for a won or lost tender when available.</li>
                    </ol>
                </div>
                """,
                unsafe_allow_html=True,
            )

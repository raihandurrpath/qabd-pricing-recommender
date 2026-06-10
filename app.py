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


# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Quotation Pricing Decision Support",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
    .small-note {font-size: 0.88rem; color: #5f6b7a;}
    .card {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 14px 16px;
        background-color: #fafafa;
        margin-bottom: 10px;
    }
    .risk-high {border-left: 5px solid #d62728;}
    .risk-medium {border-left: 5px solid #ff7f0e;}
    .risk-low {border-left: 5px solid #2ca02c;}
    </style>
    """,
    unsafe_allow_html=True,
)

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

NUMERIC_FEATURES = [f for f in MODEL_FEATURES if f != "product"]
CATEGORICAL_FEATURES = ["product"]


# -----------------------------------------------------------------------------
# Data preparation and modelling
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Load the preprocessed data and apply modelling safeguards."""
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
    train_weights = train_data["quote_id"].map(lambda q: 1.0 / train_quote_counts.loc[q]).to_numpy()

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
    all_weights = data["quote_id"].map(lambda q: 1.0 / all_quote_counts.loc[q]).to_numpy()
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


def available_competitors(a: float, b: float, c: float) -> List[float]:
    values = [a, b, c]
    return [float(v) for v in values if pd.notna(v) and float(v) > 0]


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
    comps = available_competitors(competitor_a, competitor_b, competitor_c)
    count = len(comps)
    avg_comp = float(np.mean(comps)) if comps else np.nan
    min_comp = float(np.min(comps)) if comps else np.nan

    price_gap_avg_pct = ((unit_price - avg_comp) / avg_comp) * 100 if comps else np.nan
    price_gap_min_pct = ((unit_price - min_comp) / min_comp) * 100 if comps else np.nan

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
        "avg_competitor_price": avg_comp,
        "min_competitor_price": min_comp,
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
    # Each Quote ID is counted once in the historical benchmark for the selected product.
    subset = subset.drop_duplicates(subset=["quote_id"])
    if subset.empty:
        return {"band": str(band), "n": 0, "historical_win_rate": np.nan}
    return {
        "band": str(band),
        "n": int(len(subset)),
        "historical_win_rate": float(subset["is_win"].mean()),
    }


def classify_price_position(unit_price: float, min_comp: float) -> str:
    if pd.isna(min_comp):
        return "Competitor information unavailable"
    ratio = (unit_price - min_comp) / min_comp
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
        flags.append(("Medium", "Conversion risk", f"Estimated win probability is below the review threshold of {minimum_win_probability:.0%}."))

    if int(row["competitor_count_available"]) == 0:
        flags.append(("Medium", "Information gap", "No competitor benchmark is available. Review the recommendation with additional sales information."))
    elif pd.notna(row["price_gap_min_competitor_pct"]) and float(row["price_gap_min_competitor_pct"]) > 5:
        flags.append(("High", "Competitor positioning risk", f"Quoted unit price is {row['price_gap_min_competitor_pct']:.1f}% above the minimum available competitor benchmark."))

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


def currency(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:,.0f}"


def render_title() -> None:
    st.title("Quotation Pricing Decision Support Dashboard")
    st.caption(
        "Interactive price-review support for gross-margin management, competitor positioning, "
        "win-probability estimation, and quotation risk review."
    )


# -----------------------------------------------------------------------------
# Load shared resources
# -----------------------------------------------------------------------------
data = load_data()
model, model_metrics, confusion = train_model()

with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Select dashboard section",
        [
            "Executive Overview",
            "Quote Recommender",
            "Margin Sweet Spot",
            "Competitor Positioning",
            "Data Quality & Model Notes",
        ],
    )
    st.divider()
    st.markdown("**Decision-support scope**")
    st.caption(
        "This dashboard supports price review. It does not replace managerial approval, customer knowledge, "
        "or commercial judgement."
    )


# -----------------------------------------------------------------------------
# Executive overview
# -----------------------------------------------------------------------------
if page == "Executive Overview":
    render_title()
    st.subheader("Executive Overview")

    quote_header = data.drop_duplicates(subset=["quote_id"]).copy()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Historical quotations", f"{quote_header['quote_id'].nunique():,}")
    col2.metric("Product categories", f"{data['product'].nunique():,}")
    col3.metric("Historical win rate", f"{quote_header['is_win'].mean():.1%}")
    col4.metric("Rows with competitor information", f"{(data['competitor_count_available'] > 0).mean():.1%}")

    st.info(
        "The dashboard is designed around the project objective: balancing quotation profitability with tender success "
        "while considering product characteristics, energy grants, and competitor price benchmarks."
    )

    left, right = st.columns(2)
    with left:
        st.markdown("#### Historical win rate by gross-margin band")
        margin_summary = (
            quote_header.groupby("gmr_band", observed=False)
            .agg(Quotations=("quote_id", "nunique"), Win_Rate=("is_win", "mean"))
            .reset_index()
        )
        fig = px.bar(
            margin_summary,
            x="gmr_band",
            y="Win_Rate",
            text=margin_summary["Win_Rate"].map(lambda x: f"{x:.1%}"),
            hover_data={"Quotations": True, "Win_Rate": ":.1%"},
            labels={"gmr_band": "Gross Margin Rate", "Win_Rate": "Historical Win Rate"},
        )
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("#### Product-level win rate (minimum 20 observations)")
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
            text=product_summary["Win_Rate"].map(lambda x: f"{x:.1%}"),
            hover_data={"Observations": True, "Win_Rate": ":.1%"},
            labels={"product": "Product", "Win_Rate": "Historical Win Rate"},
        )
        fig.update_layout(xaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Potential lost-tender review signals")
    product_median = data.groupby("product")["gross_margin_rate"].median().to_dict()
    failed = data[data["is_win"] == 0].copy()
    failed["Potential review signal"] = failed.apply(lambda row: potential_loss_signal(row, product_median), axis=1)
    signal_summary = (
        failed["Potential review signal"].value_counts().rename_axis("Potential review signal").reset_index(name="Failed product lines")
    )
    fig = px.bar(signal_summary, x="Potential review signal", y="Failed product lines", text_auto=True)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "These are rule-based review signals, not confirmed causal reasons for tender losses. Additional customer and sales-process data would be required for causal interpretation."
    )


# -----------------------------------------------------------------------------
# Quote recommender
# -----------------------------------------------------------------------------
elif page == "Quote Recommender":
    render_title()
    st.subheader("Interactive Quote Recommender")
    st.write(
        "Enter a product and pricing scenario to estimate the quotation's win probability, compare the proposed "
        "unit price with available competitor benchmarks, and identify a margin recommendation that balances "
        "win probability with expected gross profit."
    )

    products = sorted(data["product"].astype(str).unique().tolist())
    input_left, input_mid, input_right = st.columns([1.0, 1.0, 1.0])

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
            help="The dataset indicates that energy grant values are product-related. Adjust this value when reviewing a specific quotation.",
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
    min_comp = float(scenario["min_competitor_price"].iloc[0]) if pd.notna(scenario["min_competitor_price"].iloc[0]) else np.nan

    st.divider()
    result1, result2, result3, result4 = st.columns(4)
    result1.metric("Proposed unit price", currency(unit_price))
    result2.metric("Estimated win probability", f"{selected_score['win_probability']:.1%}")
    result3.metric("Gross profit if won", currency(selected_score["gross_profit_amount"]))
    result4.metric("Expected gross profit", currency(selected_score["expected_gross_profit"]))

    rec1, rec2, rec3, rec4 = st.columns(4)
    rec1.metric("Recommended GMR", f"{best_row['Gross Margin Rate']:.0%}")
    rec2.metric("Win probability at recommendation", f"{best_row['Win Probability']:.1%}")
    rec3.metric("Recommended unit price", currency(best_row["Unit Price"]))
    rec4.metric("Expected GP at recommendation", currency(best_row["Expected Gross Profit"]))

    st.caption(
        "The recommended margin maximizes model-estimated expected gross profit among scenarios that meet the selected minimum-margin requirement. "
        "It should be reviewed alongside sales knowledge and customer-specific considerations."
    )

    left, right = st.columns([1.5, 1.0])
    with left:
        st.markdown("#### What-if margin simulation")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=grid["Gross Margin Rate"] * 100,
                y=grid["Win Probability"],
                name="Estimated win probability",
                mode="lines",
                yaxis="y1",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=grid["Gross Margin Rate"] * 100,
                y=grid["Expected Gross Profit"],
                name="Expected gross profit",
                mode="lines",
                yaxis="y2",
            )
        )
        fig.add_vline(x=selected_gmr_pct, line_dash="dash", annotation_text="Proposed GMR")
        fig.add_vline(x=float(best_row["Gross Margin Rate"] * 100), line_dash="dot", annotation_text="Recommended GMR")
        fig.update_layout(
            xaxis_title="Gross Margin Rate (%)",
            yaxis=dict(title="Estimated Win Probability", tickformat=".0%"),
            yaxis2=dict(title="Expected Gross Profit", overlaying="y", side="right"),
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("#### Historical benchmark")
        st.markdown(
            f"""
            <div class="card">
                <b>Selected product:</b> {product}<br>
                <b>Gross-margin band:</b> {benchmark['band']}<br>
                <b>Historical records in band:</b> {benchmark['n']:,}<br>
                <b>Historical win rate:</b> {('N/A' if pd.isna(benchmark['historical_win_rate']) else f"{benchmark['historical_win_rate']:.1%}")}<br>
                <b>Price position:</b> {classify_price_position(unit_price, min_comp)}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Historical win rates are descriptive benchmarks and should not be interpreted as causal effects.")

    st.markdown("#### Quote review flags")
    for severity, title, explanation in risk_flags(scenario, selected_score, minimum_margin, review_win_threshold):
        css = "risk-low" if severity == "Low" else ("risk-medium" if severity == "Medium" else "risk-high")
        st.markdown(
            f"<div class='card {css}'><b>{severity}: {title}</b><br>{explanation}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("#### Competitor positioning")
    comps = available_competitors(competitor_a, competitor_b, competitor_c)
    if comps:
        chart_data = pd.DataFrame(
            {
                "Price reference": ["Proposed unit price"] + [f"Competitor {chr(65+i)}" for i, v in enumerate([competitor_a, competitor_b, competitor_c]) if pd.notna(v) and v > 0],
                "Price": [unit_price] + comps,
            }
        )
        fig = px.bar(chart_data, x="Price reference", y="Price", text_auto=",.0f")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No competitor benchmark is available for this scenario. The recommendation should be reviewed with additional market information.")

    with st.expander("View top margin scenarios"):
        display_grid = grid.sort_values("Expected Gross Profit", ascending=False).head(10).copy()
        display_grid["Gross Margin Rate"] = display_grid["Gross Margin Rate"].map(lambda x: f"{x:.0%}")
        display_grid["Win Probability"] = display_grid["Win Probability"].map(lambda x: f"{x:.1%}")
        st.dataframe(display_grid, use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# Margin sweet spot
# -----------------------------------------------------------------------------
elif page == "Margin Sweet Spot":
    render_title()
    st.subheader("Gross-Margin Sweet Spot Analysis")
    st.write(
        "Use this page to compare historical conversion performance and realized gross-profit outcomes across margin ranges. "
        "The recommended margin should be evaluated by product rather than treated as a universal threshold."
    )

    choices = ["All products"] + sorted(data["product"].astype(str).unique().tolist())
    selected_product = st.selectbox("Filter by product", choices)

    if selected_product == "All products":
        # Overall margin belongs to the quotation level; count each Quote ID once.
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

    a, b = st.columns(2)
    with a:
        fig = px.bar(
            summary,
            x="gmr_band",
            y="Historical_Win_Rate",
            text=summary["Historical_Win_Rate"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"),
            hover_data={"Quotations": True},
            labels={"gmr_band": "Gross Margin Rate", "Historical_Win_Rate": "Historical Win Rate"},
        )
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    with b:
        fig = px.bar(
            summary,
            x="gmr_band",
            y="Expected_GP_Proxy",
            text_auto=",.0f",
            hover_data={"Quotations": True},
            labels={"gmr_band": "Gross Margin Rate", "Expected_GP_Proxy": "Historical Expected-GP Proxy"},
        )
        st.plotly_chart(fig, use_container_width=True)

    display_summary = summary.copy()
    display_summary["Historical_Win_Rate"] = display_summary["Historical_Win_Rate"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
    st.dataframe(display_summary, use_container_width=True, hide_index=True)
    st.caption(
        "The expected-GP proxy is descriptive. For actionable quote review, use the Quote Recommender page, which simulates margin scenarios while holding estimated cost constant."
    )


# -----------------------------------------------------------------------------
# Competitor positioning
# -----------------------------------------------------------------------------
elif page == "Competitor Positioning":
    render_title()
    st.subheader("Competitor Positioning Analysis")
    st.write(
        "This section shows how historical win rates vary when the company's unit price is below, approximately aligned with, or above the minimum available competitor benchmark."
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

    left, right = st.columns([1.2, 1.0])
    with left:
        fig = px.bar(
            positioning_summary,
            x="price_position",
            y="Historical_Win_Rate",
            text=positioning_summary["Historical_Win_Rate"].map(lambda x: f"{x:.1%}"),
            hover_data={"Product_Lines": True},
            labels={"price_position": "Price position", "Historical_Win_Rate": "Historical Win Rate"},
        )
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        availability = (
            data["competitor_count_available"].value_counts().sort_index().rename_axis("Available competitor prices").reset_index(name="Product lines")
        )
        fig = px.bar(availability, x="Available competitor prices", y="Product lines", text_auto=True)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Product-level competitor positioning")
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
        hover_data={"Observations": True, "Win_Rate": ":.1%"},
        labels={"product": "Product", "Win_Rate": "Historical Win Rate", "price_position": "Price position"},
    )
    fig.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    st.warning(
        "Competitor analysis is retrospective. Competitor prices may be unavailable during quotation preparation and some values may be learned through customer feedback. "
        "Use competitor features only when the information is available at the price-review stage."
    )


# -----------------------------------------------------------------------------
# Data quality and model notes
# -----------------------------------------------------------------------------
else:
    render_title()
    st.subheader("Data Quality & Model Notes")

    quote_counts = data.groupby("quote_id").size()
    info1, info2, info3, info4 = st.columns(4)
    info1.metric("Clean product-line records", f"{len(data):,}")
    info2.metric("Unique Quote IDs", f"{data['quote_id'].nunique():,}")
    info3.metric("Multi-line Quote IDs", f"{(quote_counts > 1).sum():,}")
    info4.metric("Negative-margin records", f"{(data['gross_margin_rate'] < 0).sum():,}")

    st.markdown("#### Competitor-information completeness")
    missing_table = pd.DataFrame(
        {
            "Field": ["Competitor A", "Competitor B", "Competitor C"],
            "Missing records": [data["competitor_a"].isna().sum(), data["competitor_b"].isna().sum(), data["competitor_c"].isna().sum()],
            "Missing rate": [data["competitor_a"].isna().mean(), data["competitor_b"].isna().mean(), data["competitor_c"].isna().mean()],
        }
    )
    missing_table["Missing rate"] = missing_table["Missing rate"].map(lambda x: f"{x:.1%}")
    st.dataframe(missing_table, use_container_width=True, hide_index=True)

    st.markdown("#### Model validation summary")
    st.write(
        "The dashboard uses an explainable logistic-regression baseline. The holdout evaluation separates records by Quote ID, so product lines from the same quotation do not appear in both training and test sets. Each Quote ID is weighted equally during training."
    )
    metric_cols = st.columns(len(model_metrics))
    for col, (name, value) in zip(metric_cols, model_metrics.items()):
        col.metric(name, f"{value:.3f}")

    cm_df = pd.DataFrame(confusion, index=["Actual Loss", "Actual Win"], columns=["Predicted Loss", "Predicted Win"])
    st.dataframe(cm_df, use_container_width=True)

    st.markdown("#### Important interpretation notes")
    st.markdown(
        """
        - The dashboard estimates **associations**, not causal effects.
        - The primary use case is **during price review**, before quotation submission.
        - The dashboard works at the **product-line price-review level**. A quotation containing multiple products still requires managerial review at the full quotation level.
        - Missing competitor values are not replaced with zero. The app explicitly records whether competitor information is available.
        - Negative gross-margin records are retained because they may represent deliberate relationship-building strategies.
        - Quantity values less than or equal to zero are excluded because they were identified as system errors.
        - For repeated records with the same Quote ID and product, the highest unit price is retained following the interview guidance regarding price-list updates.
        """
    )

    st.markdown("#### Recommended next data improvements")
    st.markdown(
        """
        1. Add quotation timestamps and revision numbers to distinguish price updates from product-line records.
        2. Record whether competitor prices were known before quotation submission or learned after the tender result.
        3. Add customer segment, salesperson, region, and customer-history variables.
        4. Add full quotation-level identifiers and product-line identifiers.
        5. Record the final commercial reason for a won or lost tender when available.
        """
    )

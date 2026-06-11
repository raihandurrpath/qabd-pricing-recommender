from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =============================================================================
# APP CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Quotation Pricing Decision Support",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_VERSION = "v3.7.0"
LAST_UPDATED = "2026-06-12"

BASE_DIR = Path(__file__).resolve().parent
ASSET_DIR = BASE_DIR / "asset"
DATA_PATH = ASSET_DIR / "df_preprocessed.csv"
MODEL_BUNDLE_PATH = ASSET_DIR / "final_gmr_recommender_bundle.joblib"
SHAP_IMPORTANCE_PATH = ASSET_DIR / "shap_global_feature_importance.csv"
SHAP_IMAGE_PATH = ASSET_DIR / "shap_global_feature_importance.png"
HGB_RESULTS_PATH = ASSET_DIR / "5fold_hgb_results.csv"
HGB_SUMMARY_PATH = ASSET_DIR / "5fold_hgb_summary.csv"

MIN_PRODUCT_COUNT = 15
TARGET = "success"

GMR_BINS = [-np.inf, 0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, np.inf]
GMR_LABELS = ["< 0%", "0–10%", "10–20%", "20–30%", "30–40%", "40–50%", "50–60%", "> 60%"]

# Fallback display metrics from latest project run. If CSV files are available
# in asset/, the app will automatically use those instead.
DEFAULT_MODEL_METRICS = {
    "Accuracy": 0.920958,
    "Precision (Win)": 0.890952,
    "Recall (Win)": 0.752135,
    "F1 (Win)": 0.815280,
    "Brier Score": 0.060955,
    "ROC-AUC": 0.951074,
    "PR-AUC": 0.903076,
    "Log Loss": 0.220352,
}
DEFAULT_CONFUSION_MATRIX = [[3607, 104], [278, 844]]

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
    .stApp { background: var(--light-bg); }
    .block-container {
        padding-top: 4.5rem !important;
        padding-bottom: 2.25rem;
        max-width: 1480px;
    }
    header[data-testid="stHeader"] { background: rgba(245, 247, 251, 0.92); }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #172554 0%, #1E3A8A 100%);
        border-right: 0;
    }
    section[data-testid="stSidebar"] * { color: #F8FAFC; }
    .page-title {
        color: #172033; font-size: 2.45rem; font-weight: 800;
        line-height: 1.18; margin: 0 0 0.3rem 0;
        letter-spacing: -0.035rem;
    }
    .page-subtitle { color: #667085; font-size: 0.98rem; margin-bottom: 1rem; }
    .section-title { color: #172033; font-size: 1.2rem; font-weight: 800; margin: 0 0 0.15rem 0; }
    .section-caption { color: #667085; font-size: 0.9rem; margin: 0 0 0.7rem 0; }
    .metric-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
        border: 1px solid #E4E7EC; border-top: 4px solid var(--accent, #2563EB);
        border-radius: 18px; box-shadow: 0 4px 14px rgba(16, 24, 40, 0.045);
        padding: 0.9rem 1rem 0.85rem 1rem; min-height: 126px; margin-bottom: 0.3rem;
    }
    .metric-card.compact { min-height: 108px; }
    .metric-topline { display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.35rem; }
    .metric-label { color: #667085; font-size: 0.84rem; font-weight: 700; }
    .metric-icon { background: #EEF4FF; border-radius: 11px; padding: 0.35rem 0.5rem; font-size: 1rem; }
    .metric-value { color: #101828; font-size: 1.8rem; font-weight: 800; line-height: 1.12; }
    .metric-footnote { color: #667085; font-size: 0.78rem; line-height: 1.35; margin-top: 0.45rem; }
    .pill { display: inline-block; border-radius: 999px; padding: 0.22rem 0.62rem; font-size: 0.74rem; font-weight: 700; margin-bottom: 0.45rem; }
    .pill-blue { background: #DBEAFE; color: #1D4ED8; }
    .pill-green { background: #DCFCE7; color: #15803D; }
    .pill-purple { background: #EDE9FE; color: #6D28D9; }
    .pill-amber { background: #FEF3C7; color: #B45309; }
    .insight-box {
        background: #EFF6FF; border-left: 5px solid #2563EB; border-radius: 14px;
        color: #1D4ED8; font-size: 0.91rem; line-height: 1.5;
        padding: 0.85rem 1rem; margin: 0.65rem 0;
    }
    .soft-box { background: #F8FAFC; border: 1px solid #E4E7EC; border-radius: 16px; padding: 0.85rem 1rem; margin-bottom: 0.55rem; }
    .soft-box ul, .soft-box ol { padding-left: 1.1rem; margin: 0.2rem 0 0.1rem 0; }
    .soft-box li { color: #475467; font-size: 0.88rem; line-height: 1.45; margin-bottom: 0.3rem; }
    .risk-card {
        background: #FFFFFF; border: 1px solid #E4E7EC; border-left: 5px solid var(--risk-color, #64748B);
        border-radius: 14px; padding: 0.7rem 0.9rem; margin-bottom: 0.55rem;
    }
    .risk-card b { color: #172033; }
    .risk-card span { color: #667085; font-size: 0.86rem; }
    .sidebar-badge { background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.18); border-radius: 12px; padding: 0.55rem 0.7rem; color: #F8FAFC; font-size: 0.78rem; line-height: 1.45; }
    div[data-testid="stMetric"] { background: #FFFFFF; border: 1px solid #E4E7EC; border-radius: 14px; padding: 0.7rem 0.8rem; }
    div[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# BASIC UTILITIES
# =============================================================================
def normalize_decimal_or_pct(value: float) -> float:
    value = float(value)
    if abs(value) > 1:
        value = value / 100.0
    return value


def build_gmr_grid(gmr_min: float, gmr_max: float, gmr_interval: float) -> np.ndarray:
    gmr_min = normalize_decimal_or_pct(gmr_min)
    gmr_max = normalize_decimal_or_pct(gmr_max)
    gmr_interval = normalize_decimal_or_pct(gmr_interval)
    if gmr_interval <= 0:
        raise ValueError("GMR interval must be positive.")
    if gmr_min > gmr_max:
        raise ValueError("Minimum GMR must be less than or equal to maximum GMR.")
    if gmr_max >= 0.95:
        raise ValueError("Maximum GMR must be below 95% because price = cost / (1 - GMR).")
    grid = np.arange(gmr_min, gmr_max + (gmr_interval / 2), gmr_interval)
    grid = np.round(grid, 10)
    return grid[grid < 0.95]


def currency(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:,.0f}"


def signed_currency(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:+,.0f}"


def pct(value: float, digits: int = 1) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.{digits}%}"


def pct_from_point(value: float, digits: int = 1) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.{digits}f}%"


def to_numeric_if_exists(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def safe_float(value, default: float = np.nan) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def positive_or_nan(value) -> float:
    value = safe_float(value)
    if pd.isna(value) or value <= 0:
        return np.nan
    return value

# =============================================================================
# MODEL, DATA, AND FEATURE ENGINEERING
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_model_bundle() -> Dict:
    if not MODEL_BUNDLE_PATH.exists():
        raise FileNotFoundError(
            f"Model bundle not found: {MODEL_BUNDLE_PATH}. Put final_gmr_recommender_bundle.joblib inside the asset folder."
        )
    bundle = joblib.load(MODEL_BUNDLE_PATH)
    if not isinstance(bundle, dict):
        raise ValueError("The model bundle must be a dictionary created by pipeline_with_shap.py.")
    if "model" not in bundle or "features" not in bundle:
        raise ValueError("The model bundle must contain at least 'model' and 'features'.")
    return bundle


def recalculate_price_related_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = [
        "unit_price", "qty", "subtotal_price", "gross_margin_rate", "energy_grant_amount",
        "competitor_a", "competitor_b", "competitor_c", "avg_competitor_price",
        "min_competitor_price", "max_competitor_price", "estimated_cost",
    ]
    df = to_numeric_if_exists(df, numeric_cols)

    if "energy_grant_amount" not in df.columns:
        df["energy_grant_amount"] = 0.0
    df["energy_grant_amount"] = df["energy_grant_amount"].fillna(0.0)

    if "estimated_cost" not in df.columns:
        df["estimated_cost"] = df["subtotal_price"] * (1 - df["gross_margin_rate"])

    df["estimated_gross_profit"] = df["subtotal_price"] - df["estimated_cost"]
    df["gross_profit_amount"] = df["estimated_gross_profit"]
    df["effective_price_after_grant"] = df["subtotal_price"] - df["energy_grant_amount"]
    df["grant_ratio_to_subtotal"] = np.where(
        df["subtotal_price"].notna() & (df["subtotal_price"] != 0),
        df["energy_grant_amount"] / df["subtotal_price"],
        np.nan,
    )

    df["price_gap_avg_competitor"] = df["unit_price"] - df["avg_competitor_price"]
    df["price_gap_avg_competitor_pct"] = np.where(
        df["avg_competitor_price"].notna() & (df["avg_competitor_price"] != 0),
        (df["price_gap_avg_competitor"] / df["avg_competitor_price"]) * 100,
        np.nan,
    )
    df["price_gap_min_competitor"] = df["unit_price"] - df["min_competitor_price"]
    df["price_gap_min_competitor_pct"] = np.where(
        df["min_competitor_price"].notna() & (df["min_competitor_price"] != 0),
        (df["price_gap_min_competitor"] / df["min_competitor_price"]) * 100,
        np.nan,
    )
    df["higher_than_avg_competitor"] = np.where(
        df["avg_competitor_price"].notna(), (df["unit_price"] > df["avg_competitor_price"]).astype(int), np.nan
    )
    df["is_lower_than_competitor"] = np.where(
        df["min_competitor_price"].notna(), (df["unit_price"] < df["min_competitor_price"]).astype(int), np.nan
    )
    return df


def add_missing_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    required = ["quote_id", "product", "unit_price", "qty", "subtotal_price", "gross_margin_rate", "convert_to_order"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    for col in ["competitor_a", "competitor_b", "competitor_c"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = to_numeric_if_exists(
        df,
        ["kw", "unit_price", "qty", "subtotal_price", "gross_margin_rate", "energy_grant_amount"],
    )

    if "energy_grant_amount" not in df.columns:
        df["energy_grant_amount"] = 0.0
    df["energy_grant_amount"] = df["energy_grant_amount"].fillna(0.0)

    for raw_col, flag_col in [
        ("competitor_a", "is_compe_a"),
        ("competitor_b", "is_compe_b"),
        ("competitor_c", "is_compe_c"),
    ]:
        if flag_col not in df.columns:
            df[flag_col] = df[raw_col].notna().astype(int)

    if "competitor_count_available" not in df.columns:
        df["competitor_count_available"] = df[["competitor_a", "competitor_b", "competitor_c"]].notna().sum(axis=1)
    if "known_num_compe" not in df.columns:
        df["known_num_compe"] = df["competitor_count_available"]

    if "avg_competitor_price" not in df.columns:
        df["avg_competitor_price"] = df[["competitor_a", "competitor_b", "competitor_c"]].mean(axis=1)
    if "min_competitor_price" not in df.columns:
        df["min_competitor_price"] = df[["competitor_a", "competitor_b", "competitor_c"]].min(axis=1)
    if "max_competitor_price" not in df.columns:
        df["max_competitor_price"] = df[["competitor_a", "competitor_b", "competitor_c"]].max(axis=1)

    if "price_order" not in df.columns:
        df["price_order"] = (
            df.groupby(["quote_id", "product"])["unit_price"].rank(method="dense", ascending=False).astype("Int64")
        )
    if "is_highest_price" not in df.columns:
        df["is_highest_price"] = (df["price_order"] == 1).astype(int)

    if "estimated_gross_profit" not in df.columns:
        df["estimated_gross_profit"] = df["subtotal_price"] * df["gross_margin_rate"]
    if "gross_profit_amount" not in df.columns:
        df["gross_profit_amount"] = df["estimated_gross_profit"]
    if "estimated_cost" not in df.columns:
        df["estimated_cost"] = df["subtotal_price"] - df["estimated_gross_profit"]

    df = recalculate_price_related_features(df)
    return df


def clean_dataset_for_dashboard(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Quantity <= 0 is treated as a data-entry/system error.
    if "qty" in df.columns:
        df = df[df["qty"].notna() & (df["qty"] > 0)].copy()

    # Match the model scripts: only keep highest-price rows when duplicate
    # quote_id + product rows have price variations. If the price is identical,
    # keep all rows.
    price_variation = df.groupby(["quote_id", "product"])["unit_price"].transform("nunique")
    df = df[(price_variation == 1) | ((price_variation > 1) & (df["is_highest_price"] == 1))].copy()

    df = df[df["convert_to_order"].isin([0, 1])].copy()
    df[TARGET] = (df["convert_to_order"] == 0).astype(int)
    df["is_win"] = df[TARGET]
    df["gmr_pct"] = df["gross_margin_rate"] * 100
    df["gmr_band"] = pd.cut(df["gross_margin_rate"], bins=GMR_BINS, labels=GMR_LABELS, right=False)
    df["estimated_cost_per_unit"] = np.where(
        df["qty"].notna() & (df["qty"] != 0), df["estimated_cost"] / df["qty"], np.nan
    )
    df["multi_product_quote"] = df.groupby("quote_id")["product"].transform("nunique") > 1

    product_counts = df["product"].value_counts()
    rare_products = product_counts[product_counts < MIN_PRODUCT_COUNT].index
    df["product_model"] = df["product"].replace(rare_products, "Other")

    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}. Put df_preprocessed.csv inside the asset folder.")
    df = pd.read_csv(DATA_PATH)
    df = add_missing_feature_engineering(df)
    df = clean_dataset_for_dashboard(df)
    return df


def get_model_feature_lists(bundle: Dict) -> Tuple[List[str], List[str], List[str]]:
    features = list(bundle["features"])
    categorical_features = list(bundle.get("categorical_features", ["product_model"]))
    numeric_features = list(bundle.get("numeric_features", [c for c in features if c not in categorical_features]))
    return features, categorical_features, numeric_features


def prepare_for_model(df: pd.DataFrame, bundle: Dict) -> pd.DataFrame:
    features, categorical_features, numeric_features = get_model_feature_lists(bundle)
    out = df.copy()

    for col in features:
        if col not in out.columns:
            # Safe fallbacks for features that may exist in older bundles.
            if col == "price_order":
                out[col] = 1
            elif col == "is_highest_price":
                out[col] = 1
            elif col in ["is_compe_a", "is_compe_b", "is_compe_c", "competitor_count_available", "known_num_compe"]:
                out[col] = 0
            else:
                out[col] = np.nan

    for col in numeric_features:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in categorical_features:
        if col in out.columns:
            out[col] = out[col].astype("string").fillna("Missing")

    return out[features].copy()


def predict_success_probability(model, X: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(X)
    classifier = model.named_steps.get("classifier") if hasattr(model, "named_steps") else None
    classes = getattr(classifier, "classes_", None)
    if classes is not None and 1 in list(classes):
        positive_idx = list(classes).index(1)
    else:
        positive_idx = 1
    return proba[:, positive_idx]


def predict_model(bundle: Dict, df: pd.DataFrame) -> np.ndarray:
    model = bundle["model"]
    X = prepare_for_model(df, bundle)
    return predict_success_probability(model, X)


@st.cache_data(show_spinner=False)
def load_hgb_metrics() -> Tuple[Dict[str, float], List[List[int]]]:
    metrics = DEFAULT_MODEL_METRICS.copy()
    confusion = [row[:] for row in DEFAULT_CONFUSION_MATRIX]

    if HGB_SUMMARY_PATH.exists():
        try:
            summary = pd.read_csv(HGB_SUMMARY_PATH)
            metric_map = {
                "accuracy": "Accuracy",
                "precision": "Precision (Win)",
                "recall": "Recall (Win)",
                "f1_score": "F1 (Win)",
                "brier_score": "Brier Score",
                "roc_auc": "ROC-AUC",
                "pr_auc": "PR-AUC",
                "log_loss": "Log Loss",
            }
            for raw, label in metric_map.items():
                value = summary.loc[summary["metric"] == raw, "mean"]
                if not value.empty:
                    metrics[label] = float(value.iloc[0])
        except Exception:
            pass

    if HGB_RESULTS_PATH.exists():
        try:
            results = pd.read_csv(HGB_RESULTS_PATH)
            cm = results[["tn", "fp", "fn", "tp"]].sum()
            confusion = [[int(cm["tn"]), int(cm["fp"])], [int(cm["fn"]), int(cm["tp"])]]
        except Exception:
            pass

    return metrics, confusion

# =============================================================================
# QUOTE INPUT AND RECOMMENDATION LOGIC
# =============================================================================
def product_model_options(dataset: pd.DataFrame) -> List[str]:
    opts = sorted([str(x) for x in dataset["product_model"].dropna().unique().tolist() if str(x) != "Other"])
    if "Other" not in opts:
        opts.append("Other")
    return opts


def product_defaults(dataset: pd.DataFrame, product_model: str) -> Dict[str, float]:
    if product_model == "Other":
        subset = dataset[dataset["product_model"] == "Other"].copy()
        if subset.empty:
            subset = dataset.copy()
    else:
        subset = dataset[dataset["product_model"].astype(str) == str(product_model)].copy()
        if subset.empty:
            subset = dataset.copy()

    def med(col: str, default: float = 0.0) -> float:
        if col not in subset.columns:
            return default
        val = subset[col].dropna()
        if val.empty:
            return default
        return float(val.median())

    return {
        "product": product_model,
        "kw": med("kw", 1.0),
        "qty": 1,
        "cost_per_unit": med("estimated_cost_per_unit", 1.0),
        "proposed_gmr_pct": med("gross_margin_rate", 0.30) * 100,
        "grant": med("energy_grant_amount", 0.0),
        "competitor_a": med("competitor_a", 0.0),
        "competitor_b": med("competitor_b", 0.0),
        "competitor_c": med("competitor_c", 0.0),
    }


def make_default_quote_row(dataset: pd.DataFrame, product: Optional[str] = None) -> Dict:
    options = product_model_options(dataset)
    chosen = product or ("H1" if "H1" in options else options[0])
    d = product_defaults(dataset, chosen)
    return {
        "Product": d["product"],
        "kW": round(d["kw"], 3),
        "Qty": int(max(1, round(d["qty"]))),
        "Cost per Unit": round(d["cost_per_unit"], 2),
        "Input GMR (%)": round(d["proposed_gmr_pct"], 2),
        "Energy Grant": round(d["grant"], 2),
        "Competitor A": 0.0 if pd.isna(d["competitor_a"]) else round(d["competitor_a"], 2),
        "Competitor B": 0.0 if pd.isna(d["competitor_b"]) else round(d["competitor_b"], 2),
        "Competitor C": 0.0 if pd.isna(d["competitor_c"]) else round(d["competitor_c"], 2),
    }


def build_line_scenario(line: Dict, candidate_gmr: float, row_id: int) -> pd.DataFrame:
    product_model = str(line.get("Product", "Other"))
    if product_model.strip() == "" or product_model.lower() == "nan":
        product_model = "Other"

    qty = max(int(safe_float(line.get("Qty", 1), 1)), 1)
    cost_per_unit = max(safe_float(line.get("Cost per Unit", 1.0), 1.0), 0.0)
    kw = safe_float(line.get("kW", np.nan))
    energy_grant_amount = max(safe_float(line.get("Energy Grant", 0.0), 0.0), 0.0)
    candidate_gmr = normalize_decimal_or_pct(candidate_gmr)
    if candidate_gmr >= 0.95:
        raise ValueError("Gross margin rate must be below 95%.")

    competitor_a = positive_or_nan(line.get("Competitor A", np.nan))
    competitor_b = positive_or_nan(line.get("Competitor B", np.nan))
    competitor_c = positive_or_nan(line.get("Competitor C", np.nan))
    competitors = [x for x in [competitor_a, competitor_b, competitor_c] if pd.notna(x)]

    estimated_cost = cost_per_unit * qty
    subtotal_price = estimated_cost / (1 - candidate_gmr) if candidate_gmr < 0.95 else np.nan
    unit_price = subtotal_price / qty if qty else np.nan

    avg_comp = float(np.mean(competitors)) if competitors else np.nan
    min_comp = float(np.min(competitors)) if competitors else np.nan
    max_comp = float(np.max(competitors)) if competitors else np.nan

    row = {
        "row_id": row_id,
        "quote_id": "WEB_QUOTE",
        "product": product_model,
        "product_model": product_model,
        "kw": kw,
        "qty": qty,
        "unit_price": unit_price,
        "subtotal_price": subtotal_price,
        "gross_margin_rate": candidate_gmr,
        "energy_grant_amount": energy_grant_amount,
        "competitor_a": competitor_a,
        "competitor_b": competitor_b,
        "competitor_c": competitor_c,
        "is_compe_a": int(pd.notna(competitor_a)),
        "is_compe_b": int(pd.notna(competitor_b)),
        "is_compe_c": int(pd.notna(competitor_c)),
        "competitor_count_available": len(competitors),
        "known_num_compe": len(competitors),
        "avg_competitor_price": avg_comp,
        "min_competitor_price": min_comp,
        "max_competitor_price": max_comp,
        "estimated_cost": estimated_cost,
        "price_order": 1,
        "is_highest_price": 1,
    }
    out = pd.DataFrame([row])
    out = recalculate_price_related_features(out)
    return out


def create_line_grid(line: Dict, row_id: int, gmr_grid: np.ndarray, bundle: Dict) -> pd.DataFrame:
    scenarios = pd.concat([build_line_scenario(line, gmr, row_id) for gmr in gmr_grid], ignore_index=True)
    scenarios["predicted_win_probability"] = predict_model(bundle, scenarios)
    scenarios["predicted_win_probability_pct"] = scenarios["predicted_win_probability"] * 100
    scenarios["candidate_estimated_gross_profit"] = scenarios["estimated_gross_profit"]
    scenarios["expected_gross_profit"] = scenarios["predicted_win_probability"] * scenarios["candidate_estimated_gross_profit"]
    return scenarios


def select_best_line_candidate(
    scenarios: pd.DataFrame,
    minimum_margin: float,
    minimum_win_probability: float,
) -> Tuple[pd.Series, bool, str]:
    pool = scenarios.copy()
    eligible = pool[(pool["gross_margin_rate"] >= minimum_margin) & (pool["predicted_win_probability"] >= minimum_win_probability)].copy()
    used_fallback = False
    note = "Selected from candidates that satisfy the selected margin and win-probability rules."

    if eligible.empty:
        eligible = pool[pool["gross_margin_rate"] >= minimum_margin].copy()
        used_fallback = True
        note = "No candidate satisfied both rules; selected the best expected gross profit among candidates that satisfy the margin rule."

    if eligible.empty:
        eligible = pool.copy()
        used_fallback = True
        note = "No candidate satisfied the selected rules; selected the best expected gross profit from the tested GMR grid."

    eligible = eligible.sort_values(
        ["expected_gross_profit", "predicted_win_probability", "gross_margin_rate"],
        ascending=[False, False, True],
    )
    return eligible.iloc[0], used_fallback, note


def evaluate_quote(
    quote_rows: pd.DataFrame,
    bundle: Dict,
    gmr_min: float,
    gmr_max: float,
    gmr_interval: float,
    minimum_margin: float,
    minimum_win_probability: float,
) -> Dict:
    gmr_grid = build_gmr_grid(gmr_min, gmr_max, gmr_interval)
    if len(gmr_grid) == 0:
        raise ValueError("The GMR grid is empty.")
    if minimum_margin > float(np.max(gmr_grid)) + 1e-9:
        raise ValueError(
            f"Minimum target GMR ({minimum_margin:.0%}) is higher than the maximum tested GMR "
            f"({float(np.max(gmr_grid)):.0%}). Please increase the GMR grid max or lower the minimum target GMR."
        )

    clean_rows = quote_rows.copy()
    clean_rows = clean_rows.dropna(subset=["Product", "Qty", "Cost per Unit"], how="any")
    clean_rows = clean_rows[clean_rows["Product"].astype(str).str.strip() != ""].copy()
    clean_rows = clean_rows[pd.to_numeric(clean_rows["Qty"], errors="coerce") > 0].copy()
    clean_rows = clean_rows[pd.to_numeric(clean_rows["Cost per Unit"], errors="coerce") >= 0].copy()

    if clean_rows.empty:
        raise ValueError("Please enter at least one valid product line.")

    scenario_parts: List[pd.DataFrame] = []
    rec_rows: List[Dict] = []
    notes: List[str] = []

    for idx, (_, line) in enumerate(clean_rows.reset_index(drop=True).iterrows(), start=1):
        line_dict = line.to_dict()
        scenarios = create_line_grid(line_dict, row_id=idx, gmr_grid=gmr_grid, bundle=bundle)
        selected, used_fallback, note = select_best_line_candidate(scenarios, minimum_margin, minimum_win_probability)

        input_gmr = safe_float(line_dict.get("Input GMR (%)", 0.0), 0.0) / 100.0
        input_scenario = build_line_scenario(line_dict, input_gmr, row_id=idx)
        input_prob = float(predict_model(bundle, input_scenario)[0])

        selected_dict = selected.to_dict()
        selected_dict.update(
            {
                "line_no": idx,
                "input_gmr": input_gmr,
                "input_predicted_win_probability": input_prob,
                "used_fallback": used_fallback,
                "selection_note": note,
            }
        )
        rec_rows.append(selected_dict)
        scenario_parts.append(scenarios.assign(line_no=idx, input_gmr=input_gmr, input_predicted_win_probability=input_prob))
        if used_fallback:
            notes.append(f"Line {idx}: {note}")

    recommendations = pd.DataFrame(rec_rows)
    all_scenarios = pd.concat(scenario_parts, ignore_index=True)

    # Add rule/selection indicators for the visible scenario grid.
    all_scenarios["meets_min_gmr"] = all_scenarios["gross_margin_rate"] >= minimum_margin
    all_scenarios["meets_min_win_probability"] = all_scenarios["predicted_win_probability"] >= minimum_win_probability
    all_scenarios["meets_all_rules"] = all_scenarios["meets_min_gmr"] & all_scenarios["meets_min_win_probability"]
    selected_keys = {
        (int(row["line_no"]), round(float(row["gross_margin_rate"]), 10))
        for _, row in recommendations.iterrows()
    }
    all_scenarios["selected_candidate"] = all_scenarios.apply(
        lambda row: (int(row["line_no"]), round(float(row["gross_margin_rate"]), 10)) in selected_keys,
        axis=1,
    )

    quote_win_probability = float(recommendations["predicted_win_probability"].mean())
    quote_total_expected_gp = float(recommendations["expected_gross_profit"].sum())
    quote_total_gp_if_won = float(recommendations["candidate_estimated_gross_profit"].sum())
    quote_total_subtotal = float(recommendations["subtotal_price"].sum())
    quote_weighted_gmr = float(
        np.average(
            recommendations["gross_margin_rate"],
            weights=np.where(recommendations["subtotal_price"].abs() > 0, recommendations["subtotal_price"].abs(), 1),
        )
    )

    return {
        "quote_rows": clean_rows.reset_index(drop=True),
        "recommendations": recommendations,
        "scenarios": all_scenarios,
        "gmr_grid": gmr_grid,
        "minimum_margin": minimum_margin,
        "minimum_win_probability": minimum_win_probability,
        "quote_summary": {
            "quote_win_probability": quote_win_probability,
            "quote_total_expected_gp": quote_total_expected_gp,
            "quote_total_gp_if_won": quote_total_gp_if_won,
            "quote_total_subtotal": quote_total_subtotal,
            "quote_weighted_gmr": quote_weighted_gmr,
            "line_count": int(len(recommendations)),
        },
        "notes": notes,
    }


def historical_benchmark(dataset: pd.DataFrame, product_model: str, gmr: float) -> Dict[str, float]:
    """Product-line historical benchmark for the selected product and GMR band.

    This is intentionally product-level, not quote-level. For ordinary product
    codes such as H1, the benchmark uses the raw product column. For the grouped
    Other category, it uses product_model == "Other".
    """
    band = pd.cut(pd.Series([gmr]), bins=GMR_BINS, labels=GMR_LABELS, right=False).iloc[0]
    product_key = str(product_model)

    if product_key == "Other":
        product_subset = dataset[dataset["product_model"].astype(str) == "Other"].copy()
    else:
        product_subset = dataset[dataset["product"].astype(str) == product_key].copy()
        if product_subset.empty:
            product_subset = dataset[dataset["product_model"].astype(str) == product_key].copy()

    band_subset = product_subset[product_subset["gmr_band"] == band].copy()

    return {
        "band": str(band),
        "n": int(len(band_subset)),
        "historical_win_rate": np.nan if band_subset.empty else float(band_subset["is_win"].mean()),
        "product_total_records": int(len(product_subset)),
        "product_overall_win_rate": np.nan if product_subset.empty else float(product_subset["is_win"].mean()),
    }


def classify_price_position(unit_price: float, min_competitor: float) -> str:
    if pd.isna(min_competitor):
        return "Competitor information unavailable"
    if min_competitor == 0:
        return "Competitor information unavailable"
    price_gap_ratio = (unit_price - min_competitor) / min_competitor
    if price_gap_ratio < -0.05:
        return "Below competitor benchmark"
    if price_gap_ratio <= 0.05:
        return "Approximately aligned (±5%)"
    return "Above competitor benchmark"


def quote_review_flags(result: Dict) -> List[Tuple[str, str, str]]:
    summary = result["quote_summary"]
    rec = result["recommendations"]
    flags: List[Tuple[str, str, str]] = []
    min_win = result["minimum_win_probability"]
    min_margin = result["minimum_margin"]

    if summary["quote_win_probability"] < min_win:
        flags.append(("Medium", "Quote-level conversion review", f"Mean product-line win probability is below the selected rule of {min_win:.0%}."))
    if (rec["gross_margin_rate"] < min_margin).any():
        flags.append(("Medium", "Margin rule review", f"At least one recommended product line is below the selected minimum GMR of {min_margin:.0%}."))
    if (rec["competitor_count_available"] == 0).any():
        flags.append(("Medium", "Competitor benchmark gap", "At least one product line has no competitor benchmark available."))
    if rec["price_gap_min_competitor_pct"].fillna(-np.inf).gt(5).any():
        flags.append(("High", "Competitor positioning review", "At least one recommended unit price is more than 5% above the minimum available competitor benchmark."))
    if any(rec.get("used_fallback", pd.Series([False] * len(rec))).astype(bool)):
        flags.append(("Medium", "Rule constraint fallback", "At least one product line had no candidate satisfying all selected rules."))
    if not flags:
        flags.append(("Low", "No major rule-based warning", "The quote-level recommendation does not trigger the current review rules."))
    return flags


def potential_loss_signal(row: pd.Series, product_median: Dict[str, float]) -> str:
    if pd.notna(row.get("price_gap_min_competitor_pct")) and row["price_gap_min_competitor_pct"] > 5:
        return "Above competitor benchmark"
    if row["gross_margin_rate"] > max(0.40, product_median.get(row["product"], 0.40)):
        return "High margin for product"
    if row["competitor_count_available"] == 0:
        return "Competitor information unavailable"
    return "General review signal"

# =============================================================================
# UI HELPERS
# =============================================================================
def render_page_header(title: str, subtitle: str) -> None:
    st.markdown(f"<div class='page-title'>{title}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='page-subtitle'>{subtitle}</div>", unsafe_allow_html=True)


def render_title() -> None:
    render_page_header(
        "Quotation Pricing Decision Support Dashboard",
        "Interactive price-review support for gross-margin management, competitor positioning, win-probability estimation, and quote-level recommendation.",
    )


def render_section_header(title: str, caption: str = "") -> None:
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    if caption:
        st.markdown(f"<div class='section-caption'>{caption}</div>", unsafe_allow_html=True)


def render_metric_card(label: str, value: str, footnote: str, accent: str = BLUE, icon: str = "📌", compact: bool = False) -> None:
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


def render_what_if_chart(result: Dict, selected_line_no: int) -> None:
    scenarios = result["scenarios"]
    line_grid = scenarios[scenarios["line_no"] == selected_line_no].copy().sort_values("gross_margin_rate")
    if line_grid.empty:
        st.warning("No what-if scenario is available for this product line.")
        return

    recommended = result["recommendations"][result["recommendations"]["line_no"] == selected_line_no].iloc[0]
    input_gmr = float(recommended["input_gmr"])
    rec_gmr_pct = float(recommended["gross_margin_rate"]) * 100
    min_win_prob = float(result["minimum_win_probability"])

    line_grid["gmr_pct"] = line_grid["gross_margin_rate"] * 100
    line_grid["win_probability_pct"] = line_grid["predicted_win_probability"] * 100

    # Robust aliases for scenario-grid columns.
    # Older result objects may use candidate_estimated_gross_profit, while the chart
    # wording uses gross_profit_if_won. Keep both names available.
    if "gross_profit_if_won" not in line_grid.columns:
        if "candidate_estimated_gross_profit" in line_grid.columns:
            line_grid["gross_profit_if_won"] = line_grid["candidate_estimated_gross_profit"]
        elif "estimated_gross_profit" in line_grid.columns:
            line_grid["gross_profit_if_won"] = line_grid["estimated_gross_profit"]
        else:
            line_grid["gross_profit_if_won"] = np.nan

    if "meets_all_rules" in line_grid.columns:
        line_grid["eligible_for_recommendation"] = line_grid["meets_all_rules"].fillna(False).astype(bool)
    elif {"meets_min_gmr", "meets_min_win_probability"}.issubset(line_grid.columns):
        line_grid["eligible_for_recommendation"] = (
            line_grid["meets_min_gmr"].fillna(False).astype(bool)
            & line_grid["meets_min_win_probability"].fillna(False).astype(bool)
        )
    else:
        line_grid["eligible_for_recommendation"] = False

    line_grid["meets_both_rules_label"] = np.where(
        line_grid["eligible_for_recommendation"], "Eligible", "Not eligible"
    )

    # Main chart: show Gross Profit if Won because it follows the direct
    # pricing/margin formula and is easier for business users to interpret.
    # Expected gross profit remains available in the hover details and scenario grid.
    fig_profit = go.Figure()
    fig_profit.add_trace(
        go.Scatter(
            x=line_grid["gmr_pct"],
            y=line_grid["gross_profit_if_won"],
            name="Gross profit if won",
            mode="lines+markers",
            line=dict(color=PURPLE, width=4),
            marker=dict(
                size=7,
                color=np.where(line_grid["eligible_for_recommendation"], GREEN, SLATE),
            ),
            fill="tozeroy",
            fillcolor="rgba(124, 58, 237, 0.10)",
            customdata=np.stack(
                [
                    line_grid["win_probability_pct"],
                    line_grid["expected_gross_profit"],
                    line_grid["meets_both_rules_label"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "GMR: %{x:.0f}%<br>"
                "Gross profit if won: %{y:,.0f}<br>"
                "Win probability: %{customdata[0]:.1f}%<br>"
                "Expected gross profit: %{customdata[1]:,.0f}<br>"
                "Rule status: %{customdata[2]}<extra></extra>"
            ),
        )
    )
    fig_profit.add_vline(x=input_gmr * 100, line_dash="dash", line_color=BLUE, annotation_text="Input GMR")
    fig_profit.add_vline(x=rec_gmr_pct, line_dash="dot", line_color=GREEN, annotation_text="Recommended GMR")
    fig_profit.update_layout(
        xaxis_title="Gross Margin Rate (%)",
        yaxis=dict(title="Gross Profit if Won"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(standard_layout(fig_profit, 360), use_container_width=True)

    st.caption(
        "Main chart shows gross profit if the tender is won. Green points satisfy both the minimum GMR and minimum win-probability rules; gray points do not. Expected gross profit remains available in the grid and hover details."
    )

    with st.expander("Show model-estimated win probability curve"):
        st.caption(
            "The win-probability curve is a model-estimated historical-pattern signal. It is not constrained to decrease smoothly as GMR increases, so local jumps can appear in sparse or non-linear regions."
        )
        fig_prob = go.Figure()
        fig_prob.add_trace(
            go.Scatter(
                x=line_grid["gmr_pct"],
                y=line_grid["predicted_win_probability"],
                name="Estimated win probability",
                mode="lines+markers",
                line=dict(color=BLUE, width=3),
                marker=dict(size=6, color=np.where(line_grid["eligible_for_recommendation"], GREEN, BLUE)),
                customdata=np.stack(
                    [
                        line_grid["expected_gross_profit"],
                        line_grid["meets_both_rules_label"],
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "GMR: %{x:.0f}%<br>"
                    "Win probability: %{y:.1%}<br>"
                    "Expected gross profit: %{customdata[0]:,.0f}<br>"
                    "Rule status: %{customdata[1]}<extra></extra>"
                ),
            )
        )
        fig_prob.add_hline(
            y=min_win_prob,
            line_dash="dot",
            line_color=AMBER,
            annotation_text="Minimum win probability",
        )
        fig_prob.add_vline(x=input_gmr * 100, line_dash="dash", line_color=BLUE, annotation_text="Input GMR")
        fig_prob.add_vline(x=rec_gmr_pct, line_dash="dot", line_color=GREEN, annotation_text="Recommended GMR")
        fig_prob.update_layout(
            xaxis_title="Gross Margin Rate (%)",
            yaxis=dict(title="Estimated Win Probability", tickformat=".0%", range=[0, 1.05]),
        )
        st.plotly_chart(standard_layout(fig_prob, 320), use_container_width=True)

# =============================================================================
# SHARED RESOURCES
# =============================================================================
try:
    bundle = load_model_bundle()
    MODEL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES = get_model_feature_lists(bundle)
    model = bundle["model"]
    data = load_data()
    model_metrics, confusion = load_hgb_metrics()
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
            "Model Notes",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**Decision-support scope**")
    st.caption("This dashboard supports price review. It does not replace managerial approval, customer knowledge, or commercial judgement.")
    st.markdown(
        f"""
        <div class='sidebar-badge'>
            <b>Version:</b> {APP_VERSION}<br>
            <b>Last updated:</b> {LAST_UPDATED}<br>
            <b>Model:</b> {bundle.get('model_name', 'Saved model')}<br>
            <b>Config:</b> {bundle.get('model_config', 'unknown')}
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
            "A high-level summary of historical quotation performance, product conversion patterns, and general lost-tender review signals.",
        )
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            render_metric_card("Historical quotations", f"{quote_header['quote_id'].nunique():,}", "Unique quotation records", BLUE, "🧾")
        with col2:
            render_metric_card("Product groups", f"{data['product_model'].nunique():,}", "Product model groups including Other", TEAL, "📦")
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
            render_section_header("Product-level win rate", "Product groups with at least 20 quotation observations.")
            product_summary = (
                data.drop_duplicates(subset=["quote_id", "product_model"])
                .groupby("product_model")
                .agg(Observations=("quote_id", "nunique"), Win_Rate=("is_win", "mean"))
                .reset_index()
            )
            product_summary = product_summary[product_summary["Observations"] >= 20].sort_values("Win_Rate")
            fig = px.bar(
                product_summary,
                x="Win_Rate",
                y="product_model",
                orientation="h",
                text=product_summary["Win_Rate"].map(lambda value: f"{value:.1%}"),
                color="Win_Rate",
                color_continuous_scale=["#CCFBF1", "#14B8A6", "#0F766E"],
                hover_data={"Observations": True, "Win_Rate": ":.1%"},
                labels={"product_model": "Product", "Win_Rate": "Historical Win Rate"},
            )
            fig.update_layout(coloraxis_showscale=False, xaxis_tickformat=".0%")
            st.plotly_chart(standard_layout(fig, 360), use_container_width=True)

    with st.container(border=True):
        render_section_header("General lost-tender review signals", "Rule-based descriptive signals for failed product lines.")
        product_median = data.groupby("product")["gross_margin_rate"].median().to_dict()
        failed = data[data["is_win"] == 0].copy()
        failed["Potential review signal"] = failed.apply(lambda row: potential_loss_signal(row, product_median), axis=1)
        signal_summary = failed["Potential review signal"].value_counts().rename_axis("Potential review signal").reset_index(name="Failed product lines")
        fig = px.bar(
            signal_summary,
            x="Potential review signal",
            y="Failed product lines",
            text_auto=True,
            color="Potential review signal",
            color_discrete_map={
                "Above competitor benchmark": RED,
                "General review signal": SLATE,
                "Competitor information unavailable": PURPLE,
                "High margin for product": AMBER,
            },
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(standard_layout(fig, 370), use_container_width=True)

# =============================================================================
# PAGE 2 — QUOTE RECOMMENDER
# =============================================================================
elif page == "Quote Recommender":
    render_title()

    # ---------------------------------------------------------------------
    # Session-state setup for quote-level product-line forms
    # ---------------------------------------------------------------------
    def _new_state_line(product: Optional[str] = None) -> Dict:
        row = make_default_quote_row(data, product)
        if "quote_line_next_id" not in st.session_state:
            st.session_state.quote_line_next_id = 1
        row["_line_id"] = int(st.session_state.quote_line_next_id)
        st.session_state.quote_line_next_id += 1
        return row

    def _normalise_quote_line_state() -> None:
        """Keep quote line state as a list of dictionaries with stable line IDs."""
        if "quote_lines" not in st.session_state:
            st.session_state.quote_lines = [_new_state_line("H1" if "H1" in product_model_options(data) else None)]
            return

        existing = st.session_state.quote_lines
        if isinstance(existing, pd.DataFrame):
            lines = existing.to_dict("records")
        elif isinstance(existing, list):
            lines = existing
        else:
            lines = []

        if not lines:
            lines = [_new_state_line("H1" if "H1" in product_model_options(data) else None)]

        max_id = 0
        clean_lines = []
        for idx, line in enumerate(lines, start=1):
            line = dict(line)
            if "_line_id" not in line or pd.isna(line.get("_line_id")):
                line["_line_id"] = idx
            line["_line_id"] = int(line["_line_id"])
            max_id = max(max_id, line["_line_id"])
            clean_lines.append(line)
        st.session_state.quote_lines = clean_lines
        st.session_state.quote_line_next_id = max(int(st.session_state.get("quote_line_next_id", 1)), max_id + 1)

    def _set_numeric_default(line_id: int, field_key: str, value) -> None:
        key = f"line_{line_id}_{field_key}"
        if key not in st.session_state:
            st.session_state[key] = value

    def _reset_line_widget_values(line_id: int, defaults: Dict) -> None:
        mapping = {
            "kw": "kW",
            "qty": "Qty",
            "cost": "Cost per Unit",
            "current_gmr": "Input GMR (%)",
            "grant": "Energy Grant",
            "comp_a": "Competitor A",
            "comp_b": "Competitor B",
            "comp_c": "Competitor C",
        }
        for widget_suffix, col in mapping.items():
            st.session_state[f"line_{line_id}_{widget_suffix}"] = defaults[col]

    _normalise_quote_line_state()
    if "quote_result" not in st.session_state:
        st.session_state.quote_result = None

    with st.container(border=True):
        render_section_header(
            "Quote-Level Recommender",
            "Add one or more product lines, then click Check Quotation. The quote-level win probability is calculated as the arithmetic mean of product-line probabilities.",
        )

        st.markdown("<span class='pill pill-purple'>Quote-level recommendation rules</span>", unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            gmr_min_pct = st.number_input("GMR grid min (%)", value=0.0, step=1.0, format="%.2f")
        with c2:
            gmr_max_pct = st.number_input("GMR grid max (%)", value=60.0, step=1.0, format="%.2f")
        with c3:
            gmr_interval_pct = st.number_input("GMR increment (%)", min_value=0.1, value=1.0, step=0.5, format="%.2f")
        with c4:
            minimum_margin_pct = st.number_input(
                "Minimum allowed GMR (%)",
                value=0.0,
                step=1.0,
                format="%.2f",
                help="Quote-level rule applied to all product-line GMR candidates before selecting the model-based recommendation.",
            )
        with c5:
            minimum_win_probability_pct = st.number_input(
                "Minimum win probability (%)",
                value=80.0,
                min_value=0.0,
                max_value=100.0,
                step=5.0,
                format="%.2f",
                help="Quote-level rule applied to product-line candidates. The displayed quote-level probability is the mean across product lines.",
            )

        render_insight(
            "The model evaluates each product line across the selected GMR grid, filters candidates using the quote-level rules above, and selects the candidate with the highest expected gross profit."
        )

        st.markdown("<span class='pill pill-blue'>Product-line input forms</span>", unsafe_allow_html=True)
        product_options = product_model_options(data)
        new_lines: List[Dict] = []
        remove_line_id: Optional[int] = None

        for display_idx, line in enumerate(st.session_state.quote_lines, start=1):
            line = dict(line)
            line_id = int(line.get("_line_id", display_idx))
            if str(line.get("Product", "")).strip() == "":
                line["Product"] = product_options[0]

            product_key = f"line_{line_id}_product"
            if product_key not in st.session_state:
                st.session_state[product_key] = line.get("Product", product_options[0])

            # If the product was changed, refresh product-related default values.
            selected_product_state = str(st.session_state.get(product_key, line.get("Product", product_options[0])))
            if selected_product_state != str(line.get("Product", "")):
                refreshed = make_default_quote_row(data, selected_product_state)
                refreshed["_line_id"] = line_id
                line = refreshed
                _reset_line_widget_values(line_id, refreshed)

            with st.container(border=True):
                header_left, header_right = st.columns([5, 1])
                with header_left:
                    st.markdown(f"**Product line {display_idx}**")
                with header_right:
                    if len(st.session_state.quote_lines) > 1:
                        if st.button("Remove", key=f"remove_line_{line_id}"):
                            remove_line_id = line_id

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    current_product = str(line.get("Product", product_options[0]))
                    if current_product not in product_options:
                        current_product = "Other" if "Other" in product_options else product_options[0]
                    product = st.selectbox(
                        "Product",
                        product_options,
                        index=product_options.index(current_product),
                        key=product_key,
                    )
                    _set_numeric_default(line_id, "kw", float(line.get("kW", 1.0)))
                    kw = st.number_input("Power rating (kW)", min_value=0.0, step=0.5, format="%.3f", key=f"line_{line_id}_kw")
                    _set_numeric_default(line_id, "qty", int(max(1, safe_float(line.get("Qty", 1), 1))))
                    qty = st.number_input("Quantity", min_value=1, step=1, format="%d", key=f"line_{line_id}_qty")

                with col_b:
                    _set_numeric_default(line_id, "cost", float(line.get("Cost per Unit", 1.0)))
                    cost_per_unit = st.number_input("Estimated cost per unit", min_value=0.0, step=1000.0, format="%.2f", key=f"line_{line_id}_cost")
                    _set_numeric_default(line_id, "current_gmr", float(line.get("Input GMR (%)", 30.0)))
                    current_gmr = st.number_input(
                        "Current / proposed GMR (%)",
                        step=0.5,
                        format="%.2f",
                        key=f"line_{line_id}_current_gmr",
                        help="Used as a reference point in the what-if chart. The recommendation itself is selected from the GMR grid.",
                    )
                    _set_numeric_default(line_id, "grant", float(line.get("Energy Grant", 0.0)))
                    energy_grant = st.number_input("Energy grant amount", min_value=0.0, step=1000.0, format="%.2f", key=f"line_{line_id}_grant")

                with col_c:
                    _set_numeric_default(line_id, "comp_a", float(line.get("Competitor A", 0.0)))
                    competitor_a = st.number_input("Competitor A price", min_value=0.0, step=1000.0, format="%.2f", key=f"line_{line_id}_comp_a")
                    _set_numeric_default(line_id, "comp_b", float(line.get("Competitor B", 0.0)))
                    competitor_b = st.number_input("Competitor B price", min_value=0.0, step=1000.0, format="%.2f", key=f"line_{line_id}_comp_b")
                    _set_numeric_default(line_id, "comp_c", float(line.get("Competitor C", 0.0)))
                    competitor_c = st.number_input("Competitor C price", min_value=0.0, step=1000.0, format="%.2f", key=f"line_{line_id}_comp_c")

                new_lines.append(
                    {
                        "_line_id": line_id,
                        "Product": product,
                        "kW": float(kw),
                        "Qty": int(qty),
                        "Cost per Unit": float(cost_per_unit),
                        "Input GMR (%)": float(current_gmr),
                        "Energy Grant": float(energy_grant),
                        "Competitor A": float(competitor_a),
                        "Competitor B": float(competitor_b),
                        "Competitor C": float(competitor_c),
                    }
                )

        if remove_line_id is not None:
            st.session_state.quote_lines = [line for line in st.session_state.quote_lines if int(line.get("_line_id")) != remove_line_id]
            st.session_state.quote_result = None
            st.rerun()

        st.session_state.quote_lines = new_lines
        quote_input_df = pd.DataFrame([{k: v for k, v in line.items() if k != "_line_id"} for line in new_lines])

        action_left, action_middle, action_right = st.columns([1, 1, 4])
        with action_left:
            if st.button("➕ Add product", use_container_width=True):
                st.session_state.quote_lines.append(_new_state_line())
                st.session_state.quote_result = None
                st.rerun()
        with action_middle:
            check_clicked = st.button("🔍 Check Quotation", type="primary", use_container_width=True)

        if check_clicked:
            try:
                result = evaluate_quote(
                    quote_rows=quote_input_df,
                    bundle=bundle,
                    gmr_min=gmr_min_pct / 100.0,
                    gmr_max=gmr_max_pct / 100.0,
                    gmr_interval=gmr_interval_pct / 100.0,
                    minimum_margin=minimum_margin_pct / 100.0,
                    minimum_win_probability=minimum_win_probability_pct / 100.0,
                )
                st.session_state.quote_result = result
            except ValueError as exc:
                st.session_state.quote_result = None
                st.error(str(exc))

    result = st.session_state.quote_result
    if result is None:
        st.info("Enter quote information and click **Check Quotation** to generate model-based results.")
        st.stop()

    summary = result["quote_summary"]
    rec = result["recommendations"].copy()

    with st.container(border=True):
        render_section_header("Quote-Level Result", "Aggregated model-based recommendation across all product lines in the quote.")
        q1, q2, q3, q4, q5 = st.columns(5)
        with q1:
            render_metric_card("Product lines", f"{summary['line_count']:,}", "Number of checked product lines", BLUE, "🧩", compact=True)
        with q2:
            render_metric_card("Quote win probability", pct(summary["quote_win_probability"]), "Mean of product-line probabilities", GREEN, "🏆", compact=True)
        with q3:
            render_metric_card("Weighted recommended GMR", pct(summary["quote_weighted_gmr"]), "Weighted by recommended subtotal", TEAL, "📐", compact=True)
        with q4:
            render_metric_card("Expected gross profit", currency(summary["quote_total_expected_gp"]), "Win probability × gross profit if won", PURPLE, "⚖️", compact=True)
        with q5:
            render_metric_card("Gross profit if won", currency(summary["quote_total_gp_if_won"]), "Profit if the quote is won", AMBER, "📈", compact=True)

        if result["notes"]:
            for note in result["notes"]:
                st.warning(note)
        else:
            render_insight("All product-line recommendations satisfy the selected minimum GMR and minimum win-probability rules.")

    with st.container(border=True):
        render_section_header("Product-Level Recommendations", "Recommended GMR, price, probability, and gross profit for each product line.")
        display = rec.copy()
        display_table = pd.DataFrame(
            {
                "Line": display["line_no"].astype(int),
                "Product": display["product_model"].astype(str),
                "Qty": display["qty"].astype(int),
                "Recommended GMR": display["gross_margin_rate"].map(lambda x: f"{x:.0%}"),
                "Recommended Unit Price": display["unit_price"].map(currency),
                "Recommended Subtotal": display["subtotal_price"].map(currency),
                "Win Probability": display["predicted_win_probability"].map(lambda x: f"{x:.1%}"),
                "Gross Profit if Won": display["candidate_estimated_gross_profit"].map(currency),
                "Expected Gross Profit": display["expected_gross_profit"].map(currency),
                "Gap vs Min Competitor": display["price_gap_min_competitor_pct"].map(lambda x: "N/A" if pd.isna(x) else f"{x:.1f}%"),
            }
        )
        st.dataframe(display_table, use_container_width=True, hide_index=True)

        render_insight(
            "Gross profit if won is the profit conditional on winning the tender. Expected gross profit is probability-adjusted: win probability multiplied by gross profit if won."
        )

    simulation_col, benchmark_col = st.columns([1.55, 1.0])
    with simulation_col:
        with st.container(border=True):
            render_section_header("What-if Margin Simulation", "Explore how GMR changes gross profit if won. Expected gross profit remains available in the grid and hover details. The default increment is 1%.")
            line_options = [int(x) for x in rec["line_no"].tolist()]
            selected_line = st.selectbox("Select product line", line_options, format_func=lambda x: f"Line {x} — {rec.loc[rec['line_no'] == x, 'product_model'].iloc[0]}")
            render_what_if_chart(result, selected_line)

            st.markdown("**Scenario grid for selected product line**")
            selected_grid = result["scenarios"][result["scenarios"]["line_no"] == selected_line].copy().sort_values("gross_margin_rate")
            selected_grid_display = pd.DataFrame(
                {
                    "GMR": selected_grid["gross_margin_rate"].map(lambda x: f"{x:.0%}"),
                    "Unit Price": selected_grid["unit_price"].map(currency),
                    "Win Probability": selected_grid["predicted_win_probability"].map(lambda x: f"{x:.1%}"),
                    "Gross Profit if Won": selected_grid["candidate_estimated_gross_profit"].map(currency),
                    "Expected Gross Profit": selected_grid["expected_gross_profit"].map(currency),
                    "Meets Min GMR": selected_grid["meets_min_gmr"].map(lambda x: "Yes" if bool(x) else "No"),
                    "Meets Min Win Prob": selected_grid["meets_min_win_probability"].map(lambda x: "Yes" if bool(x) else "No"),
                    "Selected": selected_grid["selected_candidate"].map(lambda x: "✅" if bool(x) else ""),
                }
            )
            st.dataframe(selected_grid_display, use_container_width=True, hide_index=True)

    with benchmark_col:
        with st.container(border=True):
            render_section_header("Historical Benchmark", "Product-line benchmark for the selected product and recommended GMR band.")
            selected_rec = rec[rec["line_no"] == selected_line].iloc[0]
            benchmark = historical_benchmark(data, str(selected_rec["product_model"]), float(selected_rec["gross_margin_rate"]))
            price_position = classify_price_position(float(selected_rec["unit_price"]), float(selected_rec["min_competitor_price"]) if pd.notna(selected_rec["min_competitor_price"]) else np.nan)
            b1, b2 = st.columns(2)
            with b1:
                render_metric_card("Selected product", str(selected_rec["product_model"]), "Selected product line", BLUE, "📦", compact=True)
            with b2:
                render_metric_card("GMR band", benchmark["band"], "Recommended GMR interval", TEAL, "📐", compact=True)
            b3, b4 = st.columns(2)
            with b3:
                render_metric_card("Records in band", f"{benchmark['n']:,}", "Product-line records in this GMR band", AMBER, "🧾", compact=True)
            with b4:
                benchmark_win_rate = "N/A" if pd.isna(benchmark["historical_win_rate"]) else f"{benchmark['historical_win_rate']:.1%}"
                render_metric_card("Win rate in band", benchmark_win_rate, "Product-level historical benchmark", PURPLE, "🏅", compact=True)
            render_insight(
                f"<b>Total product-line records for {selected_rec['product_model']}:</b> {benchmark['product_total_records']:,}. "
                f"<b>Price position:</b> {price_position}"
            )

    flags_col, competitor_col = st.columns([1.0, 1.2])
    with flags_col:
        with st.container(border=True):
            render_section_header("Quote Review Flags", "Rule-based warnings that may require sales or supervisor review.")
            for severity, title, explanation in quote_review_flags(result):
                render_risk_card(severity, title, explanation)

    with competitor_col:
        with st.container(border=True):
            render_section_header("Competitor Positioning", "Recommended unit prices compared with available minimum competitor benchmarks.")
            comp_df = rec[["line_no", "product_model", "unit_price", "min_competitor_price"]].copy()
            comp_df["line_label"] = "Line " + comp_df["line_no"].astype(int).astype(str) + " — " + comp_df["product_model"].astype(str)
            long_parts = [
                comp_df[["line_label", "unit_price"]].rename(columns={"unit_price": "Price"}).assign(Reference="Recommended unit price"),
                comp_df[["line_label", "min_competitor_price"]].rename(columns={"min_competitor_price": "Price"}).assign(Reference="Minimum competitor"),
            ]
            chart_data = pd.concat(long_parts, ignore_index=True).dropna(subset=["Price"])
            if chart_data.empty:
                st.warning("No competitor benchmark is available for this quote.")
            else:
                fig = px.bar(
                    chart_data,
                    x="line_label",
                    y="Price",
                    color="Reference",
                    barmode="group",
                    text_auto=",.0f",
                    color_discrete_map={"Recommended unit price": BLUE, "Minimum competitor": AMBER},
                )
                fig.update_layout(xaxis_title="Product line", yaxis_title="Unit price")
                st.plotly_chart(standard_layout(fig, 330), use_container_width=True)

    with st.expander("View full scenario grid"):
        scenario_display = result["scenarios"][[
            "line_no", "product_model", "gross_margin_rate", "unit_price", "predicted_win_probability",
            "candidate_estimated_gross_profit", "expected_gross_profit", "meets_all_rules", "selected_candidate",
        ]].copy()
        scenario_display["GMR"] = scenario_display["gross_margin_rate"].map(lambda x: f"{x:.0%}")
        scenario_display["Unit Price"] = scenario_display["unit_price"].map(currency)
        scenario_display["Win Probability"] = scenario_display["predicted_win_probability"].map(lambda x: f"{x:.1%}")
        scenario_display["Gross Profit if Won"] = scenario_display["candidate_estimated_gross_profit"].map(currency)
        scenario_display["Expected Gross Profit"] = scenario_display["expected_gross_profit"].map(currency)
        scenario_display["Meets Rules"] = scenario_display["meets_all_rules"].map(lambda x: "Yes" if bool(x) else "No")
        scenario_display["Selected"] = scenario_display["selected_candidate"].map(lambda x: "✅" if bool(x) else "")
        scenario_display = scenario_display.rename(columns={"line_no": "Line", "product_model": "Product"})
        st.dataframe(
            scenario_display[["Line", "Product", "GMR", "Unit Price", "Win Probability", "Gross Profit if Won", "Expected Gross Profit", "Meets Rules", "Selected"]],
            use_container_width=True,
            hide_index=True,
        )

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
        options = ["All products"] + product_model_options(data)
        selected_product = st.selectbox("Filter by product group", options)

        if selected_product == "All products":
            subset = data.drop_duplicates(subset=["quote_id"]).copy()
        else:
            subset = data[data["product_model"].astype(str) == selected_product].drop_duplicates(subset=["quote_id"]).copy()

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
        if not summary_for_metrics.empty:
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
        display_summary["Average_Gross_Profit"] = display_summary["Average_Gross_Profit"].map(currency)
        display_summary["Expected_GP_Proxy"] = display_summary["Expected_GP_Proxy"].map(currency)
        st.dataframe(display_summary, use_container_width=True, hide_index=True)
        render_insight("The expected-GP proxy is descriptive. For quote review, use the Quote Recommender page, which simulates GMR scenarios while holding estimated cost constant.")

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
        render_insight("Competitor analysis is retrospective and should be used as a descriptive price-review benchmark.")

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
                text=positioning_summary["Historical_Win_Rate"].map(lambda value: f"{value:.1%}" if pd.notna(value) else "N/A"),
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
            availability = data["competitor_count_available"].value_counts().sort_index().rename_axis("Available competitor prices").reset_index(name="Product lines")
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
            competitor_data.groupby(["product_model", "price_position"])
            .agg(Observations=("quote_id", "size"), Win_Rate=("is_win", "mean"))
            .reset_index()
        )
        product_position = product_position[product_position["Observations"] >= 10]
        fig = px.bar(
            product_position,
            x="product_model",
            y="Win_Rate",
            color="price_position",
            barmode="group",
            hover_data={"Observations": True, "Win_Rate": ":.1%"},
            color_discrete_map={
                "Below competitor benchmark": GREEN,
                "Approximately aligned (±5%)": AMBER,
                "Above competitor benchmark": RED,
            },
            labels={"product_model": "Product", "Win_Rate": "Historical Win Rate", "price_position": "Price position"},
        )
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(standard_layout(fig, 390), use_container_width=True)

# =============================================================================
# PAGE 5 — MODEL NOTES
# =============================================================================
else:
    render_title()
    quote_counts = data.groupby("quote_id").size()

    with st.container(border=True):
        render_section_header(
            "Model Notes & Data Quality",
            "A structured overview of cleaned data, model validation results, interpretation boundaries, and data-improvement priorities.",
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
            render_section_header("Model validation summary", "Stratified 5-fold cross-validation for the final HistGradientBoosting win-probability model.")
            metric_items = list(model_metrics.items())
            row1 = st.columns(4)
            row2 = st.columns(4)
            accents = [BLUE, PURPLE, TEAL, AMBER, GREEN, ORANGE, SKY, SLATE]
            icons = ["📈", "🎯", "🔍", "⚖️", "✅", "📊", "📉", "🧮"]
            for index, ((name, value), accent, icon) in enumerate(zip(metric_items, accents, icons)):
                target_column = row1[index] if index < 4 else row2[index - 4]
                with target_column:
                    render_metric_card(name, f"{value:.3f}", "5-fold CV metric", accent, icon, compact=True)

    lower_left, lower_right = st.columns([1.0, 1.0])
    with lower_left:
        with st.container(border=True):
            render_section_header("Confusion matrix", "Aggregated 5-fold CV classification outcomes using a 0.50 decision threshold.")
            confusion_df = pd.DataFrame(confusion, index=["Actual Fail", "Actual Success"], columns=["Predicted Fail", "Predicted Success"])
            text = confusion_df.copy().astype(str)
            actual_fail_total = confusion_df.loc["Actual Fail"].sum()
            actual_success_total = confusion_df.loc["Actual Success"].sum()
            if actual_fail_total:
                text.loc["Actual Fail"] = [f"{v}<br>{v / actual_fail_total:.2%}" for v in confusion_df.loc["Actual Fail"]]
            if actual_success_total:
                text.loc["Actual Success"] = [f"{v}<br>{v / actual_success_total:.2%}" for v in confusion_df.loc["Actual Success"]]
            fig = go.Figure(
                data=go.Heatmap(
                    z=confusion_df.values,
                    x=confusion_df.columns,
                    y=confusion_df.index,
                    text=text.values,
                    texttemplate="%{text}",
                    textfont={"size": 17},
                    colorscale="Blues",
                    showscale=False,
                )
            )
            st.plotly_chart(standard_layout(fig, 310), use_container_width=True)
            st.caption("Percentages are calculated within each actual class.")

    with lower_right:
        with st.container(border=True):
            render_section_header("Model safeguards", "Safeguards applied before model training and dashboard use.")
            st.markdown(
                """
                <div class='soft-box'>
                    <ul>
                        <li>The final website loads the saved <b>HistGradientBoosting</b> model bundle instead of training inside the app.</li>
                        <li>The model was evaluated using <b>stratified 5-fold cross-validation</b>.</li>
                        <li>Quantity values less than or equal to zero are excluded as system errors.</li>
                        <li>Negative gross margins are retained as valid strategic business cases.</li>
                        <li>Missing competitor values are not replaced with zero.</li>
                        <li>For repeated Quote ID + Product records, the highest unit price is retained only when duplicate price variations exist.</li>
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
                        <li>The dashboard estimates <b>historical associations</b> between quotation features and win outcomes.</li>
                        <li>The primary use case is <b>during price review</b>, before quotation submission.</li>
                        <li>The recommender works at the <b>product-line price-review level</b> and aggregates results to the quote level.</li>
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

    if SHAP_IMAGE_PATH.exists() or SHAP_IMPORTANCE_PATH.exists():
        with st.container(border=True):
            render_section_header("Model explanation", "Global feature importance from SHAP output, if available in the asset folder.")
            if SHAP_IMAGE_PATH.exists():
                st.image(str(SHAP_IMAGE_PATH), use_container_width=True)
            elif SHAP_IMPORTANCE_PATH.exists():
                shap_df = pd.read_csv(SHAP_IMPORTANCE_PATH).head(15)
                fig = px.bar(shap_df.sort_values("mean_abs_shap"), x="mean_abs_shap", y="feature", orientation="h")
                st.plotly_chart(standard_layout(fig, 420), use_container_width=True)

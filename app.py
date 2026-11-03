from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent
DATASET_PATH = ROOT_DIR / "datasets" / "ecommerce_return_prediction_feature_engineered.csv"
MODEL_PATH = ROOT_DIR / "models" / "best_return_detection_model.joblib"
EVALUATION_PATH = ROOT_DIR / "models" / "model_evaluation.json"
INSIGHTS_PATH = ROOT_DIR / "models" / "dashboard_insights.json"

RAW_FEATURES = [
    "customer_age",
    "country",
    "device_type",
    "product_category",
    "product_price_usd",
    "discount_percent",
    "coupon_used",
    "product_rating",
    "past_purchase_count",
    "past_return_rate_pct",
    "session_length_minutes",
    "num_product_views",
    "shipping_method",
    "delivery_time_days",
    "delivery_delay_days",
    "payment_method",
    "total_orders_lifetime",
    "total_returns_lifetime",
]

NUMERIC_FEATURES = [
    "customer_age",
    "product_price_usd",
    "discount_percent",
    "product_rating",
    "past_purchase_count",
    "past_return_rate_pct",
    "session_length_minutes",
    "num_product_views",
    "delivery_time_days",
    "delivery_delay_days",
    "total_orders_lifetime",
    "total_returns_lifetime",
    "order_month",
    "order_day_of_week",
    "is_weekend",
]

CATEGORY_OPTIONS = {
    "country": ["AU", "CA", "DE", "FR", "GB", "IN", "US"],
    "device_type": ["Desktop", "Mobile", "Tablet"],
    "product_category": [
        "Beauty",
        "Books",
        "Clothing",
        "Electronics",
        "Home & Kitchen",
        "Shoes",
        "Sports",
        "Toys",
    ],
    "coupon_used": ["No", "Yes"],
    "shipping_method": ["Express", "Same Day", "Standard"],
    "payment_method": [
        "Buy Now Pay Later",
        "Credit Card",
        "Debit Card",
        "Digital Wallet",
        "PayPal",
    ],
}

st.set_page_config(
    page_title="Returns Desk | Order Analytics",
    page_icon="↩",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #252b27;
        --muted: #74766f;
        --accent: #bd583d;
        --accent-dark: #9e422d;
        --paper: #f6f4ef;
        --line: #e6e1d8;
        --panel: #fffefa;
    }

    html, body, [class*="css"] {
        font-family: 'Aptos', 'Segoe UI', sans-serif;
        color: var(--ink);
    }

    .stApp { background: var(--paper); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stMainBlockContainer"] {
        max-width: 1440px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    [data-testid="stSidebar"] {
        background: #272d29;
        border-right: 1px solid #343a35;
    }
    [data-testid="stSidebar"] * { color: #f2efe8; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #c2c2b9;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        padding: 0.5rem 0.65rem;
        border-radius: 7px;
        transition: background 140ms ease;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: #393f3a;
    }

    .brand-mark {
        display: inline-flex;
        width: 36px;
        height: 36px;
        align-items: center;
        justify-content: center;
        border: 1px solid #a65840;
        border-radius: 50%;
        background: #bd583d;
        color: #fff9f0;
        font-family: Georgia, serif;
        font-weight: 700;
        font-size: 19px;
        margin-right: 9px;
        vertical-align: middle;
    }
    .brand-name {
        display: inline-block;
        color: #fffefa;
        font-family: Georgia, 'Times New Roman', serif;
        font-size: 18px;
        font-weight: 700;
        vertical-align: middle;
    }
    .sidebar-caption {
        color: #b7b9b0;
        font-size: 12px;
        line-height: 1.6;
        margin: 12px 0 24px;
    }
    .eyebrow {
        color: var(--accent);
        font-size: 11px;
        font-weight: 750;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 9px;
    }
    .page-title {
        color: var(--ink);
        font-family: Georgia, 'Times New Roman', serif;
        font-size: clamp(32px, 3vw, 43px);
        font-weight: 500;
        letter-spacing: -0.035em;
        line-height: 1.15;
        margin: 0 0 8px;
    }
    .page-subtitle {
        color: var(--muted);
        font-size: 14px;
        line-height: 1.65;
        margin: 0 0 16px;
    }
    .section-heading {
        color: var(--ink);
        font-family: Georgia, 'Times New Roman', serif;
        font-size: 21px;
        font-weight: 700;
        margin: 18px 0 8px;
    }
    .hero-card {
        display: grid;
        grid-template-columns: minmax(0, 1.3fr) minmax(230px, 0.7fr);
        gap: 30px;
        align-items: stretch;
        background: #292f2b;
        border: 1px solid #363c37;
        border-radius: 13px;
        color: #fffefa;
        padding: 30px 34px;
        margin: 8px 0 20px;
        box-shadow: 0 12px 30px rgba(37, 43, 39, 0.08);
    }
    .hero-kicker {
        color: #d8957e;
        font-size: 10px;
        font-weight: 750;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin: 0 0 16px;
    }
    .hero-stat {
        color: #fffefa;
        font-family: Georgia, 'Times New Roman', serif;
        font-size: clamp(46px, 6vw, 72px);
        letter-spacing: -0.055em;
        line-height: 1;
        margin: 0;
    }
    .hero-caption {
        color: #d0d1c9;
        font-size: 13px;
        line-height: 1.65;
        margin: 13px 0 0;
        max-width: 540px;
    }
    .hero-aside {
        display: flex;
        flex-direction: column;
        justify-content: center;
        border-left: 1px solid #4a504a;
        padding-left: 28px;
    }
    .hero-aside-label {
        color: #c4c5bc;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    .hero-aside strong {
        color: #fffefa;
        font-family: Georgia, 'Times New Roman', serif;
        font-size: 20px;
        font-weight: 500;
        line-height: 1.35;
    }
    .hero-aside span {
        color: #c4c5bc;
        font-size: 12px;
        line-height: 1.6;
        margin-top: 8px;
    }
    .note-card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-left: 3px solid var(--accent);
        border-radius: 8px;
        color: #55574f;
        font-size: 13px;
        line-height: 1.7;
        padding: 15px 18px;
        margin: 14px 0 22px;
    }
    .result-card {
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto;
        align-items: center;
        gap: 20px;
        background: var(--panel);
        border: 1px solid var(--line);
        border-top: 3px solid var(--accent);
        border-radius: 9px;
        padding: 21px 24px;
        margin: 20px 0 12px;
    }
    .result-card.result-low { border-top-color: #63836f; }
    .result-card.result-review { border-top-color: #bd7d3d; }
    .result-card.result-high { border-top-color: #bd583d; }
    .result-label {
        color: var(--muted);
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }
    .result-score {
        color: var(--ink);
        font-family: Georgia, 'Times New Roman', serif;
        font-size: 38px;
        letter-spacing: -0.04em;
        line-height: 1.15;
        margin-top: 5px;
    }
    .result-meter {
        width: min(390px, 42vw);
        height: 7px;
        background: #e9e5dd;
        border-radius: 20px;
        margin-top: 14px;
        overflow: hidden;
    }
    .result-meter span {
        display: block;
        height: 100%;
        background: var(--accent);
        border-radius: inherit;
    }
    .result-low .result-meter span { background: #63836f; }
    .result-review .result-meter span { background: #bd7d3d; }
    .result-risk {
        color: var(--ink);
        font-family: Georgia, 'Times New Roman', serif;
        font-size: 20px;
        text-align: right;
    }
    .result-risk small {
        display: block;
        color: var(--muted);
        font-family: 'Aptos', 'Segoe UI', sans-serif;
        font-size: 11px;
        margin-top: 5px;
    }
    div[data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 9px;
        padding: 14px 13px;
        min-height: 110px;
        box-shadow: 0 3px 12px rgba(37, 43, 39, 0.025);
    }
    div[data-testid="stMetricLabel"] {
        color: var(--muted);
        font-size: 12px;
    }
    div[data-testid="stMetricValue"] {
        color: var(--ink);
        font-family: Georgia, 'Times New Roman', serif;
        font-size: 24px;
    }
    div[data-testid="stMetricDelta"] {
        color: var(--muted);
        font-size: 11px;
    }
    .stButton > button, .stDownloadButton > button {
        border-radius: 7px;
        font-weight: 650;
        min-height: 42px;
    }
    .stButton > button[kind="primary"] {
        background: var(--accent);
        border-color: var(--accent);
    }
    .stButton > button[kind="primary"]:hover {
        background: var(--accent-dark);
        border-color: var(--accent-dark);
    }
    div[data-testid="stForm"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 22px;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 8px;
        overflow: hidden;
    }
    div[data-testid="stFileUploader"] section {
        background: var(--panel);
        border: 1px dashed #c9c1b6;
        border-radius: 9px;
    }
    div[data-testid="stExpander"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
    }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #b7b9b0;
    }
    .small-muted {
        color: var(--muted);
        font-size: 12px;
    }

    @media (max-width: 760px) {
        [data-testid="stMainBlockContainer"] {
            padding: 1.25rem 1rem 2rem;
        }
        .hero-card {
            grid-template-columns: 1fr;
            gap: 22px;
            padding: 24px;
        }
        .hero-aside {
            border-left: 0;
            border-top: 1px solid #4a504a;
            padding: 18px 0 0;
        }
        .result-card {
            grid-template-columns: 1fr;
        }
        .result-meter {
            width: 100%;
        }
        .result-risk {
            text-align: left;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_detector(
    artifact_modified_ns: int,
    artifact_size_bytes: int,
) -> tuple[Any, float, str]:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained model was not found: {MODEL_PATH}")
    artifact_stat = MODEL_PATH.stat()
    if (
        artifact_stat.st_mtime_ns != artifact_modified_ns
        or artifact_stat.st_size != artifact_size_bytes
    ):
        raise ValueError("The model artifact changed while the dashboard was loading. Rerun the app.")
    artifact = joblib.load(MODEL_PATH)
    if not isinstance(artifact, dict) or "model" not in artifact or "threshold" not in artifact:
        raise ValueError("The saved return-detection artifact is missing its model or threshold.")
    model = artifact["model"]
    threshold = float(artifact["threshold"])
    model_name = str(artifact.get("model_name", type(model).__name__))
    if not hasattr(model, "predict_proba"):
        raise ValueError("The saved return-detection model does not support probability predictions.")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("The saved decision threshold must be between 0 and 1.")
    if not hasattr(model, "feature_names_in_"):
        raise ValueError("The saved model does not expose its training feature names.")
    return model, threshold, model_name


@st.cache_data
def load_evaluation(
    artifact_modified_ns: int,
    artifact_size_bytes: int,
) -> dict[str, Any]:
    if not EVALUATION_PATH.is_file():
        raise FileNotFoundError(f"Model evaluation results were not found: {EVALUATION_PATH}")
    artifact_stat = EVALUATION_PATH.stat()
    if (
        artifact_stat.st_mtime_ns != artifact_modified_ns
        or artifact_stat.st_size != artifact_size_bytes
    ):
        raise ValueError("The evaluation artifact changed while the dashboard was loading. Rerun the app.")
    with EVALUATION_PATH.open(encoding="utf-8") as file:
        evaluation = json.load(file)
    required_keys = {
        "model_name",
        "test_metrics",
    }
    if not isinstance(evaluation, dict) or not required_keys.issubset(evaluation):
        raise ValueError("The saved model evaluation is missing required test results.")
    test_metrics = evaluation["test_metrics"]
    if (
        not isinstance(test_metrics, dict)
        or "accuracy" not in test_metrics
        or not np.isfinite(test_metrics["accuracy"])
        or not 0 <= test_metrics["accuracy"] <= 1
    ):
        raise ValueError("The saved model evaluation has an invalid test accuracy.")
    return evaluation


@st.cache_data
def load_insights() -> dict[str, Any]:
    if not INSIGHTS_PATH.is_file():
        raise FileNotFoundError(f"Dashboard insights were not found: {INSIGHTS_PATH}")
    with INSIGHTS_PATH.open(encoding="utf-8") as file:
        insights = json.load(file)
    required_keys = {
        "total_orders",
        "historical_return_rate",
        "return_rate_by_product_category",
        "return_rate_by_shipping_method",
    }
    if not isinstance(insights, dict) or not required_keys.issubset(insights):
        raise ValueError("Dashboard insights file is missing required aggregate metrics.")
    return insights


@st.cache_data
def load_eda_data() -> pd.DataFrame:
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Exploratory dataset was not found: {DATASET_PATH}")
    data = pd.read_csv(DATASET_PATH)
    required_columns = {
        "returned",
        "order_date",
        "product_category",
        "shipping_method",
        "coupon_used",
    }
    missing = required_columns.difference(data.columns)
    if data.empty or missing:
        raise ValueError(
            "The exploratory dataset is empty or missing required columns: "
            + ", ".join(sorted(missing))
        )
    target = pd.to_numeric(data["returned"], errors="coerce")
    if target.isna().any() or not target.isin([0, 1]).all():
        raise ValueError("The exploratory dataset's returned column must contain only 0 and 1.")
    dates = pd.to_datetime(data["order_date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("The exploratory dataset contains invalid order_date values.")
    data["returned"] = target.astype(int)
    data["order_date"] = dates
    return data


def summarize_return_rates(data: pd.DataFrame, column: str) -> pd.DataFrame:
    return (
        data.groupby(column, dropna=False, observed=True)["returned"]
        .agg(orders="size", return_rate="mean")
        .reset_index()
        .sort_values("return_rate", ascending=False)
    )


def prepare_features(data: pd.DataFrame, expected_columns: list[str]) -> pd.DataFrame:
    frame = data.copy()
    frame.columns = frame.columns.astype(str).str.strip()
    if frame.empty:
        raise ValueError("The order record contains no data.")
    outcome_columns = {
        "returned",
        "return_rate_pct",
        "return_date",
        "days_to_return",
        "return_reason",
        "refund_amount_usd",
    }
    leaked_columns = sorted(outcome_columns.intersection(frame.columns))
    if leaked_columns:
        raise ValueError(
            "Remove outcome or post-order fields before scoring: "
            + ", ".join(leaked_columns)
        )

    missing = [column for column in RAW_FEATURES if column not in frame.columns]
    if missing:
        raise ValueError(
            "The input is missing required order/customer fields: "
            + ", ".join(missing)
        )

    if "order_date" in frame.columns:
        dates = pd.to_datetime(frame["order_date"], errors="coerce")
        if dates.isna().any():
            raise ValueError("The order_date is invalid. Use a date such as 2025-04-18.")
        frame["order_month"] = dates.dt.month
        frame["order_day_of_week"] = dates.dt.dayofweek
        frame["is_weekend"] = (dates.dt.dayofweek >= 5).astype(int)
    elif not {"order_month", "order_day_of_week"}.issubset(frame.columns):
        raise ValueError(
            "Include order_date, or provide both order_month and order_day_of_week."
        )
    elif "is_weekend" not in frame.columns:
        frame["is_weekend"] = (pd.to_numeric(frame["order_day_of_week"]) >= 5).astype(int)
    if "order_date" not in frame.columns:
        frame["is_weekend"] = (pd.to_numeric(frame["order_day_of_week"]) >= 5).astype(int)

    invalid_categories: list[str] = []
    for column, options in CATEGORY_OPTIONS.items():
        frame[column] = frame[column].astype("string").str.strip()
        unknown = frame[column].isna() | ~frame[column].isin(options)
        if unknown.any():
            invalid_categories.append(
                f"{column} (allowed: {', '.join(options)})"
            )
    if invalid_categories:
        raise ValueError(
            "These categorical fields contain missing or unsupported values: "
            + "; ".join(invalid_categories)
        )

    frame["purchase_frequency_group"] = pd.cut(
        pd.to_numeric(frame["past_purchase_count"]),
        bins=[-1, 5, 15, 30, np.inf],
        labels=["Low", "Medium", "High", "Very High"],
    )
    frame["price_category"] = pd.cut(
        pd.to_numeric(frame["product_price_usd"]),
        bins=[0, 25, 100, 300, np.inf],
        labels=["Budget", "Mid Range", "Premium", "Luxury"],
    )
    frame["customer_return_history"] = pd.cut(
        pd.to_numeric(frame["past_return_rate_pct"]),
        bins=[-1, 5, 15, 30, 100],
        labels=["Low", "Moderate", "High", "Very High"],
    )

    invalid_numeric: list[str] = []
    for column in NUMERIC_FEATURES:
        if column in frame.columns:
            converted = pd.to_numeric(frame[column], errors="coerce")
            if converted.isna().any() or not np.isfinite(converted.to_numpy(dtype=float)).all():
                invalid_numeric.append(column)
            frame[column] = converted
    if invalid_numeric:
        raise ValueError(
            "These numeric fields contain blank or non-numeric values: "
            + ", ".join(invalid_numeric)
        )

    bounds = {
        "customer_age": (18, 120),
        "product_price_usd": (0.01, None),
        "discount_percent": (0, 100),
        "product_rating": (0, 5),
        "past_purchase_count": (0, None),
        "past_return_rate_pct": (0, 100),
        "session_length_minutes": (0.01, None),
        "num_product_views": (1, None),
        "delivery_time_days": (1, None),
        "delivery_delay_days": (0, None),
        "total_orders_lifetime": (1, None),
        "total_returns_lifetime": (0, None),
        "order_month": (1, 12),
        "order_day_of_week": (0, 6),
    }
    for column, (minimum, maximum) in bounds.items():
        values = frame[column]
        if (values < minimum).any() or (maximum is not None and (values > maximum).any()):
            upper_bound = f" and at most {maximum}" if maximum is not None else ""
            raise ValueError(f"{column} must be at least {minimum}{upper_bound}.")
    if (frame["total_returns_lifetime"] > frame["total_orders_lifetime"]).any():
        raise ValueError("total_returns_lifetime cannot exceed total_orders_lifetime.")

    unavailable = [column for column in expected_columns if column not in frame.columns]
    if unavailable:
        raise ValueError("Could not prepare model features: " + ", ".join(unavailable))
    return frame.loc[:, expected_columns]


def classify_score(score: float, threshold: float) -> tuple[str, str]:
    if score >= max(0.30, threshold):
        return "High return risk", "high"
    if score >= threshold:
        return "Review recommended", "review"
    return "Lower return risk", "low"


def explain_linear_prediction(
    model: Any,
    features: pd.DataFrame,
) -> pd.DataFrame | None:
    estimator = model.named_steps.get("model")
    preprocessor = model.named_steps.get("preprocessor")
    if estimator is None or preprocessor is None or not hasattr(estimator, "coef_"):
        return None

    transformed = preprocessor.transform(features)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    contributions = np.asarray(transformed)[0] * estimator.coef_[0]
    transformed_names = preprocessor.get_feature_names_out()
    original_features = list(features.columns)
    grouped: dict[str, float] = {}
    for name, contribution in zip(transformed_names, contributions):
        transformed_name = str(name).split("__", maxsplit=1)[-1]
        source = next(
            (
                column
                for column in original_features
                if transformed_name == column
                or transformed_name.startswith(f"{column}_")
            ),
            transformed_name,
        )
        grouped[source] = grouped.get(source, 0.0) + float(contribution)

    explanation = pd.DataFrame(
        [
            {
                "Feature": feature,
                "Contribution to score": contribution,
                "Direction": "Raises score" if contribution > 0 else "Lowers score",
            }
            for feature, contribution in grouped.items()
        ]
    )
    return (
        explanation.assign(_absolute=explanation["Contribution to score"].abs())
        .sort_values("_absolute", ascending=False)
        .drop(columns="_absolute")
        .head(8)
    )


def render_page_header(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="eyebrow">{eyebrow}</div>
        <h1 class="page-title">{title}</h1>
        <p class="page-subtitle">{subtitle}</p>
        """,
        unsafe_allow_html=True,
    )


try:
    model_stat = MODEL_PATH.stat()
    evaluation_stat = EVALUATION_PATH.stat()
    detector, saved_threshold, model_name = load_detector(
        model_stat.st_mtime_ns,
        model_stat.st_size,
    )
    evaluation = load_evaluation(
        evaluation_stat.st_mtime_ns,
        evaluation_stat.st_size,
    )
    insights = load_insights()
except (FileNotFoundError, ValueError, OSError) as exc:
    st.error(f"Dashboard setup error: {exc}")
    st.info("Keep app.py beside the datasets and models folders, then verify the trained model files.")
    st.stop()

expected_features = list(detector.feature_names_in_)
if evaluation["model_name"] != model_name:
    st.error("The deployed model and saved evaluation result refer to different models.")
    st.stop()

with st.sidebar:
    st.markdown(
        '<span class="brand-mark">R</span><span class="brand-name">Returns Desk</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sidebar-caption">A practical workspace for order reviews and return patterns.</div>',
        unsafe_allow_html=True,
    )
    page = st.radio(
        "YOUR WORKSPACE",
        [
            "Overview",
            "Predict an order",
            "Exploratory analysis",
            "Model evaluation",
        ],
        label_visibility="visible",
    )
    st.markdown("---")
    st.caption("CURRENT DETECTOR")
    st.markdown(f"**{model_name} · return review**")

threshold = float(saved_threshold)

if page == "Overview":
    total_orders = int(insights["total_orders"])
    historical_return_rate = float(insights["historical_return_rate"])
    returned_orders = int(insights["returned_orders"])
    render_page_header(
        "RETURNS DESK  /  PORTFOLIO",
        "A clearer view of what comes back.",
        "A practical snapshot of historical order volume and customer return patterns.",
    )
    st.markdown(
        f"""
        <div class="hero-card">
            <div>
                <p class="hero-kicker">Across the order sample</p>
                <p class="hero-stat">{historical_return_rate:.1%}</p>
                <p class="hero-caption">
                    of {total_orders:,} orders were returned. That's about one in
                    {max(1, round(1 / historical_return_rate))} orders in this dataset.
                    Use it as context for today's reviews, not as a forecast.
                </p>
            </div>
            <div class="hero-aside">
                <span class="hero-aside-label">For the team</span>
                <strong>Start with the order. Let the data add context.</strong>
                <span>Score an individual order, then review its details before acting.</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    kpi_cols = st.columns(4, gap="medium")
    kpi_cols[0].metric("Orders", f"{total_orders:,}")
    kpi_cols[1].metric("Returned", f"{returned_orders:,}")
    kpi_cols[2].metric(
        "Product categories",
        f"{len(insights['return_rate_by_product_category']):,}",
    )
    kpi_cols[3].metric(
        "Shipping methods",
        f"{len(insights['return_rate_by_shipping_method']):,}",
    )

    st.markdown(
        """
        <div class="note-card">
            Historical return rates describe this dataset, not a forecast.
            The model's held-out evaluation is available on the separate Model evaluation page.
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.2, 1])
    with left:
        st.markdown('<div class="section-heading">Returns by product category</div>', unsafe_allow_html=True)
        category_rates = pd.Series(
            {
                item["name"]: float(item["return_rate"]) * 100
                for item in insights["return_rate_by_product_category"]
            },
            name="Return rate (%)",
        ).sort_values(ascending=False)
        st.bar_chart(category_rates, horizontal=True, color="#bd583d")
    with right:
        st.markdown('<div class="section-heading">Returns by shipping method</div>', unsafe_allow_html=True)
        shipping_rates = pd.Series(
            {
                item["name"]: float(item["return_rate"]) * 100
                for item in insights["return_rate_by_shipping_method"]
            },
            name="Return rate (%)",
        ).sort_values(ascending=False)
        st.bar_chart(shipping_rates, horizontal=True, color="#bd7d3d")

elif page == "Predict an order":
    render_page_header(
        "ORDER SCORING",
        "Predict return risk",
        f"Enter the order details below. {model_name} will return an estimated score and a review recommendation.",
    )

    with st.expander("Decision threshold", expanded=False):
        threshold = st.slider(
            "Flag an order for review when its score is at least",
            min_value=0.0,
            max_value=1.0,
            value=float(saved_threshold),
            step=0.005,
            format="%.2f",
            help="The default was selected on the validation period. Raising it reduces reviews but may miss more returns.",
        )

    with st.form("single_order_form"):
        st.markdown("#### Order and customer")
        col1, col2, col3 = st.columns(3)
        with col1:
            order_date = st.date_input("Order date")
            customer_age = st.number_input("Customer age", min_value=18, max_value=100, value=35)
            country = st.selectbox("Country", CATEGORY_OPTIONS["country"], index=6)
            device_type = st.selectbox("Device", CATEGORY_OPTIONS["device_type"], index=1)
        with col2:
            past_purchase_count = st.number_input("Past purchases", min_value=0, max_value=10000, value=8)
            past_return_rate_pct = st.number_input("Past return rate (%)", min_value=0.0, max_value=100.0, value=10.0, step=0.5)
            total_orders_lifetime = st.number_input("Lifetime orders", min_value=1, max_value=10000, value=9)
            total_returns_lifetime = st.number_input("Lifetime returns", min_value=0, max_value=10000, value=1)
        with col3:
            session_length_minutes = st.number_input("Session length (minutes)", min_value=0.1, max_value=1000.0, value=12.0, step=0.5)
            num_product_views = st.number_input("Product views", min_value=1, max_value=1000, value=8)
            shipping_method = st.selectbox("Shipping method", CATEGORY_OPTIONS["shipping_method"])
            payment_method = st.selectbox("Payment method", CATEGORY_OPTIONS["payment_method"])

        st.markdown("#### Product")
        col1, col2, col3 = st.columns(3)
        with col1:
            product_category = st.selectbox("Product category", CATEGORY_OPTIONS["product_category"])
            product_price_usd = st.number_input("Product price (USD)", min_value=0.01, max_value=100000.0, value=65.0, step=1.0)
        with col2:
            discount_percent = st.number_input("Discount (%)", min_value=0.0, max_value=100.0, value=10.0, step=0.5)
            coupon_used = st.selectbox("Coupon used", CATEGORY_OPTIONS["coupon_used"])
        with col3:
            product_rating = st.number_input("Product rating", min_value=0.0, max_value=5.0, value=4.0, step=0.1)
            delivery_time_days = st.number_input("Delivery time (days)", min_value=1, max_value=365, value=4)
            delivery_delay_days = st.number_input("Delivery delay (days)", min_value=0, max_value=365, value=0)

        submitted = st.form_submit_button("Calculate return risk", type="primary", width="stretch")

    if submitted:
        order = pd.DataFrame(
            [
                {
                    "order_date": pd.Timestamp(order_date),
                    "customer_age": customer_age,
                    "country": country,
                    "device_type": device_type,
                    "product_category": product_category,
                    "product_price_usd": product_price_usd,
                    "discount_percent": discount_percent,
                    "coupon_used": coupon_used,
                    "product_rating": product_rating,
                    "past_purchase_count": past_purchase_count,
                    "past_return_rate_pct": past_return_rate_pct,
                    "session_length_minutes": session_length_minutes,
                    "num_product_views": num_product_views,
                    "shipping_method": shipping_method,
                    "delivery_time_days": delivery_time_days,
                    "delivery_delay_days": delivery_delay_days,
                    "payment_method": payment_method,
                    "total_orders_lifetime": total_orders_lifetime,
                    "total_returns_lifetime": total_returns_lifetime,
                }
            ]
        )
        try:
            features = prepare_features(order, expected_features)
            score = float(detector.predict_proba(features)[0, 1])
        except (ValueError, KeyError) as exc:
            st.error(f"Could not score this order: {exc}")
            st.stop()
        label, risk_level = classify_score(score, threshold)
        st.markdown("---")
        st.markdown(
            f"""
            <div class="result-card result-{risk_level}" role="status">
                <div>
                    <div class="result-label">Estimated return score</div>
                    <div class="result-score">{score:.1%}</div>
                    <div class="result-meter" aria-label="Score {score:.1%}">
                        <span style="width: {min(max(score, 0.0), 1.0):.1%}"></span>
                    </div>
                </div>
                <div class="result-risk">
                    {label}
                    <small>Review threshold: {threshold:.1%}</small>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if risk_level == "high":
            st.warning("Prioritize this order for a return-prevention or fulfillment review.")
        elif risk_level == "review":
            st.info("This order meets the model's review threshold. Check the order details before taking action.")
        else:
            st.success("This order is below the current review threshold.")
        st.caption(
            "This is a model score, not a guarantee. Use it to support—not replace—business decisions."
        )
        explanation = explain_linear_prediction(detector, features)
        if explanation is not None:
            with st.expander("Why did the model assign this score?", expanded=False):
                st.dataframe(
                    explanation.style.format(
                        {"Contribution to score": "{:+.3f}"}
                    ),
                    hide_index=True,
                    width="stretch",
                )
                st.caption(
                    "Contributions are in model log-odds space and describe this prediction; "
                    "they are not causal effects."
                )

elif page == "Exploratory analysis":
    try:
        eda_data = load_eda_data()
    except (FileNotFoundError, ValueError, OSError) as exc:
        st.error(f"Could not load exploratory data: {exc}")
        st.stop()

    return_rate = float(eda_data["returned"].mean())
    render_page_header(
        "EXPLORATORY DATA ANALYSIS",
        "Patterns behind product returns",
        "Explore order outcomes, customer and product segments, delivery patterns, and the strength of numeric associations.",
    )

    kpi_cols = st.columns(2, gap="medium")
    kpi_cols[0].metric("Orders analyzed", f"{len(eda_data):,}")
    kpi_cols[1].metric("Returned orders", f"{int(eda_data['returned'].sum()):,}")
    kpi_cols = st.columns(2, gap="medium")
    kpi_cols[0].metric("Overall return rate", f"{return_rate:.1%}")
    kpi_cols[1].metric(
        "Median product price",
        f"${eda_data['product_price_usd'].median():,.2f}"
        if "product_price_usd" in eda_data
        else "Not available",
    )

    returned_label = eda_data["returned"].map({0: "Not returned", 1: "Returned"})
    outcome_counts = returned_label.value_counts().rename_axis("Outcome").to_frame("Orders")
    category_rates = summarize_return_rates(eda_data, "product_category")
    month_data = eda_data.assign(
        order_month=pd.to_datetime(eda_data["order_date"], errors="coerce").dt.month
    )
    monthly_rates = (
        month_data.groupby("order_month", dropna=True)["returned"]
        .agg(orders="size", return_rate="mean")
        .reset_index()
        .rename(columns={"order_month": "Month"})
    )
    delay_column = "delivery_delay_days"
    if delay_column in eda_data:
        delay_data = eda_data.assign(
            delivery_status=np.select(
                [
                    eda_data[delay_column].eq(0),
                    eda_data[delay_column].le(2),
                ],
                ["On time", "1-2 days late"],
                default="3+ days late",
            )
        )
        delay_rates = summarize_return_rates(delay_data, "delivery_status")
    else:
        delay_rates = pd.DataFrame()
    coupon_rates = summarize_return_rates(eda_data, "coupon_used")

    st.markdown('<div class="section-heading">Order outcomes and category patterns</div>', unsafe_allow_html=True)
    left, right = st.columns(2, gap="small")
    with left:
        st.markdown("**Returned vs. kept orders**")
        st.bar_chart(outcome_counts, horizontal=True, color="#bd583d", height=180)
        st.caption("The target is imbalanced, so accuracy alone can hide missed returns.")
    with right:
        st.markdown("**Return rate by product category**")
        st.bar_chart(
            category_rates.set_index("product_category")[["return_rate"]].rename(
                columns={"return_rate": "Return rate"}
            ),
            horizontal=True,
            color="#bd7d3d",
            height=210,
        )

    st.markdown('<div class="section-heading">Timing, delivery, and promotion</div>', unsafe_allow_html=True)
    left, right = st.columns(2, gap="small")
    with left:
        st.markdown("**Monthly return rate**")
        st.line_chart(
            monthly_rates.set_index("Month")[["return_rate"]],
            color="#bd583d",
            height=210,
        )
        st.caption("Monthly rates are descriptive; they do not establish a seasonal cause.")
    with right:
        st.markdown("**Return rate by shipping method**")
        shipping_rates = summarize_return_rates(eda_data, "shipping_method")
        st.bar_chart(
            shipping_rates.set_index("shipping_method")[["return_rate"]].rename(
                columns={"return_rate": "Return rate"}
            ),
            horizontal=True,
            color="#bd7d3d",
            height=210,
        )

    st.markdown('<div class="section-heading">Delivery and numeric distributions</div>', unsafe_allow_html=True)
    left, right = st.columns(2, gap="small")
    with left:
        st.markdown("**Return rate by delivery delay**")
        if not delay_rates.empty:
            st.bar_chart(
                delay_rates.set_index("delivery_status")[["return_rate"]].rename(
                    columns={"return_rate": "Return rate"}
                ),
                horizontal=True,
                color="#bd7d3d",
                height=210,
            )
        else:
            st.info("Delivery-delay data is not available in this dataset.")
    with right:
        distribution_options = [
            column
            for column in [
                "customer_age",
                "product_price_usd",
                "discount_percent",
                "product_rating",
                "past_purchase_count",
                "delivery_delay_days",
            ]
            if column in eda_data
        ]
        distribution_column = st.selectbox("Numeric feature", distribution_options)
        distribution_bins = pd.cut(eda_data[distribution_column].dropna(), bins=10)
        distribution_counts = (
            distribution_bins.value_counts(sort=False)
            .rename_axis("Range")
            .to_frame("Orders")
        )
        distribution_counts.index = distribution_counts.index.astype(str)
        st.markdown(f"**Distribution of {distribution_column.replace('_', ' ')}**")
        st.bar_chart(distribution_counts, color="#bd583d", height=210)

    st.markdown('<div class="section-heading">Customer and order signals</div>', unsafe_allow_html=True)
    left, right = st.columns(2, gap="small")
    with left:
        st.markdown("**Return rate by coupon use**")
        st.bar_chart(
            coupon_rates.set_index("coupon_used")[["return_rate"]].rename(
                columns={"return_rate": "Return rate"}
            ),
            color="#bd583d",
            height=190,
        )
    with right:
        numeric_candidates = [
            column
            for column in [
                "customer_age",
                "product_price_usd",
                "discount_percent",
                "product_rating",
                "past_return_rate_pct",
                "delivery_delay_days",
            ]
            if column in eda_data
        ]
        numeric_correlations = (
            eda_data[numeric_candidates]
            .corrwith(eda_data["returned"], method="spearman")
            .dropna()
            .sort_values()
            .to_frame("Spearman correlation with return")
        )
        st.markdown("**Numeric associations with returns**")
        st.bar_chart(
            numeric_correlations,
            horizontal=True,
            color="#bd7d3d",
            height=210,
        )
        st.caption("Associations are not causal effects; values near zero have little monotonic relationship.")

    top_category = category_rates.iloc[0]
    low_category = category_rates.iloc[-1]
    top_coupon = coupon_rates.iloc[0]
    insight_text = (
        f"**Category spread:** {top_category['product_category']} has the highest observed "
        f"return rate ({top_category['return_rate']:.1%}); "
        f"{low_category['product_category']} has the lowest ({low_category['return_rate']:.1%}). "
        f"The gap is {(top_category['return_rate'] - low_category['return_rate']):.1%}.\n\n"
        f"**Coupon segment:** {top_coupon['coupon_used']} orders have a "
        f"{top_coupon['return_rate']:.1%} observed return rate."
    )
    st.info(insight_text + "\n\nSegment rates are descriptive; compare sample sizes and do not infer causation.")

    with st.expander("Segment counts and return rates", expanded=False):
        display_rates = category_rates.rename(
            columns={"product_category": "Product category", "orders": "Orders", "return_rate": "Return rate"}
        )
        display_rates["Return rate"] = display_rates["Return rate"].map(lambda value: f"{value:.1%}")
        st.dataframe(display_rates, hide_index=True, width="stretch")
        st.markdown("**Data quality**")
        quality = pd.DataFrame(
            {
                "Missing values": eda_data.isna().sum(),
                "Distinct values": eda_data.nunique(dropna=True),
            }
        )
        st.dataframe(quality, width="stretch")

elif page == "Model evaluation":
    render_page_header(
        "HELD-OUT MODEL EVALUATION",
        "Accuracy, shown plainly",
        "This is the deployed model’s accuracy on a chronological holdout that was not used for training or model selection.",
    )
    test_metrics = evaluation["test_metrics"]
    st.metric("Accuracy", f"{test_metrics['accuracy']:.1%}")
    st.info(
        "This accuracy-first model mostly predicts the majority class and misses most returns. "
        "Accuracy alone does not establish return-detection effectiveness or production readiness. "
        "See the executed evaluation notebook for the class-balance and cross-validation analysis."
    )

st.markdown("---")
st.markdown(
    '<div class="small-muted">E-commerce Product Return Prediction · Decision-support tool · Validate recommendations against operational outcomes.</div>',
    unsafe_allow_html=True,
)
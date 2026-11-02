from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "best_return_detection_model.joblib"
METRICS_PATH = ROOT_DIR / "models" / "two_model_comparison_metrics.csv"
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
        padding-top: 2rem;
        padding-bottom: 3rem;
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
        margin: 0 0 23px;
    }
    .section-heading {
        color: var(--ink);
        font-family: Georgia, 'Times New Roman', serif;
        font-size: 21px;
        font-weight: 700;
        margin: 26px 0 12px;
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
    div[data-testid="stFileUploader"] section {
        background: var(--panel);
        border: 1px dashed #c9c1b6;
        border-radius: 9px;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 8px;
        overflow: hidden;
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
def load_detector() -> tuple[Any, float]:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained model was not found: {MODEL_PATH}")
    artifact = joblib.load(MODEL_PATH)
    if not isinstance(artifact, dict) or "model" not in artifact or "threshold" not in artifact:
        raise ValueError("The saved return-detection artifact is missing its model or threshold.")
    return artifact["model"], float(artifact["threshold"])


@st.cache_data
def load_metrics() -> pd.DataFrame:
    if not METRICS_PATH.is_file():
        raise FileNotFoundError(f"Evaluation metrics were not found: {METRICS_PATH}")
    return pd.read_csv(METRICS_PATH)


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


def prepare_features(data: pd.DataFrame, expected_columns: list[str]) -> pd.DataFrame:
    frame = data.copy()
    frame.columns = frame.columns.astype(str).str.strip()
    if frame.empty:
        raise ValueError("The uploaded file contains no order rows.")

    missing = [column for column in RAW_FEATURES if column not in frame.columns]
    if missing:
        raise ValueError(
            "The input is missing required order/customer fields: "
            + ", ".join(missing)
        )

    if "order_date" in frame.columns:
        dates = pd.to_datetime(frame["order_date"], errors="coerce")
        if dates.isna().any():
            raise ValueError("Some order_date values are invalid. Use dates such as 2025-04-18.")
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

    for column in CATEGORY_OPTIONS:
        frame[column] = frame[column].astype("string").str.strip()

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
    detector, saved_threshold = load_detector()
    metrics = load_metrics()
    insights = load_insights()
except (FileNotFoundError, ValueError, OSError) as exc:
    st.error(f"Dashboard setup error: {exc}")
    st.info("Keep app.py beside the datasets and models folders, then verify the trained model files.")
    st.stop()

expected_features = list(detector.feature_names_in_)
xgb_row = metrics.loc[metrics["Model"].eq("XGBoost")]
if xgb_row.empty:
    st.error("XGBoost evaluation metrics are missing from the saved comparison.")
    st.stop()
xgb_metrics = xgb_row.iloc[0]

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
        ["Overview", "Predict an order", "Score a file", "Insights & model"],
        label_visibility="visible",
    )
    st.markdown("---")
    st.caption("CURRENT DETECTOR")
    st.markdown("**XGBoost · return review**")
    st.caption("Review score uses a validation-tuned cutoff")

threshold = float(saved_threshold)

if page == "Overview":
    total_orders = int(insights["total_orders"])
    historical_return_rate = float(insights["historical_return_rate"])
    returned_orders = int(insights["returned_orders"])
    render_page_header(
        "RETURNS DESK  /  PORTFOLIO",
        "A clearer view of what comes back.",
        "A practical snapshot of historical orders and how well the return detector finds them.",
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
                <span>Score a single order or a whole file, then review the details before acting.</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    kpi_cols = st.columns(4, gap="medium")
    kpi_cols[0].metric("Orders", f"{total_orders:,}")
    kpi_cols[1].metric("Returned", f"{returned_orders:,}")
    kpi_cols[2].metric("Detector ROC-AUC", f"{xgb_metrics['Test ROC-AUC']:.3f}")
    kpi_cols[3].metric("Return recall", f"{xgb_metrics['Test Recall (F1 threshold)']:.1%}")

    st.markdown(
        f"""
        <div class="note-card">
            <strong>Reading the model:</strong> The test-set return rate is
            {historical_return_rate:.1%}. At its validation-selected review cutoff, the detector
            found {xgb_metrics['Test Recall (F1 threshold)']:.1%} of returned orders, and
            {xgb_metrics['Test Precision (F1 threshold)']:.1%} of flagged orders were returns.
            Accuracy alone can hide missed returns when most orders are kept.
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
        st.markdown('<div class="section-heading">How the detector performed</div>', unsafe_allow_html=True)
        snapshot = pd.DataFrame(
            {
                "Measure": ["Precision · flagged orders that returned", "Recall · returns found", "F1 · precision and recall balance", "ROC-AUC · ranking quality"],
                "Result": [
                    xgb_metrics["Test Precision (F1 threshold)"],
                    xgb_metrics["Test Recall (F1 threshold)"],
                    xgb_metrics["Test F1 (F1 threshold)"],
                    xgb_metrics["Test ROC-AUC"],
                ],
            }
        )
        snapshot["Result"] = snapshot["Result"].map(lambda value: f"{value:.3f}")
        st.dataframe(snapshot, hide_index=True, width="stretch")
        st.caption("Held-out test results. The review cutoff was chosen on validation data.")

elif page == "Predict an order":
    render_page_header(
        "ORDER SCORING",
        "Predict return risk",
        "Enter the order details below. The saved XGBoost model will return an estimated score and a review recommendation.",
    )

    with st.expander("Decision threshold", expanded=False):
        threshold = st.slider(
            "Flag an order for review when its score is at least",
            min_value=0.05,
            max_value=0.80,
            value=float(np.clip(saved_threshold, 0.05, 0.80)),
            step=0.01,
            format="%.2f",
            help="The default is the threshold chosen on validation data. Raising it reduces reviews but may miss more returns.",
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
        features = prepare_features(order, expected_features)
        score = float(detector.predict_proba(features)[0, 1])
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
        st.caption("This is a model score, not a guarantee. Use it to support—not replace—business decisions.")

elif page == "Score a file":
    render_page_header(
        "BATCH SCORING",
        "Score multiple orders",
        "Upload a CSV with one order per row, or download the template to check the required columns.",
    )

    template = pd.DataFrame(
        [
            {
                "order_date": "2025-04-18",
                "customer_age": 35,
                "country": "US",
                "device_type": "Mobile",
                "product_category": "Clothing",
                "product_price_usd": 65.0,
                "discount_percent": 10.0,
                "coupon_used": "No",
                "product_rating": 4.0,
                "past_purchase_count": 8,
                "past_return_rate_pct": 10.0,
                "session_length_minutes": 12.0,
                "num_product_views": 8,
                "shipping_method": "Standard",
                "delivery_time_days": 4,
                "delivery_delay_days": 0,
                "payment_method": "Credit Card",
                "total_orders_lifetime": 9,
                "total_returns_lifetime": 1,
            }
        ]
    )
    st.download_button(
        "Download CSV template",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="return_prediction_template.csv",
        mime="text/csv",
    )
    st.markdown(
        '<div class="note-card">Required inputs are customer, product, purchase-history, shipping, and payment fields shown in the template. The app creates purchase-frequency, price, and return-history groups from those values. An order date is required to generate calendar features.</div>',
        unsafe_allow_html=True,
    )
    upload = st.file_uploader("Choose an order CSV", type=["csv"])
    if upload is not None and st.button("Score uploaded orders", type="primary"):
        try:
            uploaded_data = pd.read_csv(upload)
            batch_features = prepare_features(uploaded_data, expected_features)
            scores = detector.predict_proba(batch_features)[:, 1]
            predictions = (scores >= threshold).astype(int)
            result = uploaded_data.copy()
            result["estimated_return_score"] = scores
            result["return_risk"] = [
                classify_score(float(score), threshold)[0] for score in scores
            ]
            result["flag_for_review"] = predictions
            result = result.sort_values("estimated_return_score", ascending=False)
        except (ValueError, KeyError, pd.errors.ParserError) as exc:
            st.error(f"Could not score this file: {exc}")
        else:
            flagged = int(result["flag_for_review"].sum())
            first, second, third = st.columns(3)
            first.metric("Orders scored", f"{len(result):,}")
            second.metric("Flagged for review", f"{flagged:,}")
            third.metric("Flagged share", f"{flagged / len(result):.1%}" if len(result) else "0.0%")
            st.dataframe(result, hide_index=True, width="stretch")
            st.download_button(
                "Download scored orders",
                data=result.to_csv(index=False).encode("utf-8"),
                file_name="scored_return_risk.csv",
                mime="text/csv",
            )

else:
    render_page_header(
        "MODEL TRANSPARENCY",
        "Insights and model performance",
        "A concise view of how the return detector performs, with supporting detail when you need it.",
    )
    st.markdown(
        """
        <div class="note-card">
            The deployed XGBoost model is tuned to catch returned orders. The scores below
            summarize its held-out test results; expand the sections below for the full
            comparison and supporting charts.
        </div>
        """,
        unsafe_allow_html=True,
    )

    recall_col, precision_col = st.columns(2, gap="medium")
    recall_col.metric("Returns found · test recall", f"{xgb_metrics['Test Recall (F1 threshold)']:.1%}")
    precision_col.metric("Flagged orders returned · test precision", f"{xgb_metrics['Test Precision (F1 threshold)']:.1%}")

    with st.expander("Detailed model comparison", expanded=False):
        display_metrics = metrics.copy()
        display_metrics = display_metrics[
            [
                "Model",
                "Test Accuracy (0.50)",
                "Test Precision (0.50)",
                "Test Recall (0.50)",
                "Test F1 (0.50)",
                "Test Accuracy (F1 threshold)",
                "Test Precision (F1 threshold)",
                "Test Recall (F1 threshold)",
                "Test F1 (F1 threshold)",
                "Test ROC-AUC",
            ]
        ]
        display_metrics = display_metrics.rename(
            columns={
                "Test Accuracy (0.50)": "Accuracy @ 0.50",
                "Test Precision (0.50)": "Precision @ 0.50",
                "Test Recall (0.50)": "Recall @ 0.50",
                "Test F1 (0.50)": "F1 @ 0.50",
                "Test Accuracy (F1 threshold)": "Accuracy @ tuned threshold",
                "Test Precision (F1 threshold)": "Precision @ tuned threshold",
                "Test Recall (F1 threshold)": "Recall @ tuned threshold",
                "Test F1 (F1 threshold)": "F1 @ tuned threshold",
                "Test ROC-AUC": "ROC-AUC",
            }
        )
        st.dataframe(
            display_metrics.style.format(
                {column: "{:.3f}" for column in display_metrics.columns if column != "Model"}
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            f"The deployed XGBoost alert cutoff is {saved_threshold:.1%}, selected using validation data. "
            "The 0.50 columns use the standard cutoff."
        )

    transformed_names = detector.named_steps["preprocessor"].get_feature_names_out()
    importances = detector.named_steps["model"].feature_importances_
    importance_table = (
        pd.DataFrame({"Feature": transformed_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .head(15)
        .set_index("Feature")
    )
    with st.expander("Top model signals", expanded=False):
        st.bar_chart(importance_table, horizontal=True, color="#bd583d")
        st.caption("Feature importance describes model split usage; it does not imply causation.")

    with st.expander("Observed return patterns", expanded=False):
        left, right = st.columns(2)
        with left:
            rates = pd.Series(
                {
                    item["name"]: float(item["return_rate"]) * 100
                    for item in insights["return_rate_by_product_category"]
                },
                name="Return rate (%)",
            ).sort_values(ascending=False)
            st.markdown("**Return rate by product category**")
            st.bar_chart(rates, horizontal=True, color="#bd583d")
        with right:
            rates = pd.Series(
                {
                    item["name"]: float(item["return_rate"]) * 100
                    for item in insights["return_rate_by_shipping_method"]
                },
                name="Return rate (%)",
            ).sort_values(ascending=False)
            st.markdown("**Return rate by shipping method**")
            st.bar_chart(rates, horizontal=True, color="#bd7d3d")

st.markdown("---")
st.markdown(
    '<div class="small-muted">E-commerce Product Return Prediction · Decision-support tool · Validate recommendations against operational outcomes.</div>',
    unsafe_allow_html=True,
)
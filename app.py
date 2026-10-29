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
    page_title="Return Intelligence | E-commerce Analytics",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #172a3a;
        --muted: #718096;
        --teal: #087f8c;
        --teal-dark: #07636d;
        --mint: #e7f5f2;
        --paper: #f5f8fa;
        --line: #e4ebef;
    }

    html, body, [class*="css"] {
        font-family: 'Aptos', 'Segoe UI', sans-serif;
        color: var(--ink);
    }

    .stApp { background: var(--paper); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] {
        background: #102b3a;
        border-right: 1px solid #214452;
    }
    [data-testid="stSidebar"] * { color: #eaf3f5; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        padding: 0.35rem 0.55rem;
        border-radius: 8px;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: #1b3d4c;
    }

    .brand-mark {
        display: inline-flex;
        width: 38px;
        height: 38px;
        align-items: center;
        justify-content: center;
        border: 1px solid #6cc7c3;
        border-radius: 12px;
        color: #a7e4dc;
        font-family: 'Aptos Display', 'Segoe UI', sans-serif;
        font-weight: 800;
        font-size: 17px;
        margin-right: 10px;
        vertical-align: middle;
    }
    .brand-name {
        display: inline-block;
        color: #f4fbfc;
        font-family: 'Aptos Display', 'Segoe UI', sans-serif;
        font-size: 16px;
        font-weight: 700;
        vertical-align: middle;
    }
    .sidebar-caption {
        color: #a8c1c8;
        font-size: 12px;
        line-height: 1.6;
        margin: 14px 0 24px;
    }
    .eyebrow {
        color: var(--teal);
        font-size: 16px;
        font-weight: 800;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    .page-title {
        color: var(--ink);
        font-family: 'Aptos Display', 'Segoe UI', sans-serif;
        font-size: clamp(28px, 3vw, 38px);
        font-weight: 800;
        letter-spacing: -0.04em;
        line-height: 1.15;
        margin: 0 0 7px;
    }
    .page-subtitle {
        color: var(--muted);
        font-size: 14px;
        margin: 0 0 18px;
    }
    .section-heading {
        color: var(--ink);
        font-family: 'Aptos Display', 'Segoe UI', sans-serif;
        font-size: 19px;
        font-weight: 700;
        margin: 24px 0 12px;
    }
    .hero-card {
        background: linear-gradient(120deg, #123748, #087f8c);
        border-radius: 18px;
        color: white;
        padding: 22px 26px;
        margin: 8px 0 18px;
    }
    .hero-card h2 {
        color: white;
        font-family: 'Aptos Display', 'Segoe UI', sans-serif;
        font-size: 24px;
        margin: 0 0 8px;
    }
    .hero-card p {
        color: #d7eff0;
        font-size: 13px;
        line-height: 1.5;
        margin: 0;
        max-width: 760px;
    }
    .note-card {
        background: white;
        border: 1px solid var(--line);
        border-left: 4px solid var(--teal);
        border-radius: 10px;
        color: #405366;
        font-size: 13px;
        line-height: 1.6;
        padding: 14px 17px;
        margin: 12px 0 20px;
    }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid var(--line);
        border-radius: 13px;
        padding: 15px 17px;
    }
    div[data-testid="stMetricLabel"] { color: var(--muted); }
    div[data-testid="stMetricValue"] {
        color: var(--ink);
        font-family: 'Manrope', sans-serif;
    }
    .stButton > button, .stDownloadButton > button {
        border-radius: 9px;
        font-weight: 600;
        min-height: 42px;
    }
    .stButton > button[kind="primary"] {
        background: var(--teal);
        border-color: var(--teal);
    }
    .stButton > button[kind="primary"]:hover {
        background: var(--teal-dark);
        border-color: var(--teal-dark);
    }
    div[data-testid="stForm"] {
        background: white;
        border: 1px solid var(--line);
        border-radius: 15px;
        padding: 20px;
    }
    .small-muted {
        color: var(--muted);
        font-size: 12px;
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
        '<span class="brand-mark">R</span><span class="brand-name">Return Intelligence</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sidebar-caption">E-commerce product return analytics and decision support</div>',
        unsafe_allow_html=True,
    )
    page = st.radio(
        "WORKSPACE",
        ["Overview", "Predict an order", "Score a file", "Insights & model"],
        label_visibility="visible",
    )
    st.markdown("---")
    st.caption("Model: XGBoost")
    st.caption(f"Return-review threshold: {saved_threshold:.1%}")

threshold = float(saved_threshold)

if page == "Overview":
    render_page_header(
        "E-COMMERCE ANALYTICS",
        "Return intelligence",
        "See return risk at a glance.",
    )
    st.markdown(
        """
        <div class="hero-card">
            <h2>Turn order data into clearer decisions.</h2>
            <p>
                Score an order or upload a batch to find returns that may need review.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    total_orders = int(insights["total_orders"])
    historical_return_rate = float(insights["historical_return_rate"])
    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Orders analyzed", f"{total_orders:,}")
    kpi_cols[1].metric("Historical return rate", f"{historical_return_rate:.1%}")
    kpi_cols[2].metric("XGBoost ROC-AUC", f"{xgb_metrics['Test ROC-AUC']:.3f}")
    kpi_cols[3].metric("F1-tuned test recall", f"{xgb_metrics['Test Recall (F1 threshold)']:.1%}")

    st.markdown(
        f"""
        <div class="note-card">
            <strong>Accuracy can mislead:</strong> At the default cutoff, accuracy matches the
            {max(historical_return_rate, 1 - historical_return_rate):.1%} baseline and misses most returns.
            The tuned cutoff finds {xgb_metrics['Test Recall (F1 threshold)']:.1%} of returns;
            {xgb_metrics['Test Precision (F1 threshold)']:.1%} of flagged orders are returns.
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.2, 1])
    with left:
        st.markdown('<div class="section-heading">Return rate by product category</div>', unsafe_allow_html=True)
        category_rates = pd.Series(
            {
                item["name"]: float(item["return_rate"]) * 100
                for item in insights["return_rate_by_product_category"]
            },
            name="Return rate (%)",
        ).sort_values(ascending=False)
        st.bar_chart(category_rates, horizontal=True, color="#087f8c")
    with right:
        st.markdown('<div class="section-heading">Model snapshot</div>', unsafe_allow_html=True)
        snapshot = pd.DataFrame(
            {
                "Measure": ["Test precision at tuned threshold", "Test recall at tuned threshold", "Test F1 at tuned threshold", "Test ROC-AUC"],
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
        st.caption("Metrics are from the held-out test set; threshold was chosen using validation data.")

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
        score_col, label_col = st.columns([1, 2])
        score_col.metric("Estimated return score", f"{score:.1%}")
        label_col.markdown(f"### {label}")
        st.progress(min(max(score, 0.0), 1.0), text=f"Review threshold: {threshold:.1%}")
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
        "Review the test-set results and the strongest signals used by the selected return detector.",
    )
    st.markdown(
        """
        <div class="note-card">
            The model was selected for return detection using validation F1, not raw accuracy.
            The source dataset is imbalanced, so a model can achieve high accuracy by predicting
            that nearly every order will not be returned. Always review precision, recall, F1,
            and the operational cost of investigating flagged orders.
        </div>
        """,
        unsafe_allow_html=True,
    )

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
        f"The deployed XGBoost alert threshold is {saved_threshold:.1%}, selected using validation data. "
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
    st.markdown('<div class="section-heading">Top model signals</div>', unsafe_allow_html=True)
    st.bar_chart(importance_table, horizontal=True, color="#087f8c")
    st.caption("Importance indicates how much a feature contributed to model splits; it does not imply causation.")

    st.markdown('<div class="section-heading">Observed return patterns</div>', unsafe_allow_html=True)
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
        st.bar_chart(rates, horizontal=True, color="#087f8c")
    with right:
        rates = pd.Series(
            {
                item["name"]: float(item["return_rate"]) * 100
                for item in insights["return_rate_by_shipping_method"]
            },
            name="Return rate (%)",
        ).sort_values(ascending=False)
        st.markdown("**Return rate by shipping method**")
        st.bar_chart(rates, horizontal=True, color="#4b9b8e")

st.markdown("---")
st.markdown(
    '<div class="small-muted">E-commerce Product Return Prediction · Decision-support tool · Validate recommendations against operational outcomes.</div>',
    unsafe_allow_html=True,
)
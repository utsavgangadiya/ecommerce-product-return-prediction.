from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent
DATASET_PATH = ROOT_DIR / "datasets" / "ecommerce_return_prediction_feature_engineered.csv"
OUTPUT_PATH = ROOT_DIR / "models" / "dashboard_eda.json"

SEGMENT_COLUMNS = (
    "product_category",
    "shipping_method",
    "coupon_used",
)
NUMERIC_COLUMNS = (
    "customer_age",
    "product_price_usd",
    "discount_percent",
    "product_rating",
    "past_purchase_count",
    "delivery_delay_days",
)


def rate_summary(data: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    grouped = (
        data.groupby(column, dropna=False, observed=True)["returned"]
        .agg(orders="size", returns="sum", return_rate="mean")
        .reset_index()
        .sort_values("return_rate", ascending=False)
    )
    return [
        {
            "name": str(row[column]),
            "orders": int(row["orders"]),
            "returns": int(row["returns"]),
            "return_rate": float(row["return_rate"]),
        }
        for _, row in grouped.iterrows()
    ]


def build_summary(data: pd.DataFrame) -> dict[str, Any]:
    required = {
        "order_date",
        "returned",
        "product_price_usd",
        *SEGMENT_COLUMNS,
        *NUMERIC_COLUMNS,
    }
    missing = required.difference(data.columns)
    if data.empty or missing:
        raise ValueError(
            "EDA source is empty or missing required columns: "
            + ", ".join(sorted(missing))
        )

    data = data.copy()
    data["order_date"] = pd.to_datetime(data["order_date"], errors="coerce")
    data["returned"] = pd.to_numeric(data["returned"], errors="coerce")
    if data["order_date"].isna().any():
        raise ValueError("EDA source contains invalid order dates.")
    if data["returned"].isna().any() or not data["returned"].isin([0, 1]).all():
        raise ValueError("EDA source target must contain only 0 and 1.")
    data["returned"] = data["returned"].astype(int)

    monthly = (
        data.assign(month=data["order_date"].dt.to_period("M").astype(str))
        .groupby("month", sort=True)["returned"]
        .agg(orders="size", returns="sum", return_rate="mean")
        .reset_index()
    )
    delay_data = data.assign(
        delivery_status=np.select(
            [
                data["delivery_delay_days"].eq(0),
                data["delivery_delay_days"].le(2),
            ],
            ["On time", "1-2 days late"],
            default="3+ days late",
        )
    )

    distributions: dict[str, list[dict[str, Any]]] = {}
    for column in NUMERIC_COLUMNS:
        values = pd.to_numeric(data[column], errors="coerce").dropna()
        bins = pd.cut(values, bins=10)
        counts = bins.value_counts(sort=False)
        distributions[column] = [
            {"range": str(interval), "orders": int(count)}
            for interval, count in counts.items()
        ]

    correlations = (
        data[list(NUMERIC_COLUMNS)]
        .corrwith(data["returned"], method="spearman")
        .dropna()
        .sort_values()
    )
    outcome_counts = data["returned"].value_counts().sort_index()
    return {
        "total_orders": int(len(data)),
        "returned_orders": int(data["returned"].sum()),
        "historical_return_rate": float(data["returned"].mean()),
        "median_product_price": float(data["product_price_usd"].median()),
        "data_quality": {
            "missing_values": int(data.isna().sum().sum()),
            "duplicate_rows": int(data.duplicated().sum()),
            "columns": int(data.shape[1]),
        },
        "outcome_counts": [
            {
                "name": "Not returned" if int(label) == 0 else "Returned",
                "orders": int(count),
            }
            for label, count in outcome_counts.items()
        ],
        "return_rate_by_product_category": rate_summary(data, "product_category"),
        "monthly_return_rates": [
            {
                "month": str(row["month"]),
                "orders": int(row["orders"]),
                "returns": int(row["returns"]),
                "return_rate": float(row["return_rate"]),
            }
            for _, row in monthly.iterrows()
        ],
        "return_rate_by_shipping_method": rate_summary(data, "shipping_method"),
        "return_rate_by_delivery_status": rate_summary(
            delay_data,
            "delivery_status",
        ),
        "return_rate_by_coupon": rate_summary(data, "coupon_used"),
        "numeric_distributions": distributions,
        "numeric_return_correlations": [
            {"name": str(name), "correlation": float(value)}
            for name, value in correlations.items()
        ],
    }


def main() -> None:
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"EDA source dataset was not found: {DATASET_PATH}")
    summary = build_summary(pd.read_csv(DATASET_PATH))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = OUTPUT_PATH.with_suffix(OUTPUT_PATH.suffix + ".tmp")
    temporary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    os.replace(temporary_path, OUTPUT_PATH)
    print(f"Saved aggregate EDA summary: {OUTPUT_PATH} ({summary['total_orders']:,} orders)")


if __name__ == "__main__":
    main()

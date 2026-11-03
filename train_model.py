from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "datasets" / "ecommerce_return_prediction_feature_engineered.csv"
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "best_return_detection_model.joblib"
VALIDATION_PATH = MODEL_DIR / "model_validation_comparison.csv"
EVALUATION_PATH = MODEL_DIR / "model_evaluation.json"

FEATURE_COLUMNS = [
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
    "purchase_frequency_group",
    "price_category",
    "order_month",
    "order_day_of_week",
    "is_weekend",
    "customer_return_history",
]


def build_preprocessor(features: pd.DataFrame, *, dense: bool = False) -> ColumnTransformer:
    numeric_columns = features.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = [
        column for column in features.columns if column not in numeric_columns
    ]
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=not dense,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        sparse_threshold=0 if dense else 0.3,
    )


def build_candidates(features: pd.DataFrame) -> list[tuple[str, Pipeline]]:
    sparse_preprocessor = build_preprocessor(features)
    candidates: list[tuple[str, Pipeline]] = []

    for regularization in (0.03, 0.1, 1.0):
        candidates.append(
            (
                f"Logistic Regression (C={regularization:g})",
                Pipeline(
                    [
                        ("preprocessor", clone(sparse_preprocessor)),
                        (
                            "model",
                            LogisticRegression(
                                C=regularization,
                                class_weight="balanced",
                                max_iter=2000,
                                solver="liblinear",
                                random_state=42,
                            ),
                        ),
                    ]
                ),
            )
        )

    candidates.append(
        (
            "Logistic Regression (accuracy-first, C=0.03)",
            Pipeline(
                [
                    ("preprocessor", clone(sparse_preprocessor)),
                    (
                        "model",
                        LogisticRegression(
                            C=0.03,
                            max_iter=2000,
                            solver="liblinear",
                            random_state=42,
                        ),
                    ),
                ]
            ),
        )
    )

    candidates.append(
        (
            "Logistic Regression L1 feature selection (C=0.03)",
            Pipeline(
                [
                    ("preprocessor", clone(sparse_preprocessor)),
                    (
                        "model",
                        LogisticRegression(
                            C=0.03,
                            class_weight="balanced",
                            l1_ratio=1.0,
                            max_iter=2000,
                            solver="liblinear",
                            random_state=42,
                        ),
                    ),
                ]
            ),
        )
    )

    for depth, leaf_size in ((5, 10), (10, 20)):
        candidates.append(
            (
                f"Decision Tree (depth={depth})",
                Pipeline(
                    [
                        ("preprocessor", clone(sparse_preprocessor)),
                        (
                            "model",
                            DecisionTreeClassifier(
                                max_depth=depth,
                                min_samples_leaf=leaf_size,
                                min_samples_split=20,
                                class_weight="balanced",
                                random_state=42,
                            ),
                        ),
                    ]
                ),
            )
        )

    candidates.append(
        (
            "Random Forest",
            Pipeline(
                [
                    ("preprocessor", clone(sparse_preprocessor)),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=160,
                            max_depth=14,
                            min_samples_leaf=5,
                            max_features=0.8,
                            class_weight="balanced_subsample",
                            n_jobs=4,
                            random_state=42,
                        ),
                    ),
                ]
            ),
        )
    )

    for depth, positive_weight in ((2, 1.0), (3, 1.0), (3, 2.5), (3, 4.48)):
        candidates.append(
            (
                f"XGBoost (depth={depth}, positive_weight={positive_weight:g})",
                Pipeline(
                    [
                        ("preprocessor", clone(sparse_preprocessor)),
                        (
                            "model",
                            XGBClassifier(
                                n_estimators=180,
                                max_depth=depth,
                                learning_rate=0.05,
                                subsample=0.85,
                                colsample_bytree=0.85,
                                min_child_weight=5,
                                reg_lambda=5,
                                scale_pos_weight=positive_weight,
                                eval_metric="logloss",
                                tree_method="hist",
                                n_jobs=4,
                                random_state=42,
                            ),
                        ),
                    ]
                ),
            )
        )

    dense_preprocessor = build_preprocessor(features, dense=True)
    for leaf_count in (7, 15):
        candidates.append(
            (
                f"Histogram Gradient Boosting (leaves={leaf_count})",
                Pipeline(
                    [
                        ("preprocessor", clone(dense_preprocessor)),
                        (
                            "model",
                            HistGradientBoostingClassifier(
                                max_iter=160,
                                max_leaf_nodes=leaf_count,
                                learning_rate=0.05,
                                min_samples_leaf=30,
                                l2_regularization=1,
                                class_weight="balanced",
                                random_state=42,
                            ),
                        ),
                    ]
                ),
            )
        )

    return candidates


def best_f1_threshold(
    labels: pd.Series,
    probabilities: np.ndarray,
) -> tuple[float, float, float, float]:
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    f1_values = (
        2 * precision[:-1] * recall[:-1]
        / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    )
    best_index = int(np.nanargmax(f1_values))
    return (
        float(thresholds[best_index]),
        float(f1_values[best_index]),
        float(precision[best_index]),
        float(recall[best_index]),
    )


def temporal_cv_summary(
    estimator: Pipeline,
    features: pd.DataFrame,
    labels: pd.Series,
) -> dict[str, float]:
    scores: list[float] = []
    auc_scores: list[float] = []
    pr_auc_scores: list[float] = []
    splitter = TimeSeriesSplit(n_splits=3)

    for train_indices, validation_indices in splitter.split(features):
        fold_model = clone(estimator)
        fold_model.fit(
            features.iloc[train_indices],
            labels.iloc[train_indices],
        )
        probabilities = fold_model.predict_proba(
            features.iloc[validation_indices]
        )[:, 1]
        threshold, fold_f1, _, _ = best_f1_threshold(
            labels.iloc[validation_indices],
            probabilities,
        )
        scores.append(fold_f1)
        auc_scores.append(
            float(
                roc_auc_score(
                    labels.iloc[validation_indices],
                    probabilities,
                )
            )
        )
        pr_auc_scores.append(
            float(
                average_precision_score(
                    labels.iloc[validation_indices],
                    probabilities,
                )
            )
        )
        print(f"  CV threshold {threshold:.4f}, F1 {fold_f1:.4f}", flush=True)

    return {
        "CV F1": float(np.mean(scores)),
        "CV F1 std": float(np.std(scores)),
        "CV ROC-AUC": float(np.mean(auc_scores)),
        "CV PR-AUC": float(np.mean(pr_auc_scores)),
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(json_safe(payload), indent=2),
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Training dataset was not found: {DATA_PATH}")
    frame = pd.read_csv(DATA_PATH)
    if frame.empty:
        raise ValueError("The training dataset is empty.")
    required = set(FEATURE_COLUMNS) | {"order_date", "returned", "return_rate_pct"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "The training dataset is missing required columns: "
            + ", ".join(sorted(missing))
        )
    if frame["returned"].isna().any() or not frame["returned"].isin([0, 1]).all():
        raise ValueError("The returned target must contain only 0 and 1.")

    frame["order_date"] = pd.to_datetime(frame["order_date"], errors="coerce")
    if frame["order_date"].isna().any():
        raise ValueError("The training dataset contains invalid order dates.")
    frame = frame.drop_duplicates().sort_values("order_date").reset_index(drop=True)

    train_end = frame["order_date"].quantile(0.75)
    validation_end = frame["order_date"].quantile(0.875)
    train_mask = frame["order_date"] < train_end
    validation_mask = (
        (frame["order_date"] >= train_end)
        & (frame["order_date"] < validation_end)
    )
    test_mask = frame["order_date"] >= validation_end
    if not train_mask.any() or not validation_mask.any() or not test_mask.any():
        raise ValueError("Could not create non-empty chronological data partitions.")

    X = frame.loc[:, FEATURE_COLUMNS]
    y = frame["returned"].astype(int)
    X_train = X.loc[train_mask].reset_index(drop=True)
    y_train = y.loc[train_mask].reset_index(drop=True)
    X_validation = X.loc[validation_mask].reset_index(drop=True)
    y_validation = y.loc[validation_mask].reset_index(drop=True)
    X_test = X.loc[test_mask].reset_index(drop=True)
    y_test = y.loc[test_mask].reset_index(drop=True)

    print(
        f"Chronological split: train={len(X_train):,}, "
        f"validation={len(X_validation):,}, test={len(X_test):,}",
        flush=True,
    )
    print(
        f"Dates: train through {train_end.date()}, "
        f"validation before {validation_end.date()}, "
        f"test from {validation_end.date()}",
        flush=True,
    )

    validation_results: list[dict[str, Any]] = []
    candidate_models: dict[str, Any] = {}
    for name, candidate in build_candidates(X_train):
        print(f"Fit {name}", flush=True)
        candidate.fit(X_train, y_train)
        probabilities = candidate.predict_proba(X_validation)[:, 1]
        threshold, f1, precision, recall = best_f1_threshold(
            y_validation,
            probabilities,
        )
        result = {
            "Model": name,
            "Validation threshold": threshold,
            "Validation precision": precision,
            "Validation recall": recall,
            "Validation F1": f1,
            "Validation ROC-AUC": float(
                roc_auc_score(y_validation, probabilities)
            ),
            "Validation PR-AUC": float(
                average_precision_score(y_validation, probabilities)
            ),
            "Validation Brier": float(
                brier_score_loss(y_validation, probabilities)
            ),
        }
        validation_results.append(result)
        candidate_models[name] = candidate
        print(
            f"  F1={f1:.4f}, precision={precision:.4f}, "
            f"recall={recall:.4f}, PR-AUC={result['Validation PR-AUC']:.4f}",
            flush=True,
        )

    logistic_name = "Logistic Regression (C=0.03)"
    calibrated_name = "Sigmoid-calibrated Logistic Regression (C=0.03)"
    calibrated_model = CalibratedClassifierCV(
        estimator=clone(candidate_models[logistic_name]),
        method="sigmoid",
        cv=TimeSeriesSplit(n_splits=3),
    )
    print(f"Fit {calibrated_name}", flush=True)
    calibrated_model.fit(X_train, y_train)
    calibrated_probabilities = calibrated_model.predict_proba(X_validation)[:, 1]
    calibrated_threshold, calibrated_f1, calibrated_precision, calibrated_recall = (
        best_f1_threshold(y_validation, calibrated_probabilities)
    )
    calibrated_result = {
        "Model": calibrated_name,
        "Validation threshold": calibrated_threshold,
        "Validation precision": calibrated_precision,
        "Validation recall": calibrated_recall,
        "Validation F1": calibrated_f1,
        "Validation ROC-AUC": float(
            roc_auc_score(y_validation, calibrated_probabilities)
        ),
        "Validation PR-AUC": float(
            average_precision_score(y_validation, calibrated_probabilities)
        ),
        "Validation Brier": float(
            brier_score_loss(y_validation, calibrated_probabilities)
        ),
    }
    validation_results.append(calibrated_result)
    candidate_models[calibrated_name] = calibrated_model
    print(
        f"  F1={calibrated_f1:.4f}, PR-AUC={calibrated_result['Validation PR-AUC']:.4f}, "
        f"Brier={calibrated_result['Validation Brier']:.4f}",
        flush=True,
    )

    accuracy_first_name = "Logistic Regression (accuracy-first, C=0.03)"
    accuracy_candidates = [accuracy_first_name, logistic_name]
    stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    stratified_scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "recall_returned": "recall",
    }
    stratified_cv_results: dict[str, dict[str, float]] = {}
    for name in accuracy_candidates:
        print(f"5-fold StratifiedKFold: {name}", flush=True)
        fold_scores = cross_validate(
            clone(candidate_models[name]),
            X_train,
            y_train,
            cv=stratified_cv,
            scoring=stratified_scoring,
            n_jobs=1,
        )
        stratified_cv_results[name] = {
            "accuracy_mean": float(fold_scores["test_accuracy"].mean()),
            "accuracy_std": float(fold_scores["test_accuracy"].std()),
            "balanced_accuracy_mean": float(
                fold_scores["test_balanced_accuracy"].mean()
            ),
            "returned_recall_mean": float(
                fold_scores["test_recall_returned"].mean()
            ),
        }
        matching_row = next(row for row in validation_results if row["Model"] == name)
        matching_row.update(
            {
                "Stratified CV accuracy": stratified_cv_results[name]["accuracy_mean"],
                "Stratified CV accuracy std": stratified_cv_results[name][
                    "accuracy_std"
                ],
                "Stratified CV balanced accuracy": stratified_cv_results[name][
                    "balanced_accuracy_mean"
                ],
                "Stratified CV returned recall": stratified_cv_results[name][
                    "returned_recall_mean"
                ],
            }
        )
        print(
            f"  accuracy={stratified_cv_results[name]['accuracy_mean']:.4f}, "
            f"balanced_accuracy={stratified_cv_results[name]['balanced_accuracy_mean']:.4f}, "
            f"returned_recall={stratified_cv_results[name]['returned_recall_mean']:.4f}",
            flush=True,
        )

    baseline_cv_scores = cross_validate(
        DummyClassifier(strategy="most_frequent"),
        X_train,
        y_train,
        cv=stratified_cv,
        scoring=stratified_scoring,
        n_jobs=1,
    )
    baseline_cv = {
        "accuracy_mean": float(baseline_cv_scores["test_accuracy"].mean()),
        "accuracy_std": float(baseline_cv_scores["test_accuracy"].std()),
        "balanced_accuracy_mean": float(
            baseline_cv_scores["test_balanced_accuracy"].mean()
        ),
        "returned_recall_mean": float(
            baseline_cv_scores["test_recall_returned"].mean()
        ),
    }

    selected_name = max(
        accuracy_candidates,
        key=lambda name: stratified_cv_results[name]["accuracy_mean"],
    )
    selected_validation = next(
        row for row in validation_results if row["Model"] == selected_name
    )
    selected_model = candidate_models[selected_name]
    threshold = 0.5
    validation_probabilities = selected_model.predict_proba(X_validation)[:, 1]
    validation_predictions = (validation_probabilities >= threshold).astype(int)
    selected_validation.update(
        {
            "Validation threshold": threshold,
            "Validation accuracy": float(
                accuracy_score(y_validation, validation_predictions)
            ),
            "Validation precision": float(
                precision_score(y_validation, validation_predictions, zero_division=0)
            ),
            "Validation recall": float(
                recall_score(y_validation, validation_predictions, zero_division=0)
            ),
            "Validation F1": float(
                f1_score(y_validation, validation_predictions, zero_division=0)
            ),
        }
    )
    test_probabilities = selected_model.predict_proba(X_test)[:, 1]
    test_predictions = (test_probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_test,
        test_predictions,
        labels=[0, 1],
    ).ravel()
    report = classification_report(
        y_test,
        test_predictions,
        labels=[0, 1],
        target_names=["Not returned", "Returned"],
        output_dict=True,
        zero_division=0,
    )
    evaluation = {
        "model_name": selected_name,
        "threshold": threshold,
        "threshold_selection": "Default probability threshold of 0.5; not tuned against the holdout",
        "model_selection": "Highest mean accuracy among unweighted and class-weighted Logistic Regression using 5-fold StratifiedKFold on the training partition",
        "cv_metrics": stratified_cv_results[selected_name],
        "majority_baseline_cv_metrics": baseline_cv,
        "validation_metrics": {
            key: selected_validation[key]
            for key in selected_validation
            if key.startswith("Validation ")
        },
        "test_period_start": str(frame.loc[test_mask, "order_date"].min().date()),
        "test_period_end": str(frame.loc[test_mask, "order_date"].max().date()),
        "test_rows": int(len(y_test)),
        "test_return_rate": float(y_test.mean()),
        "test_metrics": {
            "accuracy": float(accuracy_score(y_test, test_predictions)),
            "precision": float(
                precision_score(y_test, test_predictions, zero_division=0)
            ),
            "recall": float(recall_score(y_test, test_predictions, zero_division=0)),
            "f1": float(f1_score(y_test, test_predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, test_probabilities)),
            "pr_auc": float(average_precision_score(y_test, test_probabilities)),
            "brier_score": float(brier_score_loss(y_test, test_probabilities)),
        },
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "classification_report": report,
        "calibration_experiment": {
            "method": "Sigmoid calibration with expanding-window cross-validation",
            "uncalibrated_validation_brier": next(
                row["Validation Brier"]
                for row in validation_results
                if row["Model"] == logistic_name
            ),
            "calibrated_validation_brier": calibrated_result["Validation Brier"],
            "uncalibrated_validation_f1": next(
                row["Validation F1"]
                for row in validation_results
                if row["Model"] == logistic_name
            ),
            "calibrated_validation_f1": calibrated_result["Validation F1"],
            "decision": "Keep the uncalibrated score to prioritize F1; label it as a model score, not a calibrated probability.",
        },
        "test_note": "The latest chronological partition is reserved for final evaluation and is not used for fitting, candidate selection, cross-validation, or threshold tuning.",
    }
    artifact = {
        "model": selected_model,
        "model_name": selected_name,
        "threshold": threshold,
        "threshold_selection": evaluation["threshold_selection"],
        "feature_columns": FEATURE_COLUMNS,
        "training_period_end": str(
            frame.loc[train_mask, "order_date"].max().date()
        ),
    }

    MODEL_DIR.mkdir(exist_ok=True)
    temporary_model_path = MODEL_PATH.with_suffix(MODEL_PATH.suffix + ".tmp")
    joblib.dump(artifact, temporary_model_path)
    os.replace(temporary_model_path, MODEL_PATH)
    validation_results.sort(
        key=lambda row: row.get("Stratified CV accuracy", -1.0),
        reverse=True,
    )
    pd.DataFrame(validation_results).to_csv(VALIDATION_PATH, index=False)
    write_json(EVALUATION_PATH, evaluation)

    print(f"\nSelected model: {selected_name}")
    print(f"Decision threshold: {threshold:.4f}")
    print("Final chronological test metrics:")
    print(json.dumps(evaluation["test_metrics"], indent=2))


if __name__ == "__main__":
    main()

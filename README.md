# E-commerce Product Return Prediction

## Project overview

This project explores whether an e-commerce order is likely to be returned. It uses product, customer history, order, and delivery information to estimate return risk and help teams decide which orders may need closer review.

The project includes data understanding, exploratory analysis, preprocessing, feature engineering, classification, model evaluation, and a Streamlit dashboard for individual predictions and return-pattern analysis.

## Abstract

Returns can create operational costs and affect customer experience. This project analyzes patterns associated with product returns and trains machine-learning classifiers to estimate return probability. It evaluates model performance with accuracy, precision, recall, F1-score, and ROC-AUC, and includes feature-importance information to help explain predictions. Because returns are the minority class, the project emphasizes the trade-off between finding more returns and generating more manual reviews.

## Project objectives

- Analyze customer, product, order, discount, and delivery patterns related to returns.
- Prepare and engineer features for classification.
- Train and compare return-prediction models.
- Evaluate results with business-relevant metrics and class imbalance in mind.
- Score an individual order in the dashboard.
- Explore return patterns with interactive charts and data-backed summaries.
- Review held-out model performance, threshold behavior, and prediction explanations.

## Project modules

1. Order data collection
2. Data cleaning
3. Customer behavior analysis
4. Product return analysis
5. Feature preparation
6. Return prediction model
7. Model testing
8. Return-risk prediction
9. Customer and product insights
10. Results dashboard

## Dataset and features

The feature-engineered dataset contains 60,000 orders and the target column `returned`, where `1` means returned and `0` means not returned. In this dataset, 10,961 orders are returned, for a historical return rate of approximately 18.3%.

The model uses available order and customer-history inputs such as:

- Customer age, country, device, and purchase history
- Product category, price, rating, and discount
- Coupon use and payment method
- Shipping method, delivery time, and delay
- Session length and product views
- Engineered purchase-frequency, price, calendar, and return-history groups

Identifiers or outcome-derived values are not intended as prediction inputs. The target `returned` and overall `return_rate_pct` are excluded from the model features. The raw `order_date` is replaced by calendar features.

## Models and evaluation

The dashboard deploys **Logistic Regression (accuracy-first, C=0.03)**. It was chosen by mean accuracy in 5-fold `StratifiedKFold` run only on the training period. The default 0.5 threshold was not tuned on the chronological holdout.

| Final test metric | Result |
|---|---:|
| Accuracy | 81.5% |

This accuracy is higher than the previous deployed model's 56.3%, but it is **not a better return detector**. The dataset is imbalanced: the chronological test holdout contains 18.6% returns. The new model found only **15 of 1,400** returned orders (1.1% recall) and missed **1,385**. Its test accuracy (81.48%) is only marginally above the majority-class baseline (81.45%), which predicts every order as not returned. The Streamlit evaluation page shows only accuracy as requested, with a warning that it misses most returns; supporting metrics are retained in the evaluation artifact and notebook.

The final test period is **2024-09-30 through 2024-12-30** (7,545 orders). The 5-fold stratified cross-validation was applied to the earlier training partition only. Its mean accuracy was **81.83%** for the selected unweighted Logistic Regression and **81.79%** for the majority-class baseline; returned-class recall in those folds was only **0.97%**. The accuracy-first behavior is therefore real, but not operationally useful for finding returns.

The dataset contains 60,000 synthetic orders. To improve actual return detection rather than the accuracy display, collect more representative labeled data and point-in-time customer/order features; evaluate with class-aware metrics and a future-period holdout. The model is not production-ready.

## Streamlit dashboard

The dashboard provides:

- **Overview:** dataset-level return rate, evaluation highlights, and aggregate return rates by product category.
- **Predict an order:** enter customer, product, order, and delivery details to get an estimated score and review recommendation.
- **Exploratory analysis:** inspect return distributions, category and delivery patterns, monthly trends, coupon use, and numeric associations.
- **Model evaluation:** see the deployed model's held-out accuracy only. Accuracy can be misleading on an imbalanced target; the executed `notebooks/06_model_evaluation.ipynb` contains the data-quality audit, class-imbalance baseline, feature engineering, stratified cross-validation, and supporting evaluation metrics.

Individual Logistic Regression predictions include per-order feature contributions. A review flag is a recommendation for closer inspection, not a confirmed return.

## Notebooks and application

```text
.
├── app.py
├── train_model.py
├── requirements.txt
├── datasets/
│   └── ecommerce_return_prediction_feature_engineered.csv
├── models/
│   ├── best_return_detection_model.joblib
│   ├── model_evaluation.json
│   ├── model_validation_comparison.csv
│   ├── dashboard_insights.json
│   └── two_model_comparison_metrics.csv
└── notebooks/
    ├── 01_data_understanding.ipynb
    ├── 02_eda.ipynb
    ├── 03_data_preprocessing.ipynb
    ├── 04_data_engineering.ipynb
    ├── 05_model_training.ipynb
    └── 06_model_evaluation.ipynb
```

The notebooks document data understanding, EDA, preprocessing, engineering, model training, and evaluation. `train_model.py` trains and compares the candidate models, runs expanding-window cross-validation, selects a validation threshold, and evaluates the chosen model once on the final chronological holdout. The app loads those saved artifacts.

## Run the dashboard locally

Install the packages listed in `requirements.txt`, then run:

```powershell
python train_model.py
python -m streamlit run app.py
```

The dashboard uses scikit-learn, pandas, NumPy, and Streamlit. XGBoost is also used by the model-training script. Keep the engineered dataset in `datasets/` when retraining.

## Keywords

E-commerce, Return Prediction, Customer Behavior, Product Analysis, Classification, XGBoost, Machine Learning, Customer Analytics.

# E-commerce Product Return Prediction

## Project overview

This project explores whether an e-commerce order is likely to be returned. It uses product, customer history, order, and delivery information to estimate return risk and help teams decide which orders may need closer review.

The project includes data understanding, exploratory analysis, preprocessing, feature engineering, classification, model evaluation, and a Streamlit dashboard for individual and batch predictions.

## Abstract

Returns can create operational costs and affect customer experience. This project analyzes patterns associated with product returns and trains machine-learning classifiers to estimate return probability. It evaluates model performance with accuracy, precision, recall, F1-score, and ROC-AUC, and includes feature-importance information to help explain predictions. Because returns are the minority class, the project emphasizes the trade-off between finding more returns and generating more manual reviews.

## Project objectives

- Analyze customer, product, order, discount, and delivery patterns related to returns.
- Prepare and engineer features for classification.
- Train and compare return-prediction models.
- Evaluate results with business-relevant metrics and class imbalance in mind.
- Score individual orders and batches of orders in a dashboard.
- Surface influential model features and aggregate product and shipping insights.

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

The saved dashboard model is **XGBoost**. Its classification threshold is approximately **0.1549**, selected on a validation split to maximize F1. The dashboard also compares it with a Random Forest model.

| Model and decision threshold | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| XGBoost, default threshold 0.50 | 81.7% | 50.8% | 1.4% | 0.027 | 0.630 |
| XGBoost, validation-tuned threshold 0.1549 | 50.8% | 22.9% | 71.8% | 0.347 | 0.630 |
| Random Forest, default threshold 0.50 | 81.8% | 75.0% | 0.1% | 0.003 | 0.617 |

The majority-class accuracy baseline is approximately **81.7%**. Therefore, accuracy at the default threshold is misleading: the models identify very few returned orders at that cutoff. The validation-tuned threshold finds more returns but also flags many orders that are not returned. The threshold should be chosen according to the operational cost of missed returns versus manual review.

These are results for this dataset and holdout split; they do not guarantee the same performance on future or different orders.

## Streamlit dashboard

The dashboard provides:

- **Overview:** dataset-level return rate, evaluation highlights, and aggregate return rates by product category.
- **Predict an order:** enter customer, product, order, and delivery details to get an estimated score and review recommendation.
- **Score a file:** upload a CSV, score each row, and download the results.
- **Insights & model:** compare evaluation metrics and view model feature importance and aggregate return patterns.

The estimated return score is a probability-like model output, not a guarantee. A review flag is a recommendation for closer inspection, not a confirmed return.

### Batch CSV input

Use the template available in the dashboard. Each row represents one order and needs these columns:

```text
order_date,customer_age,country,device_type,product_category,product_price_usd,discount_percent,coupon_used,product_rating,past_purchase_count,past_return_rate_pct,session_length_minutes,num_product_views,shipping_method,delivery_time_days,delivery_delay_days,payment_method,total_orders_lifetime,total_returns_lifetime
```

The app derives calendar, purchase-frequency, price, and customer-return-history features from these inputs. The `order_date` should be a recognizable date such as `2025-04-18`. A `returned` target value is not required for new predictions.

The scored output adds:

- `estimated_return_score`: estimated score from 0 to 1.
- `return_risk`: a readable risk or review label.
- `flag_for_review`: `1` when the selected threshold is met, otherwise `0`.

## Notebooks and application

```text
.
├── app.py
├── requirements.txt
├── datasets/
│   └── ecommerce_return_prediction_feature_engineered.csv
├── models/
│   ├── best_return_detection_model.joblib
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

The notebooks document data understanding, EDA, preprocessing, engineering, model training, and evaluation. The Streamlit app loads the saved model, comparison metrics, and aggregate dashboard insights.

## Run the dashboard locally

Install the packages listed in `requirements.txt`, then run:

```powershell
python -m streamlit run app.py
```

The dashboard uses XGBoost, scikit-learn, pandas, NumPy, and Streamlit. The model and metrics are read from the `models/` folder.

## Keywords

E-commerce, Return Prediction, Customer Behavior, Product Analysis, Classification, XGBoost, Machine Learning, Customer Analytics.

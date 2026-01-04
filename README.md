Fraud Detection System – End-to-End Machine Learning Project

This repository contains an end-to-end fraud detection system designed to simulate how real-world financial fraud detection pipelines work in production environments.

The project covers the complete lifecycle of a Machine Learning solution — data generation, feature engineering, model training, evaluation, business cost analysis, and real-time deployment using FastAPI.

This project is built with a strong focus on practical ML engineering, not just model accuracy.

Key Highlights

Complete ML pipeline (data → model → API)

Handles highly imbalanced fraud data

Advanced, behavior-based feature engineering

Multiple ML models trained and compared

Business-driven evaluation (cost of fraud vs false alarms)

Production-style REST API for real-time predictions

Clean, modular, and readable codebase

Problem Statement

Fraud detection systems must:

Detect fraudulent transactions early

Minimize false positives (blocking genuine users)

Handle extreme class imbalance

Work in real-time with low latency

This project demonstrates how these challenges can be addressed using Machine Learning and modern backend tools.

Machine Learning Approach
Data

Synthetic transaction data generated to simulate real payment behavior

Includes customer activity, merchant behavior, device usage, location, and time

Feature Engineering

The model uses engineered features inspired by real fraud systems:

Transaction velocity (last 5 minutes, 1 hour, 24 hours)

Distance from previous transaction (Haversine distance)

Device change detection

Customer spending pattern comparison

Merchant fraud history

Time-based features (hour of day, day of week)

These features help capture behavioral anomalies, not just transaction amount.

Models Trained

XGBoost (final production model)

Random Forest

Logistic Regression

XGBoost is selected as the final model due to:

Better performance on imbalanced data

Strong recall for fraud cases

Robust probability estimates

Class imbalance is handled using scale_pos_weight.

Model Evaluation Metrics

Models are evaluated using:

Precision

Recall

F1-Score

ROC-AUC

PR-AUC

In addition, a cost-based evaluation is included to reflect real business impact:

False Negatives (missed fraud) → high cost

False Positives (incorrect blocks) → lower cost

Project Structure
fraud-detection/
│
├── fraud_detection.py        # Training + feature engineering + API
├── requirements.txt          # Project dependencies
├── fraud_xgb_model.joblib    # Trained XGBoost model
├── scaler.joblib             # Feature scaler
├── ohe.joblib                # One-hot encoder
├── merchant_freq.joblib      # Merchant frequency mapping
└── README.md

Installation

Clone the repository and install dependencies:

git clone https://github.com/Mahith-data/fraud-detection.git
cd fraud-detection
pip install -r requirements.txt

Model Training

Training must be run once before starting the API:

python fraud_detection.py --train


This step:

Generates synthetic data

Trains all models

Saves trained artifacts for production use

Running the API

After training:

uvicorn fraud_detection:app --reload


The API will be available at:

http://127.0.0.1:8000

API Testing (Swagger UI)

Open the browser and go to:

http://127.0.0.1:8000/docs

Sample Request
{
  "transaction_id": "T10001",
  "transaction_amount": 15000,
  "transaction_time": "2026-01-04T10:30:00",
  "customer_id": "C000123",
  "merchant_id": "M00045",
  "latitude": 28.61,
  "longitude": 77.23,
  "device_type": "android"
}

Sample Response
{
  "transaction_id": "T10001",
  "fraud_probability": 0.84,
  "decision": "FLAGGED"
}

Technical Stack

Python

NumPy, Pandas

Scikit-learn

XGBoost

FastAPI

Uvicorn

Joblib

Real-World Relevance

This project reflects how fraud detection systems are designed in real companies:

Feature engineering based on user behavior

Imbalance-aware model training

Business cost considerations

Real-time scoring via REST APIs

Model artifact management for deployment

Future Enhancements

Integration with real transaction datasets

Dockerization for production deployment

CI/CD pipeline with automated testing

Threshold tuning based on business risk

Model monitoring and data drift detection

Author

Mahith K
Computer Science (Data Science)
Interested in Machine Learning, Data Engineering, and Applied AI


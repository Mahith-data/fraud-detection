# Fraud Detection System 

This project implements an end-to-end fraud detection system that mirrors how real-world financial fraud detection pipelines are built and deployed. It covers the complete Machine Learning lifecycle, from data generation and feature engineering to model training, evaluation, and real-time prediction using a REST API.

The focus of this project is not just model accuracy, but practical ML engineering, business relevance, and deployability.



## Project Overview

Fraud detection systems must identify fraudulent transactions accurately while minimizing false alerts and operating in real time. This project demonstrates how these challenges can be handled using Machine Learning models combined with a production-style API.

Key objectives:
- Handle highly imbalanced fraud data
- Capture behavioral anomalies instead of relying only on transaction amount
- Deploy a trained model for real-time inference



## Machine Learning Pipeline

### Data
- Synthetic transaction data generated to simulate real payment behavior
- Includes customer, merchant, device, time, and location information

### Feature Engineering
Features are designed based on real-world fraud detection practices:
- Transaction velocity (last 5 minutes, 1 hour, 24 hours)
- Distance from previous transaction using Haversine distance
- Device change detection
- Customer spending behavior comparison
- Merchant-level fraud risk
- Time-based features (hour of day, day of week)

These features help detect unusual behavior patterns commonly associated with fraud.



## Models Used

- XGBoost (final production model)
- Random Forest
- Logistic Regression

XGBoost is selected as the final model due to its strong performance on imbalanced datasets and better recall for fraudulent transactions. Class imbalance is handled using appropriate weighting.



## Model Evaluation

Models are evaluated using:
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC

A cost-based evaluation is also included to reflect real business impact:
- False negatives (missed fraud) have a high cost
- False positives (incorrectly flagged transactions) have a lower cost


## Project Structure

fraud-detection/
│
├── fraud_detection.py # Data generation, training, and API

├── requirements.txt # Project dependencies

├── fraud_xgb_model.joblib # Trained XGBoost model

├── scaler.joblib # Feature scaler

├── ohe.joblib # One-hot encoder

├── merchant_freq.joblib # Merchant frequency mapping

└── README.md



## Installation

```bash
git clone https://github.com/Mahith-data/fraud-detection.git
cd fraud-detection
pip install -r requirements.txt
Training the Model
bash

python fraud_detection.py --train
This step generates data, trains the models, and saves all required artifacts for inference.

Running the API
bash

uvicorn fraud_detection:app --reload
API will be available at:


http://127.0.0.1:8000
API Usage Example
Endpoint: POST /score

Request:

json

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
Response:

json

{
  "transaction_id": "T10001",
  "fraud_probability": 0.84,
  "decision": "FLAGGED"
}
Tech Stack
Python

NumPy, Pandas

Scikit-learn

XGBoost

FastAPI

Uvicorn

Joblib


Author
Mahith K
Computer Science (Data Science)
Interested in Machine Learning and Data Engineering

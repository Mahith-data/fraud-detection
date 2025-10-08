# 🕵️‍♂️ Fraud Detection System (with FastAPI)

This project builds an **end-to-end fraud detection system** that generates **synthetic transaction data**, performs **feature engineering**, trains **machine learning models (XGBoost, RandomForest, LogisticRegression)**, and exposes a **real-time fraud scoring API** using **FastAPI**.

---

## 🚀 Features

- ✅ Synthetic transaction dataset generator  
- ✅ Advanced feature engineering (time-based, behavioral, and risk features)  
- ✅ Model training with XGBoost, RandomForest, and Logistic Regression  
- ✅ Automatic scaling, encoding, and model artifact saving  
- ✅ Business cost analysis (False Positive / False Negative cost)  
- ✅ Real-time fraud scoring API endpoint (`/score`)  

---

## 🧠 Tech Stack

- **Python 3.9+**
- **FastAPI**
- **scikit-learn**
- **XGBoost**
- **Joblib**
- **Pandas / NumPy**

---

## 📦 Installation

```bash
# 1️⃣ Clone this repository
git clone https://github.com/your-username/fraud-detection-system.git
cd fraud-detection-system

# 2️⃣ Create a virtual environment
python -m venv venv
source venv/bin/activate      # (Linux/Mac)
venv\Scripts\activate         # (Windows)

# 3️⃣ Install dependencies
pip install -r requirements.txt
⚙️ Training the Model

To generate a synthetic dataset and train all models:

python fraud_detection.py --train


This will:

Generate synthetic transaction data

Perform preprocessing and feature engineering

Train XGBoost, RandomForest, and Logistic Regression models

Save artifacts:

fraud_xgb_model.joblib
scaler.joblib
ohe.joblib
merchant_freq.joblib

📊 Example Output
[INFO] Synthetic dataset: 100000 rows, frauds=400
[INFO] Train frauds=400, nonfrauds=99600, scale_pos_weight=249.00
XGBoost metrics: {'precision': 0.72, 'recall': 0.68, 'f1': 0.70, 'roc_auc': 0.92, 'pr_auc': 0.51}
RandomForest metrics: {'precision': 0.68, 'recall': 0.65, 'f1': 0.66, 'roc_auc': 0.89, 'pr_auc': 0.46}
LogisticRegression metrics: {'precision': 0.61, 'recall': 0.60, 'f1': 0.60, 'roc_auc': 0.85, 'pr_auc': 0.41}

🧮 Business Cost Example
Confusion matrix: TN=19850, FP=50, FN=20, TP=80
Total business cost: 20300 (cost_fn=1000, cost_fp=10)

🌐 Running the API

After training, start the FastAPI server:

uvicorn fraud_detection:app --reload


API will be available at:
👉 http://127.0.0.1:8000/docs

🧾 Example API Request

POST /score

Request:

{
  "transaction_id": "T12345",
  "transaction_amount": 250.0,
  "transaction_time": "2025-10-07T12:34:56",
  "customer_id": "C000001",
  "merchant_id": "M00001",
  "latitude": 28.6,
  "longitude": 77.1,
  "device_type": "android"
}


Response:

{
  "transaction_id": "T12345",
  "fraud_probability": 0.872345,
  "decision": "FLAGGED"
}

🧩 Project Structure
fraud_detection.py         # Main script (data gen + training + API)
fraud_xgb_model.joblib     # Trained XGBoost model
scaler.joblib              # Feature scaler
ohe.joblib                 # One-hot encoder for device_type
merchant_freq.joblib       # Merchant frequency map
requirements.txt           # Dependencies
README.md                  # Project documentation

🧑‍💻 Author

KARRI NAGA MAHITH KUMAR
📧 k.mahith2006@gmail.com



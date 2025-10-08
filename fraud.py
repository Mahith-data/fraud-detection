from __future__ import annotations
import argparse
import logging
import math
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# ---------- Configuration ----------
MODEL_PATH = "fraud_xgb_model.joblib"
SCALER_PATH = "scaler.joblib"
ENCODER_PATH = "ohe.joblib"
MERCHANT_FREQ_PATH = "merchant_freq.joblib"
RANDOM_STATE = 42
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

app = FastAPI(title="Fraud Detection API", version="1.0")

# ---------- Utilities ----------


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------- Synthetic Data Generator ----------
def generate_synthetic_transactions(n_customers=5000, n_transactions=200000, fraud_rate=0.005) -> pd.DataFrame:
    np.random.seed(RANDOM_STATE)
    customers = [f"C{str(i).zfill(6)}" for i in range(n_customers)]
    merchants = [f"M{str(i).zfill(5)}" for i in range(500)]
    base_time = datetime.utcnow()
    records = []
    for t in range(n_transactions):
        customer = np.random.choice(customers)
        merchant = np.random.choice(merchants)
        amount = max(0.5, np.random.exponential(scale=50.0))
        lat = 28.5 + np.random.normal(scale=0.5)
        lon = 77.0 + np.random.normal(scale=0.5)
        device = np.random.choice(["android", "ios", "web"])
        txn_time = base_time - timedelta(seconds=int(np.random.exponential(scale=60 * 60 * 24)))
        is_fraud = 0
        if np.random.rand() < fraud_rate:
            is_fraud = 1
            amount *= np.random.uniform(5, 50)
            lat += np.random.uniform(2.0, 10.0)
            lon += np.random.uniform(2.0, 10.0)
            device = np.random.choice(["android", "ios"])
        records.append(
            {
                "transaction_id": f"T{t}",
                "transaction_amount": round(amount, 2),
                "transaction_time": txn_time,
                "customer_id": customer,
                "merchant_id": merchant,
                "latitude": lat,
                "longitude": lon,
                "device_type": device,
                "is_fraud": is_fraud,
            }
        )
    df = pd.DataFrame.from_records(records)
    df = df.sort_values("transaction_time").reset_index(drop=True)
    logging.info(f"Synthetic dataset: {df.shape[0]} rows, frauds={df['is_fraud'].sum()}")
    return df


# ---------- Preprocessing & Feature Engineering ----------
def preprocess_basic(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates(subset=["transaction_id"])
    df = df.dropna(subset=["transaction_amount", "transaction_time", "customer_id"])
    df["transaction_time"] = pd.to_datetime(df["transaction_time"])
    df["transaction_amount"] = pd.to_numeric(df["transaction_amount"], errors="coerce").fillna(0.0)
    upper = df["transaction_amount"].quantile(0.995)
    df["transaction_amount_wins"] = df["transaction_amount"].clip(upper=upper)
    return df


def feature_engineering(df: pd.DataFrame, merchant_freq: Optional[dict] = None) -> Tuple[pd.DataFrame, list]:
    df = df.copy()
    df = df.sort_values(["customer_id", "transaction_time"]).reset_index(drop=True)
    df["hour_of_day"] = df["transaction_time"].dt.hour
    df["day_of_week"] = df["transaction_time"].dt.weekday
    df["txns_5m"] = 0
    df["txns_1h"] = 0
    df["txns_24h"] = 0
    group = df.groupby("customer_id")
    for cust, sub in group:
        times = sub["transaction_time"].values
        idxs = sub.index.values
        n = len(times)
        left_5m = left_1h = left_24h = 0
        for i in range(n):
            t = times[i]
            while times[left_5m] < t - np.timedelta64(5, "m"):
                left_5m += 1
            while times[left_1h] < t - np.timedelta64(1, "h"):
                left_1h += 1
            while times[left_24h] < t - np.timedelta64(24, "h"):
                left_24h += 1
            df.at[idxs[i], "txns_5m"] = i - left_5m
            df.at[idxs[i], "txns_1h"] = i - left_1h
            df.at[idxs[i], "txns_24h"] = i - left_24h
    df["prev_lat"] = df.groupby("customer_id")["latitude"].shift(1)
    df["prev_lon"] = df.groupby("customer_id")["longitude"].shift(1)
    df["dist_prev_km"] = df.apply(
        lambda r: haversine_distance(r["latitude"], r["longitude"], r["prev_lat"], r["prev_lon"])
        if not pd.isnull(r["prev_lat"]) else 0.0, axis=1
    )
    df["prev_device"] = df.groupby("customer_id")["device_type"].shift(1)
    df["device_changed"] = (df["device_type"] != df["prev_device"]).astype(int).fillna(0)
    df["cust_avg_amount_24h"] = group["transaction_amount"].transform(
        lambda x: x.rolling(window=10, min_periods=1).mean()
    )
    df["amount_over_avg"] = df["transaction_amount"] / (df["cust_avg_amount_24h"] + 1e-6)
    if "is_fraud" in df.columns:
        df["merchant_fraud_rate"] = df.groupby("merchant_id")["is_fraud"].transform("mean").fillna(0.0)
    else:
        if merchant_freq is None:
            merchant_freq = {}
        df["merchant_fraud_rate"] = df["merchant_id"].map(merchant_freq).fillna(0.0)

    features = [
        "transaction_amount_wins",
        "hour_of_day",
        "day_of_week",
        "txns_5m",
        "txns_1h",
        "txns_24h",
        "dist_prev_km",
        "device_changed",
        "amount_over_avg",
        "merchant_fraud_rate",
        "device_type",
        "merchant_id",
    ]
    return df, features


# ---------- Modeling ----------
def train_and_evaluate(df: pd.DataFrame, features: list, label_col: str = "is_fraud") -> Dict[str, Any]:
    # Prepare X, y
    X = df[features].copy()
    y = df[label_col].values

    # Time-aware split would be ideal; using stratified shuffle for demo (keeps imbalance distribution)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(sss.split(X, y))
    X_train, X_test = X.iloc[train_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True)
    y_train, y_test = y[train_idx], y[test_idx]

    # Frequency encode merchant_id using training set distribution
    merchant_freq = X_train["merchant_id"].value_counts(normalize=True).to_dict()
    X_train["merchant_freq"] = X_train["merchant_id"].map(merchant_freq).fillna(0.0)
    X_test["merchant_freq"] = X_test["merchant_id"].map(merchant_freq).fillna(0.0)

    # One-hot encode device_type
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    dev_train = ohe.fit_transform(X_train[["device_type"]])
    dev_test = ohe.transform(X_test[["device_type"]])
    dev_cols = [f"device_{c}" for c in ohe.categories_[0]]
    dev_train_df = pd.DataFrame(dev_train, columns=dev_cols, index=X_train.index)
    dev_test_df = pd.DataFrame(dev_test, columns=dev_cols, index=X_test.index)

    # Drop original categorical cols and concat encoded
    X_train = pd.concat([X_train.reset_index(drop=True).drop(columns=["device_type", "merchant_id"]), dev_train_df.reset_index(drop=True)], axis=1)
    X_test = pd.concat([X_test.reset_index(drop=True).drop(columns=["device_type", "merchant_id"]), dev_test_df.reset_index(drop=True)], axis=1)

    # Ensure columns expected by scoring code are present and align ordering
    expected_cols = [
        "transaction_amount_wins",
        "hour_of_day",
        "day_of_week",
        "txns_5m",
        "txns_1h",
        "txns_24h",
        "dist_prev_km",
        "device_changed",
        "amount_over_avg",
        "merchant_fraud_rate",
        "merchant_freq",
    ]
    # Add device columns dynamically from ohe categories
    dev_expected = [f"device_{c}" for c in ohe.categories_[0]]
    # Combine
    expected_cols = expected_cols + dev_expected

    for col in expected_cols:
        if col not in X_train.columns:
            X_train[col] = 0
        if col not in X_test.columns:
            X_test[col] = 0

    # Reorder to expected cols (safe - all present now)
    X_train = X_train[expected_cols]
    X_test = X_test[expected_cols]

    # Scaling numeric features
    scaler = StandardScaler()
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # Save preprocessor artifacts
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(ohe, ENCODER_PATH)
    joblib.dump(merchant_freq, MERCHANT_FREQ_PATH)

    # Compute scale_pos_weight for XGBoost
    pos = int(y_train.sum())
    neg = len(y_train) - pos
    scale_pos_weight = max(1, neg / max(1, pos))
    logging.info(f"Train frauds={pos}, nonfrauds={neg}, scale_pos_weight={scale_pos_weight:.2f}")

    # XGBoost primary model
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        objective="binary:logistic",
        use_label_encoder=False,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)

    # Baseline models
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)

    lr = LogisticRegression(max_iter=500, class_weight="balanced", n_jobs=-1)
    lr.fit(X_train, y_train)

    def eval_model(m, Xv, yv) -> Dict[str, float]:
        prob = m.predict_proba(Xv)[:, 1]
        pred = (prob >= 0.5).astype(int)
        metrics = {
            "precision": precision_score(yv, pred, zero_division=0),
            "recall": recall_score(yv, pred, zero_division=0),
            "f1": f1_score(yv, pred, zero_division=0),
            "roc_auc": roc_auc_score(yv, prob) if len(np.unique(yv)) > 1 else 0.0,
            "pr_auc": average_precision_score(yv, prob) if len(np.unique(yv)) > 1 else 0.0,
        }
        return metrics

    res_xgb = eval_model(xgb, X_test, y_test)
    res_rf = eval_model(rf, X_test, y_test)
    res_lr = eval_model(lr, X_test, y_test)

    # Save final model
    joblib.dump(xgb, MODEL_PATH)
    logging.info(f"Saved XGBoost model to {MODEL_PATH}")

    return {
        "xgb": res_xgb,
        "rf": res_rf,
        "lr": res_lr,
        "n_test": len(y_test),
        "fraud_in_test": int(y_test.sum()),
    }


# ---------- Cost-based analysis ----------
def business_cost_matrix(y_true, y_pred, cost_fn=1000, cost_fp=10):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_cost = fn * cost_fn + fp * cost_fp
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "total_cost": total_cost}


# ---------- Real-time scoring ----------
class Transaction(BaseModel):
    transaction_id: str
    transaction_amount: float
    transaction_time: datetime
    customer_id: str
    merchant_id: str
    latitude: float
    longitude: float
    device_type: str


_artifacts = {"model": None, "scaler": None, "ohe": None, "merchant_freq": None}


def load_artifacts():
    if _artifacts["model"] is None:
        _artifacts["model"] = joblib.load(MODEL_PATH)
        _artifacts["scaler"] = joblib.load(SCALER_PATH)
        _artifacts["ohe"] = joblib.load(ENCODER_PATH)
        _artifacts["merchant_freq"] = joblib.load(MERCHANT_FREQ_PATH)
        logging.info("Artifacts loaded into memory")


def prepare_features_for_scoring(txn: Dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame([txn])
    df["transaction_time"] = pd.to_datetime(df["transaction_time"])
    df = preprocess_basic(df)
    df, feats = feature_engineering(df, merchant_freq=_artifacts.get("merchant_freq"))

    ohe = _artifacts["ohe"]
    # If for some reason the ohe was trained on different categories, handle_unknown='ignore' will keep shape consistent
    dev = ohe.transform(df[["device_type"]])
    dev_cols = [f"device_{c}" for c in ohe.categories_[0]]
    dev_df = pd.DataFrame(dev, columns=dev_cols, index=df.index)

    X = pd.concat([df.reset_index(drop=True).drop(columns=["device_type", "merchant_id"]), dev_df.reset_index(drop=True)], axis=1)

    # ---- Add expected column alignment code here ----
    expected_cols = [
        "transaction_amount_wins",
        "hour_of_day",
        "day_of_week",
        "txns_5m",
        "txns_1h",
        "txns_24h",
        "dist_prev_km",
        "device_changed",
        "amount_over_avg",
        "merchant_fraud_rate",
        "merchant_freq",
    ]
    dev_expected = [f"device_{c}" for c in ohe.categories_[0]]
    expected_cols = expected_cols + dev_expected

    for col in expected_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[expected_cols]
    # -----------------------------------------------

    scaler = _artifacts["scaler"]
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X[numeric_cols] = scaler.transform(X[numeric_cols])
    return X


@app.post("/score")
def score_transaction(txn: Transaction):
    load_artifacts()
    payload = txn.dict()
    X = prepare_features_for_scoring(payload)
    model = _artifacts["model"]
    prob = float(model.predict_proba(X)[:, 1][0])
    decision = "FLAGGED" if prob >= 0.5 else "APPROVED"
    return {"transaction_id": payload["transaction_id"], "fraud_probability": round(prob, 6), "decision": decision}


# ---------- CLI ----------
def main(args):
    if args.train:
        df = generate_synthetic_transactions(n_customers=3000, n_transactions=100000, fraud_rate=0.004)
        df = preprocess_basic(df)
        df, features = feature_engineering(df)
        # persist merchant frequency mapping (transactions frequency per merchant) for online scoring
        merchant_freq = df["merchant_id"].value_counts(normalize=True).to_dict()
        joblib.dump(merchant_freq, MERCHANT_FREQ_PATH)
        res = train_and_evaluate(df, features, label_col="is_fraud")
        logging.info("Evaluation summary:")
        logging.info(res)
    else:
        logging.info("Run with --train to generate data and train model. Then start API with uvicorn.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train models on synthetic dataset and save artifacts")
    args = parser.parse_args()
    main(args)

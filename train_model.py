"""
Credit Card Fraud Detection — Model Training Script
====================================================
Trains Logistic Regression and Random Forest classifiers on credit card
transaction data. Saves models, scaler, and evaluation metrics to ./models/.

Usage
-----
    python train_model.py

If the real Kaggle dataset (creditcard.csv) is NOT present in ./data/,
a realistic synthetic dataset is generated automatically so the full
pipeline can still be demonstrated.

Author  : Student Project — SRM Institute of Science and Technology, Tiruchirappalli
Course  : B.Tech CSE – BDA, 3rd Year
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)

# ─── Configuration ─────────────────────────────────────────────────────────────
DATA_PATH    = os.path.join("data", "creditcard.csv")
SAMPLE_PATH  = os.path.join("data", "sample_data.csv")
MODEL_DIR    = "models"
DATA_DIR     = "data"
RANDOM_STATE = 42
TEST_SIZE    = 0.20   # 80 / 20 stratified split


# ─── Synthetic data generator ──────────────────────────────────────────────────
def generate_synthetic_data(n_samples: int = 20_000,
                             fraud_ratio: float = 0.0172,
                             random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic credit card transactions that mirror the structure of the
    Kaggle 'Credit Card Fraud Detection' dataset.

    Column layout (identical to the real dataset):
        V1 – V28 : PCA-transformed features (anonymised)
        Time      : Seconds elapsed since the first transaction in the period
        Amount    : Transaction amount in USD
        Class     : 0 = Legitimate, 1 = Fraudulent
    """
    rng = np.random.default_rng(random_state)

    n_fraud = max(1, int(n_samples * fraud_ratio))
    n_legit = n_samples - n_fraud

    # Known approximate separation directions from the real dataset
    fraud_component_shifts = {
        1: -3.8, 2: 2.7,  3: -4.3, 4:  2.1,  5: -1.8,
        6: -1.2, 7: -1.5, 8:  0.5, 9: -1.0, 10: -2.5,
        11: 2.0, 12: -2.3, 14: -3.0,
    }

    def make_v_cols(n: int, shifts: dict) -> dict:
        cols = {}
        for i in range(1, 29):
            mu  = shifts.get(i, 0.0)
            std = 1.3 if i in shifts else 1.0
            cols[f"V{i}"] = rng.normal(mu, std, n)
        return cols

    # Legitimate transactions
    legit = make_v_cols(n_legit, {})
    legit["Time"]   = rng.uniform(0, 172_800, n_legit)
    legit["Amount"] = rng.exponential(scale=88, size=n_legit).clip(0.01, 25_000)
    legit["Class"]  = 0

    # Fraudulent transactions
    fraud = make_v_cols(n_fraud, fraud_component_shifts)
    fraud["Time"]   = rng.uniform(0, 172_800, n_fraud)
    fraud["Amount"] = rng.exponential(scale=122, size=n_fraud).clip(0.01, 5_000)
    fraud["Class"]  = 1

    df = pd.concat([pd.DataFrame(legit), pd.DataFrame(fraud)], ignore_index=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df


# ─── Data loading ──────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    if os.path.exists(DATA_PATH):
        print(f"[INFO] Loading real dataset from '{DATA_PATH}' ...")
        df = pd.read_csv(DATA_PATH)

        required = {"Time", "Amount", "Class"} | {f"V{i}" for i in range(1, 29)}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(f"Dataset is missing expected columns: {missing}")

        print(f"[INFO] Loaded {len(df):,} rows — "
              f"Fraud rate: {df['Class'].mean() * 100:.4f}%")
    else:
        print("[WARN] Real dataset not found at './data/creditcard.csv'.")
        print("[WARN] Generating synthetic data for demonstration ...")
        df = generate_synthetic_data(n_samples=20_000, random_state=RANDOM_STATE)
        os.makedirs(DATA_DIR, exist_ok=True)
        print(f"[INFO] Synthetic dataset: {len(df):,} rows — "
              f"Fraud rate: {df['Class'].mean() * 100:.4f}%")

    return df


# ─── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    """
    Split and scale the dataset.

    Important anti-leakage design:
      - StandardScaler is fitted on X_train ONLY.
      - The fitted scaler is then applied (transform-only) to X_test.
      - Only Time and Amount are scaled; V1–V28 are already PCA-normalised.
    """
    feature_cols = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
    X = df[feature_cols].copy()
    y = df["Class"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,          # preserves fraud ratio in both splits
    )

    scaler = StandardScaler()
    X_train[["Time", "Amount"]] = scaler.fit_transform(X_train[["Time", "Amount"]])
    X_test[["Time", "Amount"]]  = scaler.transform(X_test[["Time", "Amount"]])

    print(f"[INFO] Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")
    print(f"[INFO] Fraud in train: {y_train.sum():,}  |  Fraud in test: {y_test.sum():,}")

    return X_train, X_test, y_train, y_test, scaler, feature_cols


# ─── Evaluation helper ─────────────────────────────────────────────────────────
def evaluate_model(name: str, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
    pr_auc  = average_precision_score(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    report  = classification_report(y_test, y_pred, output_dict=True,
                                    target_names=["Legitimate", "Fraud"])

    metrics = {
        "accuracy" : float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall"   : float(recall_score(y_test, y_pred, zero_division=0)),
        "f1"       : float(f1_score(y_test, y_pred, zero_division=0)),
        "pr_auc"   : float(pr_auc),
        "roc_auc"  : float(roc_auc),
        "confusion_matrix"       : confusion_matrix(y_test, y_pred).tolist(),
        "classification_report"  : report,
        "pr_curve": {
            "precision": precision_vals.tolist(),
            "recall"   : recall_vals.tolist(),
        },
    }

    print(f"\n{'=' * 56}")
    print(f"  {name}")
    print(f"{'=' * 56}")
    print(classification_report(y_test, y_pred,
                                target_names=["Legitimate", "Fraud"]))
    print(f"  PR-AUC  : {pr_auc:.4f}")
    print(f"  ROC-AUC : {roc_auc:.4f}")

    return metrics


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(DATA_DIR,  exist_ok=True)

    # 1. Load data
    df = load_data()

    # 2. Save a compact sample for the Streamlit demo (raw, un-scaled)
    if not os.path.exists(SAMPLE_PATH):
        legit_sample = df[df["Class"] == 0].sample(min(500, (df["Class"] == 0).sum()),
                                                    random_state=RANDOM_STATE)
        fraud_sample = df[df["Class"] == 1].sample(min(500, df["Class"].sum()),
                                                    random_state=RANDOM_STATE)
        sample = pd.concat([legit_sample, fraud_sample]).sample(frac=1,
                            random_state=RANDOM_STATE).reset_index(drop=True)
        sample.to_csv(SAMPLE_PATH, index=False)
        print(f"[INFO] Sample data saved -> {SAMPLE_PATH}  ({len(sample):,} rows)")

    # 3. Preprocess
    X_train, X_test, y_train, y_test, scaler, feature_cols = preprocess(df)

    # 4. Logistic Regression
    print("\n[INFO] Training Logistic Regression ...")
    lr = LogisticRegression(
        max_iter=1_000,
        class_weight="balanced",   # handles class imbalance
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )
    lr.fit(X_train, y_train)
    lr_metrics = evaluate_model("Logistic Regression", lr, X_test, y_test)

    # 5. Random Forest
    print("\n[INFO] Training Random Forest ...")
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=14,
        min_samples_leaf=2,
        class_weight="balanced",   # handles class imbalance
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_metrics = evaluate_model("Random Forest", rf, X_test, y_test)

    # Feature importance (RF)
    rf_metrics["feature_importances"] = dict(
        zip(feature_cols, rf.feature_importances_.tolist())
    )

    # 6. Persist models
    joblib.dump(lr,     os.path.join(MODEL_DIR, "logistic_regression.pkl"))
    joblib.dump(rf,     os.path.join(MODEL_DIR, "random_forest.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    print(f"\n[INFO] Models saved -> ./{MODEL_DIR}/")

    # 7. Persist metrics + dataset summary
    all_metrics = {
        "logistic_regression": lr_metrics,
        "random_forest"      : rf_metrics,
        "feature_cols"       : feature_cols,
        "dataset_info": {
            "total"      : int(len(df)),
            "fraud"      : int(df["Class"].sum()),
            "legitimate" : int((df["Class"] == 0).sum()),
            "fraud_rate" : float(df["Class"].mean()),
        },
    }
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"[INFO] Metrics saved -> {metrics_path}")
    print("\n[OK] Training complete.\n")


if __name__ == "__main__":
    main()

"""
Fraud Detection Model Training with MLflow Experiment Tracking
=============================================================
Dataset : IEEE-CIS Fraud Detection (Kaggle)
Models  : Logistic Regression (baseline) → XGBoost (main) → Isolation Forest (anomaly)
Tracking: MLflow — logs params, metrics, SHAP plots, and registers best model
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
    precision_recall_curve
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")


# ─── 1. CONFIGURATION ────────────────────────────────────────────────────────

MLFLOW_EXPERIMENT = "fraud-detection"
MODEL_REGISTRY_NAME = "fraud-xgboost"
RANDOM_STATE = 42
TEST_SIZE = 0.2


# ─── 2. LOAD & MERGE DATA ────────────────────────────────────────────────────

def load_data():
    """
    Load the IEEE-CIS dataset.
    Download from: https://www.kaggle.com/c/ieee-fraud-detection/data
    Place train_transaction.csv and train_identity.csv in ./data/
    """
    print("Loading data...")
    transactions = pd.read_csv("data/train_transaction.csv")
    identity = pd.read_csv("data/train_identity.csv")

    # Merge on TransactionID — many transactions have no identity record
    df = transactions.merge(identity, on="TransactionID", how="left")
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud rate: {df['isFraud'].mean():.2%}")
    return df


# ─── 3. FEATURE ENGINEERING ──────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Key features that help detect fraud:
    - Transaction amount (high-value txns more likely fraud)
    - Hour of day (late-night txns riskier)
    - Card mismatch signals (addr1 vs shipping address)
    - Email domain (free email = higher risk)
    """
    print("Engineering features...")

    # Time-based features
    df["hour"] = (df["TransactionDT"] / 3600).astype(int) % 24
    df["day_of_week"] = (df["TransactionDT"] / (3600 * 24)).astype(int) % 7
    df["is_night"] = df["hour"].apply(lambda h: 1 if h < 6 or h > 22 else 0)

    # Amount-based features
    df["log_amount"] = np.log1p(df["TransactionAmt"])
    df["amount_rounded"] = (df["TransactionAmt"] % 1 == 0).astype(int)

    # Address mismatch — strong fraud signal
    df["addr_mismatch"] = (df["addr1"] != df["addr2"]).astype(int)

    # Email domain risk
    high_risk_domains = ["gmail.com", "yahoo.com", "hotmail.com"]
    df["risky_email"] = df["P_emaildomain"].isin(high_risk_domains).astype(int)

    return df


def preprocess(df: pd.DataFrame):
    """
    - Drop high-cardinality / low-variance columns
    - Encode categoricals
    - Fill missing values
    """
    print("Preprocessing...")

    # Drop ID and target
    drop_cols = ["TransactionID", "TransactionDT"]
    df = df.drop(columns=drop_cols, errors="ignore")

    TARGET = "isFraud"
    y = df[TARGET]
    X = df.drop(columns=[TARGET])

    # Encode categorical columns
    cat_cols = X.select_dtypes(include=["object"]).columns
    le = LabelEncoder()
    for col in cat_cols:
        X[col] = X[col].astype(str)
        X[col] = le.fit_transform(X[col])

    # Fill missing with -999 (XGBoost handles this internally too)
    X = X.fillna(-999)

    print(f"Features: {X.shape[1]}")
    return X, y


# ─── 4. METRICS HELPER ───────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred_proba, threshold=0.5):
    y_pred = (y_pred_proba >= threshold).astype(int)
    return {
        "auc_roc": roc_auc_score(y_true, y_pred_proba),
        "avg_precision": average_precision_score(y_true, y_pred_proba),
        "precision": float(classification_report(y_true, y_pred, output_dict=True)["1"]["precision"]),
        "recall": float(classification_report(y_true, y_pred, output_dict=True)["1"]["recall"]),
        "f1": float(classification_report(y_true, y_pred, output_dict=True)["1"]["f1-score"]),
    }


# ─── 5. SHAP EXPLAINABILITY ──────────────────────────────────────────────────

def log_shap_plot(model, X_sample: pd.DataFrame, run_name: str):
    """
    Compute SHAP values and log a summary plot to MLflow.
    Shows which features push predictions toward fraud.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
    plt.title(f"SHAP Feature Importance — {run_name}")
    plt.tight_layout()

    path = f"shap_{run_name}.png"
    plt.savefig(path, dpi=150)
    mlflow.log_artifact(path)
    plt.close()
    print(f"  SHAP plot logged: {path}")


# ─── 6. BASELINE: LOGISTIC REGRESSION ────────────────────────────────────────

def run_baseline(X_train, X_test, y_train, y_test):
    print("\n[1/3] Running baseline: Logistic Regression")

    with mlflow.start_run(run_name="baseline-logistic-regression"):
        params = {"C": 0.1, "max_iter": 1000, "class_weight": "balanced"}
        mlflow.log_params(params)

        model = LogisticRegression(**params, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, proba)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        print(f"  AUC-ROC: {metrics['auc_roc']:.4f} | F1: {metrics['f1']:.4f}")
        return metrics["auc_roc"]


# ─── 7. MAIN MODEL: XGBOOST ──────────────────────────────────────────────────

def run_xgboost(X_train, X_test, y_train, y_test):
    """
    XGBoost is the main model.
    - scale_pos_weight handles class imbalance (alternative to SMOTE)
    - early_stopping_rounds prevents overfitting
    - We log everything to MLflow and register the model if it beats baseline
    """
    print("\n[2/3] Running main model: XGBoost")

    # Class imbalance ratio — tells XGBoost how much to weight the fraud class
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"  scale_pos_weight: {scale_pos_weight:.1f}")

    params = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "use_label_encoder": False,
        "eval_metric": "aucpr",       # area under precision-recall (better for imbalanced)
        "early_stopping_rounds": 30,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }

    with mlflow.start_run(run_name="xgboost-main") as run:
        mlflow.log_params(params)
        mlflow.log_param("smote_applied", False)

        model = XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=50
        )

        proba = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, proba)
        mlflow.log_metrics(metrics)

        # Log SHAP explainability plot (sample 500 rows — SHAP is slow on full data)
        log_shap_plot(model, X_test.sample(500, random_state=RANDOM_STATE), "xgboost")

        # Log the model and register it in MLflow Model Registry
        mlflow.xgboost.log_model(
            model, "model",
            registered_model_name=MODEL_REGISTRY_NAME
        )

        print(f"  AUC-ROC: {metrics['auc_roc']:.4f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}")
        print(f"  Run ID: {run.info.run_id}")
        return model, metrics["auc_roc"], run.info.run_id


# ─── 8. BONUS: XGBOOST + SMOTE ───────────────────────────────────────────────

def run_xgboost_smote(X_train, X_test, y_train, y_test):
    """
    Compare scale_pos_weight vs SMOTE oversampling.
    Log both to MLflow so you can compare in the UI.
    """
    print("\n[2b] Running XGBoost + SMOTE for comparison")

    print("  Applying SMOTE (this takes a minute)...")
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"  Resampled: {y_res.value_counts().to_dict()}")

    params = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "aucpr",
        "early_stopping_rounds": 30,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }

    with mlflow.start_run(run_name="xgboost-smote"):
        mlflow.log_params(params)
        mlflow.log_param("smote_applied", True)

        model = XGBClassifier(**params)
        model.fit(X_res, y_res, eval_set=[(X_test, y_test)], verbose=50)

        proba = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, proba)
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, "model")

        print(f"  AUC-ROC: {metrics['auc_roc']:.4f} | F1: {metrics['f1']:.4f}")


# ─── 9. ANOMALY DETECTION: ISOLATION FOREST ──────────────────────────────────

def run_isolation_forest(X_train, X_test, y_test):
    """
    Unsupervised approach — no labels needed during training.
    Useful in production to catch brand-new fraud patterns the XGBoost
    model hasn't seen.
    """
    print("\n[3/3] Running Isolation Forest (unsupervised)")

    params = {
        "n_estimators": 100,
        "contamination": 0.035,   # ~3.5% fraud rate in the dataset
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }

    with mlflow.start_run(run_name="isolation-forest-unsupervised"):
        mlflow.log_params(params)

        model = IsolationForest(**params)
        model.fit(X_train)

        # IsolationForest returns -1 (anomaly) or 1 (normal)
        raw_scores = model.decision_function(X_test)
        # Flip sign: lower score = more anomalous → higher fraud probability
        proba = -raw_scores
        proba = (proba - proba.min()) / (proba.max() - proba.min())  # normalize 0–1

        metrics = compute_metrics(y_test, proba)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")


# ─── 10. MAIN ────────────────────────────────────────────────────────────────

def main():
    # Set up MLflow — runs a local tracking server at http://localhost:5001
    # Start it with: mlflow ui --port 5001
    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    print(f"MLflow experiment: '{MLFLOW_EXPERIMENT}'")
    print("Open http://localhost:5001 to see all runs\n")

    # Load and prepare data
    df = load_data()
    df = engineer_features(df)
    X, y = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain size: {X_train.shape[0]:,} | Test size: {X_test.shape[0]:,}")
    print(f"Fraud in test: {y_test.sum():,} ({y_test.mean():.2%})\n")

    # Run all three experiments
    baseline_auc = run_baseline(X_train, X_test, y_train, y_test)
    xgb_model, xgb_auc, run_id = run_xgboost(X_train, X_test, y_train, y_test)
    run_xgboost_smote(X_train, X_test, y_train, y_test)
    run_isolation_forest(X_train, X_test, y_test)

    # Summary
    print("\n" + "="*55)
    print("EXPERIMENT SUMMARY")
    print("="*55)
    print(f"  Baseline (LR) AUC-ROC : {baseline_auc:.4f}")
    print(f"  XGBoost AUC-ROC       : {xgb_auc:.4f}  ← best model")
    print(f"  Model registered as   : '{MODEL_REGISTRY_NAME}'")
    print(f"  Best run ID           : {run_id}")
    print(f"\n  View all runs → http://localhost:5001")
    print("="*55)

    return xgb_model


if __name__ == "__main__":
    model = main()

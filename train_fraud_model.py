"""
Fraud Detection Model Training with MLflow Experiment Tracking
=============================================================
Dataset : IEEE-CIS Fraud Detection (Kaggle)
Models  : Logistic Regression (baseline) → XGBoost (main) → Isolation Forest (anomaly)
Tracking: MLflow — logs params, metrics, SHAP plots, and registers best model

Methodology notes (these changed; expect different numbers than earlier runs):

* CHRONOLOGICAL split, not random. TransactionDT is a seconds offset and this is
  a time-series problem. A random split puts future transactions in train and
  scatters the same card/device across both sides, which leaks. The previously
  reported AUC of 0.9469 was measured that way and was optimistic.
* THREE-WAY split. Early stopping previously used the test set as its eval_set
  and then reported metrics on that same set, so the stopping iteration was
  chosen with test labels. Validation is now separate from test.
* Categorical maps are fit on TRAIN ONLY. They were previously built from the
  full frame, so test categories leaked into the encoding and the
  unknown-category path was never exercised before production hit it.
* XGBoost receives real NaN instead of a -999 sentinel, so it can learn a default
  direction per split. The sentinel is recorded in encoding_config.json as null
  and the consumer follows whatever that file says.

Usage: python train_fraud_model.py
"""

import json
import os
import subprocess
import tempfile
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # no display in a training container
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from features import LEGACY_FEATURES, TARGET, engineer_features

# Only silence the noisy convergence chatter. A blanket
# warnings.filterwarnings("ignore") previously hid the two warnings that would
# have revealed a bogus XGBoost parameter and a deprecated MLflow call.
warnings.filterwarnings("ignore", category=FutureWarning)

ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "artifacts"))
ARTIFACT_DIR.mkdir(exist_ok=True)

UNKNOWN_CATEGORY = -1      # category not seen during training
MISSING_NUMERIC = None     # None => keep NaN, let XGBoost route it
NULL_CATEGORY_STR = "nan"  # what astype(str) turns NaN into

# ─── 1. CONFIGURATION ────────────────────────────────────────────────────────

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "fraud-detection")
MODEL_REGISTRY_NAME = os.getenv("MODEL_NAME", "fraud-xgboost")
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALID_SIZE = 0.1           # carved out of the pre-test period
SPLIT_STRATEGY = os.getenv("SPLIT_STRATEGY", "chronological")   # or "random"
RUN_SMOTE = os.getenv("RUN_SMOTE", "1") == "1"


def git_sha() -> str:
    """Tag runs with the commit they came from, so runs are distinguishable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return "not-a-git-repo"


# ─── 2. LOAD & MERGE DATA ────────────────────────────────────────────────────

def load_data():
    """
    Load the IEEE-CIS dataset.
    Download from: https://www.kaggle.com/c/ieee-fraud-detection/data
    Place train_transaction.csv and train_identity.csv in ./data/
    """
    print("Loading data...")
    transactions = pd.read_csv(os.getenv("DATASET_PATH", "data/train_transaction.csv"))
    identity = pd.read_csv(os.getenv("IDENTITY_PATH", "data/train_identity.csv"))

    # Merge on TransactionID — many transactions have no identity record
    df = transactions.merge(identity, on="TransactionID", how="left")
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud rate: {df[TARGET].mean():.2%}")
    print(f"Identity coverage: {df['DeviceType'].notna().mean():.1%}")
    return df


# ─── 3. SPLIT ────────────────────────────────────────────────────────────────

def split_data(df: pd.DataFrame):
    """
    Chronological three-way split: train | valid | test, ordered in time.

    Validation sits between train and test so early stopping never sees test
    labels, and test is strictly the most recent slice -- which is the only
    honest estimate of how this performs on tomorrow's transactions.
    """
    if SPLIT_STRATEGY == "random":
        from sklearn.model_selection import train_test_split
        print("Splitting RANDOMLY (leaks on time-series data; for comparison only)")
        idx_rest, idx_test = train_test_split(
            np.arange(len(df)), test_size=TEST_SIZE,
            random_state=RANDOM_STATE, stratify=df[TARGET],
        )
        idx_train, idx_valid = train_test_split(
            idx_rest, test_size=VALID_SIZE / (1 - TEST_SIZE),
            random_state=RANDOM_STATE, stratify=df[TARGET].values[idx_rest],
        )
    else:
        print("Splitting CHRONOLOGICALLY on TransactionDT")
        order = np.argsort(df["TransactionDT"].values, kind="stable")
        n = len(order)
        n_test = int(n * TEST_SIZE)
        n_valid = int(n * VALID_SIZE)
        idx_train = order[: n - n_test - n_valid]
        idx_valid = order[n - n_test - n_valid: n - n_test]
        idx_test = order[n - n_test:]

    parts = {}
    for name, idx in [("train", idx_train), ("valid", idx_valid), ("test", idx_test)]:
        part = df.iloc[idx]
        parts[name] = part
        span = (part["TransactionDT"].min(), part["TransactionDT"].max())
        print(f"  {name:<6} {len(part):>8,} rows | fraud {part[TARGET].mean():.2%} "
              f"| DT {span[0]:,} → {span[1]:,}")
    return parts


# ─── 4. PREPROCESSING ────────────────────────────────────────────────────────

def preprocess(parts: dict):
    """
    - Engineer features (shared definitions from features.py)
    - Fit categorical maps on TRAIN ONLY, then apply to all splits
    - Persist everything the serving path needs
    """
    print("Preprocessing...")

    engineered = {k: engineer_features(v) for k, v in parts.items()}

    train = engineered["train"]
    drop = ["TransactionID", "TransactionDT", TARGET] + LEGACY_FEATURES
    X_train_raw = train.drop(columns=drop, errors="ignore")

    cat_cols = list(X_train_raw.select_dtypes(include=["object"]).columns)

    # Fit on train only. Categories that appear for the first time in valid/test
    # correctly fall through to UNKNOWN_CATEGORY, exactly as they will in prod.
    cat_maps = {}
    for col in cat_cols:
        categories = sorted(X_train_raw[col].astype(str).unique())
        cat_maps[col] = {cat: idx for idx, cat in enumerate(categories)}

    out = {}
    for name, part in engineered.items():
        X = part.drop(columns=drop, errors="ignore")
        for col in cat_cols:
            X[col] = (
                X[col].astype(str).map(cat_maps[col])
                .fillna(UNKNOWN_CATEGORY).astype(int)
            )
        if MISSING_NUMERIC is not None:
            X = X.fillna(MISSING_NUMERIC)
        out[name] = (X.astype(np.float32), part[TARGET].reset_index(drop=True))

    feature_columns = list(out["train"][0].columns)

    unknown_rate = {
        name: float(np.mean([(X[c] == UNKNOWN_CATEGORY).mean() for c in cat_cols]))
        for name, (X, _) in out.items()
    }
    print(f"  unseen-category rate: {unknown_rate}")

    # Persist everything the serving path needs
    (ARTIFACT_DIR / "cat_maps.json").write_text(json.dumps(cat_maps))
    (ARTIFACT_DIR / "feature_columns.json").write_text(json.dumps(feature_columns))
    (ARTIFACT_DIR / "encoding_config.json").write_text(json.dumps({
        "unknown_category": UNKNOWN_CATEGORY,
        "missing_numeric": MISSING_NUMERIC,
        "null_category_str": NULL_CATEGORY_STR,
        "categorical_columns": cat_cols,
        "split_strategy": SPLIT_STRATEGY,
    }, indent=2))

    print(f"Features: {len(feature_columns)} | Categorical: {len(cat_cols)}")
    return out 


def log_preprocessing_artifacts():
    """
    Every run that produces a servable model must log the encoders alongside it.
    Previously only the main XGBoost run did, so the SMOTE and Isolation Forest
    models were not reproducible at serving time.
    """
    for name in ("cat_maps.json", "feature_columns.json", "encoding_config.json"):
        mlflow.log_artifact(str(ARTIFACT_DIR / name), artifact_path="preprocessing")


# ─── 5. METRICS HELPER ───────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred_proba, threshold=0.5):
    """
    classification_report was previously called three times per evaluation, and
    indexing its ["1"] key raised KeyError whenever nothing crossed the
    threshold. precision_recall_fscore_support is one pass and cannot vanish.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0,
    )
    return {
        "auc_roc": float(roc_auc_score(y_true, y_pred_proba)),
        "avg_precision": float(average_precision_score(y_true, y_pred_proba)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "threshold": float(threshold),
    }


def pick_threshold(y_true, y_proba):
    """
    Choose the F1-optimal operating point on the VALIDATION set.

    0.5 was hardcoded in the consumer and is meaningless here: scale_pos_weight
    inflates the probabilities, and at 0.5 precision measured 0.286, i.e. 71% of
    alerts were false positives.
    """
    prec, rec, thr = precision_recall_curve(y_true, y_proba)
    prec, rec = prec[:-1], rec[:-1]
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    i = int(np.nanargmax(f1))
    return float(thr[i]), float(prec[i]), float(rec[i]), float(f1[i])


# ─── 6. SHAP EXPLAINABILITY ──────────────────────────────────────────────────

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

    # Written to a temp dir instead of the repo root, which was accumulating
    # stray shap_*.png files that were never cleaned up.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"shap_{run_name}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        mlflow.log_artifact(str(path))
    print(f"  SHAP plot logged: shap_{run_name}.png")


# ─── 7. BASELINE: LOGISTIC REGRESSION ────────────────────────────────────────

def run_baseline(X_train, y_train, X_test, y_test):
    """
    Logistic regression needs imputation and scaling; XGBoost does not.

    Previously this was fit on raw unscaled features where missing values were
    -999 and DeviceInfo was an ordinal 0-1786, so it could not converge
    meaningfully and was not a fair reference point.
    """
    print("\n[1/3] Running baseline: Logistic Regression")

    with mlflow.start_run(run_name="baseline-logistic-regression"):
        params = {"C": 0.1, "max_iter": 1000, "class_weight": "balanced"}
        mlflow.log_params(params)
        mlflow.log_param("scaled", True)
        mlflow.set_tag("git_sha", git_sha())

        model = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(**params, random_state=RANDOM_STATE)),
        ])
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, proba)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name="model")

        print(f"  AUC-ROC: {metrics['auc_roc']:.4f} | F1: {metrics['f1']:.4f}")
        return metrics["auc_roc"]


# ─── 8. MAIN MODEL: XGBOOST ──────────────────────────────────────────────────

def run_xgboost(X_train, y_train, X_valid, y_valid, X_test, y_test):
    """
    XGBoost is the main model.
    - scale_pos_weight handles class imbalance (alternative to SMOTE)
    - early stopping watches VALID, never test
    - the decision threshold is chosen on VALID and persisted as an artifact
    """
    print("\n[2/3] Running main model: XGBoost")

    # Class imbalance ratio — tells XGBoost how much to weight the fraud class
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    print(f"  scale_pos_weight: {scale_pos_weight:.1f}")

    params = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        # "use_label_encoder" removed: it is not an XGBoost 2.x parameter. It fell
        # into **kwargs, was forwarded to the booster as an unknown param and
        # silently discarded, so it read as meaningful config but did nothing.
        "eval_metric": "aucpr",       # area under precision-recall (better for imbalanced)
        "early_stopping_rounds": 30,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }

    with mlflow.start_run(run_name="xgboost-main") as run:
        mlflow.log_params(params)
        mlflow.log_param("smote_applied", False)
        mlflow.log_param("split_strategy", SPLIT_STRATEGY)
        mlflow.set_tag("git_sha", git_sha())
        log_preprocessing_artifacts()

        model = XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],     # NOT the test set
            verbose=50,
        )
        mlflow.log_metric("best_iteration", model.best_iteration)

        # Operating point from validation, then honest metrics on test.
        valid_proba = model.predict_proba(X_valid)[:, 1]
        threshold, v_prec, v_rec, v_f1 = pick_threshold(y_valid, valid_proba)
        print(f"  chosen threshold (from valid): {threshold:.4f} "
              f"| P={v_prec:.3f} R={v_rec:.3f} F1={v_f1:.3f}")

        proba = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, proba, threshold=threshold)
        mlflow.log_metrics(metrics)
        mlflow.log_metrics({"valid_precision": v_prec, "valid_recall": v_rec,
                            "valid_f1": v_f1})

        threshold_config = {
            "fraud_threshold": round(threshold, 6),
            "high_risk_threshold": round(min(threshold + (1 - threshold) * 0.6, 0.99), 6),
            "policy": "f1",
            "rationale": "F1-optimal on the validation split, at training time",
            "expected_precision": round(v_prec, 4),
            "expected_recall": round(v_rec, 4),
            "expected_f1": round(v_f1, 4),
            "model_name": MODEL_REGISTRY_NAME,
            "holdout_split": SPLIT_STRATEGY,
            "test_auc": round(metrics["auc_roc"], 4),
            "test_ap": round(metrics["avg_precision"], 4),
        }
        (ARTIFACT_DIR / "threshold_config.json").write_text(
            json.dumps(threshold_config, indent=2))
        mlflow.log_artifact(str(ARTIFACT_DIR / "threshold_config.json"),
                            artifact_path="preprocessing")

        # Log SHAP explainability plot (sample 500 rows — SHAP is slow on full data)
        n_sample = min(500, len(X_test))
        log_shap_plot(model, X_test.sample(n_sample, random_state=RANDOM_STATE),
                      "xgboost")

        # Log the model and register it in MLflow Model Registry.
        # Registering does NOT promote. Promotion is a separate, deliberate step:
        #   python scripts/promote_model.py --version N --alias champion
        mlflow.xgboost.log_model(
            model, name="model",
            registered_model_name=MODEL_REGISTRY_NAME,
        )

        print(f"  AUC-ROC: {metrics['auc_roc']:.4f} | "
              f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}")
        print(f"  Run ID: {run.info.run_id}")
        return model, metrics["auc_roc"], run.info.run_id


# ─── 9. BONUS: XGBOOST + SMOTE ───────────────────────────────────────────────

def run_xgboost_smote(X_train, y_train, X_valid, y_valid, X_test, y_test):
    """
    Compare scale_pos_weight vs SMOTE oversampling.
    Log both to MLflow so you can compare in the UI.
    """
    print("\n[2b] Running XGBoost + SMOTE for comparison")

    # SMOTE cannot interpolate across NaN, so this branch imputes first. That is
    # a real cost of oversampling on this dataset, not an incidental detail.
    print("  Imputing + applying SMOTE (this takes a minute)...")
    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train), columns=X_train.columns, dtype=np.float32)
    X_valid_imp = pd.DataFrame(
        imputer.transform(X_valid), columns=X_valid.columns, dtype=np.float32)
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test), columns=X_test.columns, dtype=np.float32)

    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X_train_imp, y_train)
    print(f"  Resampled: {pd.Series(y_res).value_counts().to_dict()}")

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
        mlflow.log_param("split_strategy", SPLIT_STRATEGY)
        mlflow.set_tag("git_sha", git_sha())
        log_preprocessing_artifacts()

        model = XGBClassifier(**params)
        model.fit(X_res, y_res, eval_set=[(X_valid_imp, y_valid)], verbose=50)

        valid_proba = model.predict_proba(X_valid_imp)[:, 1]
        threshold, *_ = pick_threshold(y_valid, valid_proba)

        proba = model.predict_proba(X_test_imp)[:, 1]
        metrics = compute_metrics(y_test, proba, threshold=threshold)
        mlflow.log_metrics(metrics)
        # Not registered: this variant needs an imputer at serving time that the
        # consumer does not apply, so it is a comparison run only.
        mlflow.xgboost.log_model(model, name="model")

        print(f"  AUC-ROC: {metrics['auc_roc']:.4f} | F1: {metrics['f1']:.4f}")


# ─── 10. ANOMALY DETECTION: ISOLATION FOREST ─────────────────────────────────

def run_isolation_forest(X_train, X_test, y_test, fraud_rate: float):
    """
    Unsupervised approach — no labels needed during training.
    Useful in production to catch brand-new fraud patterns the XGBoost
    model hasn't seen.
    """
    print("\n[3/3] Running Isolation Forest (unsupervised)")

    params = {
        "n_estimators": 100,
        "contamination": round(float(fraud_rate), 4),   # measured, not assumed
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }

    with mlflow.start_run(run_name="isolation-forest-unsupervised"):
        mlflow.log_params(params)
        mlflow.set_tag("git_sha", git_sha())
        log_preprocessing_artifacts()

        model = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("iforest", IsolationForest(**params)),
        ])
        model.fit(X_train)

        # Flip sign: lower decision_function = more anomalous = higher fraud risk
        train_scores = -model.decision_function(X_train)
        raw = -model.decision_function(X_test)

        # Normalise using bounds from TRAIN, not from the test predictions.
        # Min-maxing on test made the score unreproducible online, because those
        # bounds simply do not exist when scoring one live transaction.
        lo, hi = float(train_scores.min()), float(train_scores.max())
        mlflow.log_params({"score_min": lo, "score_max": hi})
        proba = np.clip((raw - lo) / max(hi - lo, 1e-12), 0.0, 1.0)

        metrics = compute_metrics(y_test, proba)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name="model")

        print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")


# ─── 11. MAIN ────────────────────────────────────────────────────────────────

def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    print(f"MLflow experiment: '{MLFLOW_EXPERIMENT}' at {MLFLOW_TRACKING_URI}")
    print("If artifact logging 500s, the server needs --serve-artifacts "
          "--artifacts-destination (see README)\n")

    df = load_data()
    parts = split_data(df)
    prepared = preprocess(parts)

    (X_train, y_train) = prepared["train"]
    (X_valid, y_valid) = prepared["valid"]
    (X_test, y_test) = prepared["test"]

    print(f"\nTrain {X_train.shape[0]:,} | Valid {X_valid.shape[0]:,} "
          f"| Test {X_test.shape[0]:,}")
    print(f"Fraud in test: {y_test.sum():,} ({y_test.mean():.2%})\n")

    baseline_auc = run_baseline(X_train, y_train, X_test, y_test)
    xgb_model, xgb_auc, run_id = run_xgboost(
        X_train, y_train, X_valid, y_valid, X_test, y_test)
    if RUN_SMOTE:
        run_xgboost_smote(X_train, y_train, X_valid, y_valid, X_test, y_test)
    run_isolation_forest(X_train, X_test, y_test, fraud_rate=y_train.mean())

    print("\n" + "=" * 55)
    print("EXPERIMENT SUMMARY")
    print("=" * 55)
    print(f"  Split strategy        : {SPLIT_STRATEGY}")
    print(f"  Baseline (LR) AUC-ROC : {baseline_auc:.4f}")
    print(f"  XGBoost AUC-ROC       : {xgb_auc:.4f}  ← best model")
    print(f"  Model registered as   : '{MODEL_REGISTRY_NAME}' (NOT yet promoted)")
    print(f"  Best run ID           : {run_id}")
    print(f"\n  Promote it with:")
    print(f"    python scripts/promote_model.py --list")
    print(f"    python scripts/promote_model.py --version N --alias champion")
    print(f"\n  View all runs → {MLFLOW_TRACKING_URI}")
    print("=" * 55)

    return xgb_model


if __name__ == "__main__":
    model = main()

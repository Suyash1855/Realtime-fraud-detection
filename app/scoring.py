"""
Scoring engine for the deployed demo
====================================
Loads the frozen bundle produced by scripts/export_for_deploy.py and scores
transactions with the real XGBoost model. No MLflow, no network calls.

Feature engineering comes from features.py -- the same module training uses -- so
the demo inherits the train/serve parity guarantee rather than reimplementing the
pipeline a third time.

Two scoring paths, deliberately:

  score_frame()  vectorised, used once at startup to precompute every demo score.
                 Powers the threshold slider and the analytics charts, which then
                 need no further inference and respond instantly.

  score_batch()  the live path used by the simulator. Re-scores rows in real time
                 so the latency shown on the dashboard is genuinely measured, not
                 read back from a cache.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from features import prepare_matrix

log = logging.getLogger(__name__)


@dataclass
class Contract:
    """The train/serve contract, loaded from the frozen bundle."""
    cat_maps: dict
    feature_columns: list
    unknown_category: int
    missing_numeric: float | None
    null_category_str: str
    fraud_threshold: float
    high_risk_threshold: float
    manifest: dict

    @property
    def model_version(self) -> str:
        return str(self.manifest.get("model_version", "?"))

    @property
    def split_strategy(self) -> str:
        return str(self.manifest.get("split_strategy", "unknown"))


class ScoringEngine:
    def __init__(self, bundle_dir: Path):
        self.bundle_dir = Path(bundle_dir)
        self.contract = self._load_contract()
        self.booster = self._load_model()
        self._explainer = None          # SHAP is built lazily; it is slow to init
        log.info(
            "Scoring engine ready | model v%s | %d features | threshold %.4f",
            self.contract.model_version,
            len(self.contract.feature_columns),
            self.contract.fraud_threshold,
        )

    # ── loading ──────────────────────────────────────────────────────────────

    def _load_contract(self) -> Contract:
        def read(name):
            path = self.bundle_dir / name
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} missing. Run: python scripts/export_for_deploy.py"
                )
            return json.loads(path.read_text())

        cfg = read("encoding_config.json")
        thr = read("threshold_config.json")
        manifest = read("manifest.json")

        return Contract(
            cat_maps=read("cat_maps.json"),
            feature_columns=read("feature_columns.json"),
            unknown_category=cfg.get("unknown_category", -1),
            missing_numeric=cfg.get("missing_numeric", -999),
            null_category_str=cfg.get("null_category_str", "nan"),
            fraud_threshold=float(thr["fraud_threshold"]),
            high_risk_threshold=float(thr.get("high_risk_threshold", 0.95)),
            manifest=manifest,
        )

    def _load_model(self) -> xgb.Booster:
        path = self.bundle_dir / "model.xgb"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing. Run: python scripts/export_for_deploy.py"
            )
        booster = xgb.Booster()
        booster.load_model(str(path))

        # Feature order is positional in XGBoost. If the booster carries names,
        # they must match the contract or every score is silently wrong.
        if booster.feature_names:
            if list(booster.feature_names) != list(self.contract.feature_columns):
                raise RuntimeError(
                    "Feature order mismatch between model and contract: "
                    f"model={len(booster.feature_names)} "
                    f"contract={len(self.contract.feature_columns)}. "
                    "The bundle is inconsistent -- re-export it."
                )
        return booster

    # ── feature preparation ──────────────────────────────────────────────────

    def build_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Raw transactions -> ordered, encoded float32 feature matrix."""
        return prepare_matrix(
            df,
            self.contract.feature_columns,
            self.contract.cat_maps,
            self.contract.unknown_category,
            self.contract.missing_numeric,
            self.contract.null_category_str,
        )

    def _dmatrix(self, X: pd.DataFrame) -> xgb.DMatrix:
        return xgb.DMatrix(X, feature_names=list(self.contract.feature_columns))

    # ── scoring ──────────────────────────────────────────────────────────────

    def score_frame(self, df: pd.DataFrame) -> np.ndarray:
        """Vectorised scoring over a whole frame. Used once at startup."""
        X = self.build_matrix(df)
        return self.booster.predict(self._dmatrix(X)).astype(np.float64)

    def score_batch(self, df: pd.DataFrame) -> tuple[np.ndarray, float]:
        """
        Live scoring path. Returns (probabilities, elapsed_ms) where elapsed_ms
        covers feature building AND inference -- the number a real consumer would
        report, not just the model call.
        """
        t0 = time.perf_counter()
        X = self.build_matrix(df)
        proba = self.booster.predict(self._dmatrix(X)).astype(np.float64)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return proba, elapsed_ms

    def classify(self, proba: float, threshold: float | None = None,
                 high_risk: float | None = None) -> tuple[bool, str]:
        thr = self.contract.fraud_threshold if threshold is None else threshold
        hi = self.contract.high_risk_threshold if high_risk is None else high_risk
        hi = max(hi, thr)
        if proba >= hi:
            return True, "HIGH"
        if proba >= thr:
            return True, "MEDIUM"
        return False, "LOW"

    # ── explainability ───────────────────────────────────────────────────────

    def explain(self, df_row: pd.DataFrame, top_n: int = 12) -> dict:
        """
        Per-transaction SHAP contributions: which features pushed this particular
        transaction toward or away from fraud.

        Uses XGBoost's built-in pred_contribs rather than the shap package. It is
        exact for trees, needs no extra dependency in the image, and is fast
        enough to call synchronously per request.
        """
        X = self.build_matrix(df_row)
        contribs = self.booster.predict(self._dmatrix(X), pred_contribs=True)[0]

        # Last element is the bias/base value.
        base_value = float(contribs[-1])
        feature_contribs = contribs[:-1]
        names = list(self.contract.feature_columns)
        values = X.iloc[0].to_numpy()

        order = np.argsort(np.abs(feature_contribs))[::-1][:top_n]
        top = [
            {
                "feature": names[i],
                "contribution": float(feature_contribs[i]),
                "value": (None if not np.isfinite(values[i]) else float(values[i])),
                "direction": "fraud" if feature_contribs[i] > 0 else "legitimate",
            }
            for i in order
        ]

        logit = base_value + float(feature_contribs.sum())
        return {
            "base_value": base_value,
            "logit": logit,
            "probability": float(1.0 / (1.0 + np.exp(-logit))),
            "top_features": top,
            "total_features": len(names),
        }

    # ── metadata for the UI ──────────────────────────────────────────────────

    def model_info(self) -> dict:
        m = self.contract.manifest
        metrics = m.get("metrics", {})
        return {
            "model_name": m.get("model_name", "fraud-xgboost"),
            "model_version": self.contract.model_version,
            "run_id": m.get("run_id"),
            "split_strategy": self.contract.split_strategy,
            "n_features": len(self.contract.feature_columns),
            "n_trees": int(self.booster.num_boosted_rounds()),
            "fraud_threshold": self.contract.fraud_threshold,
            "high_risk_threshold": self.contract.high_risk_threshold,
            "threshold_config": m.get("threshold_config", {}),
            "metrics": {
                k: round(v, 4) for k, v in metrics.items()
                if k in {"auc_roc", "avg_precision", "precision", "recall", "f1",
                         "best_iteration", "valid_precision", "valid_recall", "valid_f1"}
            },
        }

"""
Train/serve parity — the most important test in this repo
========================================================
Two independent implementations of feature engineering exist, and they must agree:

  features.py            operates on a DataFrame  (training, batch scoring)
  consumer.engineer_features / build_feature_row   operates on one dict (streaming)

They cannot be merged: one is vectorised over a frame, the other scores a single
Kafka message. So the duplication is guarded here instead.

This test would have caught both of the original skew bugs:
  * addr_mismatch disagreeing on 5.2% of rows (NaN != NaN in pandas vs a
    truthiness check in the consumer)
  * the producer never merging train_identity.csv, leaving 41 features
    permanently missing at serving time

Run: python -m pytest tests/ -v
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import consumer                                  # noqa: E402
from features import prepare_matrix              # noqa: E402

ARTIFACTS = ROOT / "artifacts"
TRANSACTIONS = ROOT / "data" / "train_transaction.csv"
IDENTITY = ROOT / "data" / "train_identity.csv"
N_ROWS = 500

pytestmark = pytest.mark.skipif(
    not TRANSACTIONS.exists() or not (ARTIFACTS / "cat_maps.json").exists(),
    reason="needs data/ and artifacts/ (run train_fraud_model.py first)",
)


@pytest.fixture(scope="module")
def contract():
    cat_maps = json.loads((ARTIFACTS / "cat_maps.json").read_text())
    feature_columns = json.loads((ARTIFACTS / "feature_columns.json").read_text())
    cfg = json.loads((ARTIFACTS / "encoding_config.json").read_text())
    policy = {
        "unknown_category": cfg.get("unknown_category", -1),
        "missing_numeric": cfg.get("missing_numeric", -999),
        "null_category_str": cfg.get("null_category_str", "nan"),
    }
    return cat_maps, feature_columns, policy


@pytest.fixture(scope="module")
def raw_rows():
    """Rows exactly as producer.py now publishes them: transaction JOIN identity."""
    tx = pd.read_csv(TRANSACTIONS, nrows=N_ROWS)
    if IDENTITY.exists():
        tx = tx.merge(pd.read_csv(IDENTITY), on="TransactionID", how="left")
    return tx


def _as_kafka_messages(df):
    """Mirror producer.build_transaction: NaN -> None, numpy scalars -> Python."""
    out = []
    for row in df.to_dict("records"):
        msg = {}
        for k, v in row.items():
            if v is None or (isinstance(v, (float, np.floating)) and np.isnan(v)):
                msg[k] = None
            elif isinstance(v, np.generic):
                msg[k] = v.item()
            else:
                msg[k] = v
        out.append(msg)
    return out


def test_feature_vectors_are_identical(contract, raw_rows):
    """The core assertion: same input row -> same feature vector, both paths."""
    cat_maps, feature_columns, policy = contract

    batch = prepare_matrix(
        raw_rows, feature_columns, cat_maps,
        policy["unknown_category"], policy["missing_numeric"],
    )

    rows = [
        consumer.build_feature_row(msg, feature_columns, cat_maps, policy)
        for msg in _as_kafka_messages(raw_rows)
    ]
    stream = consumer.build_feature_frame(rows, feature_columns)

    assert list(batch.columns) == list(stream.columns), "feature ORDER differs"
    assert batch.shape == stream.shape

    mismatched = [
        col for col in feature_columns
        if not np.allclose(
            batch[col].values, stream[col].values,
            equal_nan=True, rtol=1e-6, atol=1e-6,
        )
    ]
    assert not mismatched, (
        f"{len(mismatched)} feature(s) differ between the training and serving "
        f"paths: {mismatched[:20]}"
    )


def test_engineered_features_agree_row_by_row(contract, raw_rows):
    """
    Narrower check on just the hand-written features, so a failure points at the
    specific definition that drifted rather than at a 438-column diff.
    """
    from features import engineer_features as batch_engineer

    engineered = [
        "hour", "day_of_week", "is_night", "log_amount", "amount_rounded",
        "addr_mismatch", "addr_missing", "email_mismatch", "risky_email",
    ]

    batch = batch_engineer(raw_rows)
    messages = _as_kafka_messages(raw_rows)

    for i, msg in enumerate(messages):
        served = consumer.engineer_features(dict(msg))
        for feat in engineered:
            if feat not in batch.columns:
                continue
            expected = batch[feat].iloc[i]
            actual = served[feat]
            assert np.isclose(float(expected), float(actual), rtol=1e-6, atol=1e-6), (
                f"row {i} feature '{feat}': training={expected} serving={actual}"
            )


def test_identity_features_are_populated(contract, raw_rows):
    """
    Guards the producer regression: if identity is not merged, all 41 identity
    features fall to the missing sentinel and scores drift from training.
    """
    cat_maps, feature_columns, policy = contract
    missing = policy["missing_numeric"]

    rows = [
        consumer.build_feature_row(msg, feature_columns, cat_maps, policy)
        for msg in _as_kafka_messages(raw_rows)
    ]
    frame = consumer.build_feature_frame(rows, feature_columns)

    id_cols = [c for c in feature_columns if c.startswith("id_")]
    assert id_cols, "expected id_* features in the contract"

    def is_absent(series):
        return series.isna().all() if missing is None else (series == missing).all()

    populated = [c for c in id_cols if not is_absent(frame[c])]
    assert populated, (
        "every id_* feature is missing for all rows — identity data is not "
        "reaching the scorer (is producer.py merging train_identity.csv?)"
    )


def test_unknown_category_maps_to_sentinel(contract):
    """A category never seen in training must land on unknown_category, not crash."""
    cat_maps, feature_columns, policy = contract
    tx = {
        "TransactionID": 1, "TransactionDT": 86400, "TransactionAmt": 100.0,
        "ProductCD": "!!!-not-a-real-category-!!!",
        "DeviceInfo": "!!!-also-fake-!!!",
    }
    row = consumer.build_feature_row(tx, feature_columns, cat_maps, policy)
    assert row["ProductCD"] == policy["unknown_category"]
    assert row["DeviceInfo"] == policy["unknown_category"]


def test_missing_fields_do_not_crash(contract):
    """A nearly-empty message must still score — real streams have gaps."""
    cat_maps, feature_columns, policy = contract
    row = consumer.build_feature_row({"TransactionID": 1}, feature_columns,
                                     cat_maps, policy)
    assert len(row) == len(feature_columns)
    assert row["hour"] == 0                      # TransactionDT absent -> 0
    assert row["log_amount"] == pytest.approx(0.0)


def test_feature_order_matches_model_if_available(contract):
    """
    XGBoost is positional. If the registry is reachable, the artifact order must
    match the model's own -- a mismatch means silently wrong scores.
    """
    cat_maps, feature_columns, policy = contract
    try:
        model, _run_id = consumer.load_model()
    except Exception as e:
        pytest.skip(f"registry unreachable: {type(e).__name__}")

    model_features = consumer.get_feature_columns(model)
    assert model_features == list(feature_columns), (
        f"artifact/model feature order differs: "
        f"artifacts={len(feature_columns)} model={len(model_features)}"
    )

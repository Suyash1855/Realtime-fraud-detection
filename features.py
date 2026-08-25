"""
Feature engineering — single source of truth (batch / DataFrame side)
====================================================================
train_fraud_model.py and scripts/select_threshold.py both import from here so
the training-side definitions can never drift apart.

consumer.py necessarily has its own implementation because it works on one dict
at a time rather than a DataFrame. That duplication is guarded by
tests/test_train_serve_parity.py, which asserts both paths produce identical
feature vectors for the same rows. That test is the regression guard for the
whole train/serve contract -- if you change a feature here, change it in
consumer.engineer_features() too and the test will tell you if you got it wrong.
"""

import numpy as np
import pandas as pd

HIGH_RISK_DOMAINS = ["gmail.com", "yahoo.com", "hotmail.com"]

# Columns that must never become features.
ID_COLS = ["TransactionID", "TransactionDT"]
TARGET = "isFraud"

# Engineered features that are kept for scoring compatibility with models already
# in the registry, but dropped from NEW training runs. See engineer_features().
LEGACY_FEATURES = ["addr_mismatch"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Key features that help detect fraud:
    - Transaction amount (high-value txns more likely fraud)
    - Hour of day (late-night txns riskier)
    - Missing/mismatched contact details
    - Email domain (free email = higher risk)
    """
    df = df.copy()

    # Time-based features
    df["hour"] = (df["TransactionDT"] / 3600).astype(int) % 24
    df["day_of_week"] = (df["TransactionDT"] / (3600 * 24)).astype(int) % 7
    df["is_night"] = ((df["hour"] < 6) | (df["hour"] > 22)).astype(int)

    # Amount-based features
    df["log_amount"] = np.log1p(df["TransactionAmt"])
    df["amount_rounded"] = (df["TransactionAmt"] % 1 == 0).astype(int)

    # Address signals.
    #
    # `addr_mismatch = (addr1 != addr2)` is a DEAD feature: addr1 is a billing
    # REGION code and addr2 a billing COUNTRY code, so they are never equal
    # (measured: 0 matches in 50,000 rows). It is constant 1 across the entire
    # training set and the trained model gives it zero splits. It also disagreed
    # with the serving implementation on 5.2% of rows, because pandas evaluates
    # NaN != NaN as True while the consumer required both values to be truthy.
    #
    # Still computed, unchanged, so models already in the registry (whose
    # feature_columns.json includes it) keep scoring identically. It is excluded
    # from LEGACY_FEATURES for new training runs. Do not "fix" its definition
    # in place -- that would silently change what deployed models receive.
    df["addr_mismatch"] = (df["addr1"] != df["addr2"]).astype(int)

    # Replacements that actually carry information:
    df["addr_missing"] = df["addr1"].isna().astype(int)

    # Purchaser vs recipient email domain mismatch -- a genuine fraud signal in
    # this dataset, and what the original feature was probably reaching for.
    both_present = df["P_emaildomain"].notna() & df["R_emaildomain"].notna()
    df["email_mismatch"] = (
        both_present & (df["P_emaildomain"] != df["R_emaildomain"])
    ).astype(int)

    # Email domain risk
    df["risky_email"] = df["P_emaildomain"].isin(HIGH_RISK_DOMAINS).astype(int)

    return df


def encode_categoricals(X: pd.DataFrame, cat_maps: dict, unknown_category: int,
                        null_category_str: str = "nan"):
    """
    Apply persisted categorical mappings to a frame.

    Missing values are normalised to `null_category_str` BEFORE the string cast,
    rather than relying on whatever the source format used to represent null.
    That distinction is not cosmetic:

      training read CSVs, so missing was pandas NaN, and `astype(str)` produced
      the literal "nan" -- which is therefore the key stored in cat_maps.

      Parquet round-trips a missing object value as Python `None`, and
      `str(None)` is "None", which is NOT in cat_maps. It fell through to
      unknown_category instead of the trained missing-value index.

    Measured on the 12k demo set, that mismatch mis-encoded 60% of categorical
    cells, touched every single row, and flipped 1.04% of fraud decisions. The
    encoding must depend on the training contract, never on the file format.
    """
    X = X.copy()
    for col, mapping in cat_maps.items():
        if col in X.columns:
            s = X[col]
            # Covers None, float nan, pd.NA and pd.NaT in one pass.
            s = s.where(s.notna(), null_category_str)
            X[col] = (
                s.astype(str).map(mapping)
                .fillna(unknown_category).astype(int)
            )
    return X


def prepare_matrix(df: pd.DataFrame, feature_columns: list, cat_maps: dict,
                   unknown_category: int, missing_numeric,
                   null_category_str: str = "nan"):
    """
    Full batch scoring path: engineer -> encode -> align columns -> fill missing.

    missing_numeric=None means leave NaN in place so XGBoost can learn a default
    direction per split, which is strictly better than a -999 sentinel for trees.

    null_category_str must match the value in encoding_config.json -- it is the
    string training produced for a missing categorical. See encode_categoricals().
    """
    df = engineer_features(df)
    df = df.drop(columns=ID_COLS + [TARGET], errors="ignore")
    df = encode_categoricals(df, cat_maps, unknown_category, null_category_str)
    X = df.reindex(columns=feature_columns)
    if missing_numeric is not None:
        X = X.fillna(missing_numeric)
    return X.astype(np.float32)

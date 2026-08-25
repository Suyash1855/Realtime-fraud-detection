"""
Build the demo dataset shipped with the deployed app
====================================================
The full IEEE-CIS data is 1.3 GB and cannot be deployed. This carves out a small,
honest sample instead.

Two properties matter and are non-negotiable:

1. Rows are drawn from the CHRONOLOGICAL TEST PERIOD -- the last slice by
   TransactionDT, which the model never trained on. If the demo replayed training
   rows it would show memorised scores, and every metric on the dashboard would be
   a lie.

2. The real fraud base rate (~3.5%) is preserved. Oversampling fraud would make
   the demo livelier and the precision/recall numbers meaningless.

Identity is pre-joined, so the demo stream carries the same feature coverage the
model was trained on.

    python scripts/build_demo_data.py --rows 10000
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TRANSACTIONS = os.getenv("DATASET_PATH", str(ROOT / "data" / "train_transaction.csv"))
IDENTITY = os.getenv("IDENTITY_PATH", str(ROOT / "data" / "train_identity.csv"))
OUT_DIR = ROOT / "demo_data"
TEST_FRACTION = 0.2          # must match TEST_SIZE in train_fraud_model.py
CHUNK = 100_000


def main():
    p = argparse.ArgumentParser(description="Build the deployable demo sample")
    p.add_argument("--rows", type=int, default=10_000,
                   help="How many transactions to ship (default: 10000)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("Locating the chronological test period...")
    dt = pd.read_csv(TRANSACTIONS, usecols=["TransactionDT"])["TransactionDT"]
    n_total = len(dt)
    order = np.argsort(dt.values, kind="stable")
    test_idx = set(order[-int(n_total * TEST_FRACTION):].tolist())
    cutoff = float(dt.values[order[-int(n_total * TEST_FRACTION)]])
    print(f"  {n_total:,} rows total | test period = last {TEST_FRACTION:.0%} "
          f"(TransactionDT >= {cutoff:,.0f})")

    # Reservoir-free approach: we know the test size, so sample positions upfront.
    test_positions = np.array(sorted(test_idx))
    take = min(args.rows, len(test_positions))
    chosen = set(rng.choice(test_positions, size=take, replace=False).tolist())
    print(f"  sampling {take:,} of {len(test_positions):,} test rows")

    print("Reading and joining identity...")
    identity = pd.read_csv(IDENTITY)
    frames, offset = [], 0
    for chunk in pd.read_csv(TRANSACTIONS, chunksize=CHUNK):
        idx = np.arange(offset, offset + len(chunk))
        offset += len(chunk)
        mask = np.fromiter((i in chosen for i in idx), bool, len(idx))
        if mask.any():
            frames.append(chunk[mask])

    df = pd.concat(frames, ignore_index=True)
    df = df.merge(identity, on="TransactionID", how="left")
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    # Downcast float64 -> float32; smaller output, and the model consumes float32
    # anyway.
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype(np.float32)

    # Gzipped CSV, not Parquet. Parquet would pull in pyarrow -- 105 MB of
    # compiled Arrow libraries to read one 12k-row file once at startup. CSV.gz
    # is actually SMALLER here (1.03 MB vs 1.52 MB) and costs ~140 ms extra at
    # boot, which is noise next to loading the model.
    #
    # pandas reads .csv.gz natively via the stdlib gzip module, so the deployed
    # image needs no extra dependency at all.
    out = OUT_DIR / "demo_transactions.csv.gz"
    df.to_csv(out, index=False, compression="gzip")

    fraud_rate = float(df["isFraud"].mean())
    identity_cov = float(df["DeviceType"].notna().mean())
    meta = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "fraud_count": int(df["isFraud"].sum()),
        "fraud_rate": round(fraud_rate, 4),
        "identity_coverage": round(identity_cov, 4),
        "transaction_dt_min": float(df["TransactionDT"].min()),
        "transaction_dt_max": float(df["TransactionDT"].max()),
        "fraud_amount_total": round(float(df.loc[df["isFraud"] == 1, "TransactionAmt"].sum()), 2),
        "amount_total": round(float(df["TransactionAmt"].sum()), 2),
        "source": "IEEE-CIS train_transaction + train_identity",
        "provenance": (
            "Sampled from the chronological test period only (last "
            f"{TEST_FRACTION:.0%} by TransactionDT). The model never saw these "
            "rows during training, so demo metrics are genuine out-of-sample."
        ),
    }
    (OUT_DIR / "demo_metadata.json").write_text(json.dumps(meta, indent=2))

    size_mb = out.stat().st_size / 1024 / 1024
    print(f"\n  wrote {out}  ({size_mb:.1f} MB)")
    print(f"  rows            : {meta['rows']:,}")
    print(f"  fraud           : {meta['fraud_count']:,} ({fraud_rate:.2%})")
    print(f"  identity coverage: {identity_cov:.1%}")
    print(f"  fraud exposure  : ${meta['fraud_amount_total']:,.0f} "
          f"of ${meta['amount_total']:,.0f}")
    print(f"  wrote {OUT_DIR / 'demo_metadata.json'}\n")


if __name__ == "__main__":
    main()

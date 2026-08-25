"""
Decision threshold calibration
==============================
FRAUD_THRESHOLD was hardcoded to 0.5. That is not a meaningful operating point
for this model: scale_pos_weight (~27) inflates the predicted probabilities, so at
0.5 the measured precision is 0.247 -- three quarters of every fraud alert is a
false positive (8,185 legitimate customers blocked to catch 2,687 frauds, measured
on the 118,108-row chronological holdout).

This script scores a holdout set, sweeps the precision/recall curve, and writes
artifacts/threshold_config.json, which consumer.py reads at startup. The operating
point becomes a versioned artifact instead of a magic number in code.

    python scripts/select_threshold.py                       # F1-optimal
    python scripts/select_threshold.py --policy precision --target 0.90
    python scripts/select_threshold.py --policy cost --fp-cost 5 --fn-cost 100
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mlflow.xgboost                                    # noqa: E402
from features import prepare_matrix                      # noqa: E402

ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "artifacts"))
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
MODEL_NAME = os.getenv("MODEL_NAME", "fraud-xgboost")
TRANSACTIONS = os.getenv("DATASET_PATH", "data/train_transaction.csv")
IDENTITY = os.getenv("IDENTITY_PATH", "data/train_identity.csv")
CHUNK = 100_000


def resolve_model(alias, version):
    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.MlflowClient()
    if version:
        uri, resolved = f"models:/{MODEL_NAME}/{version}", version
    else:
        try:
            mv = client.get_model_version_by_alias(MODEL_NAME, alias)
            uri, resolved = f"models:/{MODEL_NAME}@{alias}", mv.version
        except Exception:
            versions = client.search_model_versions(f"name='{MODEL_NAME}'")
            if not versions:
                sys.exit(f"No versions of '{MODEL_NAME}' at {TRACKING_URI}")
            latest = max(versions, key=lambda v: int(v.version))
            uri, resolved = f"models:/{MODEL_NAME}/{latest.version}", latest.version
            print(f"  alias '{alias}' not set — using highest version {resolved}")
    print(f"  loading {uri}")
    return mlflow.xgboost.load_model(uri), resolved


def holdout_indices(split, test_size, random_state):
    """
    Rebuild the same holdout the model was evaluated on.

    'random' reproduces train_test_split(random_state=42, stratify=y) exactly --
    it depends only on n_samples, y and the seed.
    'chronological' takes the last test_size fraction by TransactionDT, which is
    the correct holdout for this dataset and what training now uses.
    """
    if split == "random":
        y = pd.read_csv(TRANSACTIONS, usecols=[  "isFraud"])["isFraud"]
        _, te = train_test_split(
            np.arange(len(y)), test_size=test_size,
            random_state=random_state, stratify=y,
        )
        return set(te.tolist()), len(y)

    dt = pd.read_csv(TRANSACTIONS, usecols=["TransactionDT"])["TransactionDT"]
    order = np.argsort(dt.values, kind="stable")
    n_test = int(len(order) * test_size)
    return set(order[-n_test:].tolist()), len(dt)


def score_holdout(model, keep, feature_columns, cat_maps, cfg):
    identity = pd.read_csv(IDENTITY) if Path(IDENTITY).exists() else None
    scores, ys, amts = [], [], []
    offset = 0

    for chunk in pd.read_csv(TRANSACTIONS, chunksize=CHUNK):
        idx = np.arange(offset, offset + len(chunk))
        offset += len(chunk)
        mask = np.fromiter((i in keep for i in idx), bool, len(idx))
        if not mask.any():
            continue

        d = chunk[mask]
        if identity is not None:
            d = d.merge(identity, on="TransactionID", how="left")

        ys.append(d["isFraud"].values)
        amts.append(d["TransactionAmt"].values)
        X = prepare_matrix(
            d, feature_columns, cat_maps,
            cfg.get("unknown_category", -1), cfg.get("missing_numeric", -999),
            cfg.get("null_category_str", "nan"),
        )
        scores.append(model.predict_proba(X)[:, 1])

    return (np.concatenate(scores), np.concatenate(ys), np.concatenate(amts))


def main():
    p = argparse.ArgumentParser(description="Calibrate the fraud decision threshold")
    p.add_argument("--policy", choices=["f1", "precision", "recall", "cost"], default="f1")
    p.add_argument("--target", type=float, default=0.90,
                   help="Target precision/recall for those policies")
    p.add_argument("--fp-cost", type=float, default=5.0,
                   help="Cost of reviewing/blocking a legitimate transaction")
    p.add_argument("--fn-cost", type=float, default=100.0,
                   help="Cost of a missed fraud")
    p.add_argument("--high-risk-precision", type=float, default=0.92,
                   help="Precision target for the HIGH risk tier")
    p.add_argument("--split", choices=["random", "chronological"], default="random",
                   help="Must match how the model being calibrated was trained")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--alias", default="champion")
    p.add_argument("--version", default=None)
    p.add_argument("--dry-run", action="store_true", help="Print, do not write")
    args = p.parse_args()

    cat_maps = json.loads((ARTIFACT_DIR / "cat_maps.json").read_text())
    feature_columns = json.loads((ARTIFACT_DIR / "feature_columns.json").read_text())
    cfg_path = ARTIFACT_DIR / "encoding_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}

    print("Resolving model...")
    model, version = resolve_model(args.alias, args.version)

    print(f"Rebuilding {args.split} holdout...")
    keep, n_total = holdout_indices(args.split, args.test_size, args.random_state)
    print(f"  {len(keep):,} of {n_total:,} rows held out")

    print("Scoring holdout...")
    s, y, amt = score_holdout(model, keep, feature_columns, cat_maps, cfg)
    print(f"  scored {len(s):,} rows | fraud rate {y.mean():.2%}")
    print(f"  AUC={roc_auc_score(y, s):.4f}  AP={average_precision_score(y, s):.4f}")

    prec, rec, thr = precision_recall_curve(y, s)
    # precision_recall_curve returns len(thr) == len(prec) - 1
    prec, rec = prec[:-1], rec[:-1]

    if args.policy == "f1":
        f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
        i = int(np.nanargmax(f1))
        rationale = f"maximises F1 ({f1[i]:.4f})"
    elif args.policy == "precision":
        ok = np.where(prec >= args.target)[0]
        if not len(ok):
            sys.exit(f"No threshold reaches precision {args.target}")
        i = int(ok[np.argmax(rec[ok])])   # best recall among those meeting precision
        rationale = f"highest recall with precision >= {args.target}"
    elif args.policy == "recall":
        ok = np.where(rec >= args.target)[0]
        if not len(ok):
            sys.exit(f"No threshold reaches recall {args.target}")
        i = int(ok[np.argmax(prec[ok])])
        rationale = f"highest precision with recall >= {args.target}"
    else:  # cost
        n_pos = int(y.sum())
        tp = rec * n_pos
        fp = np.where(prec > 0, tp * (1 - prec) / np.maximum(prec, 1e-12), 0)
        fn = n_pos - tp
        total = fp * args.fp_cost + fn * args.fn_cost
        i = int(np.nanargmin(total))
        rationale = (f"minimises {args.fp_cost}*FP + {args.fn_cost}*FN "
                     f"(cost {total[i]:,.0f})")

    chosen = float(thr[i])

    hr = np.where(prec >= args.high_risk_precision)[0]
    high_risk = float(thr[int(hr[np.argmax(rec[hr])])]) if len(hr) else min(chosen * 1.1, 0.99)
    high_risk = max(high_risk, chosen)

    # Report the neighbourhood so the trade-off is visible, not just the pick.
    print(f"\n{'thr':>7} {'precision':>10} {'recall':>8} {'F1':>7} {'alerts':>8} {'FP':>7}")
    print("-" * 52)
    for t in [0.5, 0.6, 0.7, 0.8, chosen, 0.9, 0.95, 0.99]:
        pred = s >= t
        tp_ = int((pred & (y == 1)).sum()); fp_ = int((pred & (y == 0)).sum())
        pr_ = tp_ / max(tp_ + fp_, 1); rc_ = tp_ / max(int(y.sum()), 1)
        f1_ = 2 * pr_ * rc_ / max(pr_ + rc_, 1e-12)
        tag = "  <-- chosen" if t == chosen else ""
        print(f"{t:7.4f} {pr_:10.3f} {rc_:8.3f} {f1_:7.3f} {tp_+fp_:8d} {fp_:7d}{tag}")

    pred = s >= chosen
    out = {
        "fraud_threshold": round(chosen, 6),
        "high_risk_threshold": round(high_risk, 6),
        "policy": args.policy,
        "rationale": rationale,
        "expected_precision": round(float(prec[i]), 4),
        "expected_recall": round(float(rec[i]), 4),
        "expected_f1": round(float(2 * prec[i] * rec[i] / max(prec[i] + rec[i], 1e-12)), 4),
        "model_name": MODEL_NAME,
        "model_version": str(version),
        "holdout_split": args.split,
        "holdout_rows": int(len(s)),
        "holdout_auc": round(float(roc_auc_score(y, s)), 4),
        "holdout_ap": round(float(average_precision_score(y, s)), 4),
        "fraud_amount_captured": round(float(amt[pred & (y == 1)].sum()), 2),
        "fraud_amount_total": round(float(amt[y == 1].sum()), 2),
        "false_positives": int((pred & (y == 0)).sum()),
    }

    print(f"\nchosen threshold {chosen:.4f} — {rationale}")
    print(f"  ${out['fraud_amount_captured']:,.0f} of ${out['fraud_amount_total']:,.0f} "
          f"fraud captured | {out['false_positives']:,} good customers blocked")

    if args.dry_run:
        print("\n--dry-run: nothing written\n")
        return

    path = ARTIFACT_DIR / "threshold_config.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}\n")


if __name__ == "__main__":
    main()

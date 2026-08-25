"""
Export a self-contained serving bundle for the deployed demo
===========================================================
The deployed app must not talk to MLflow at runtime -- there is no tracking
server in the container, and a network dependency on startup is exactly the kind
of thing that fails during an interview.

So the registry stays the source of truth at BUILD time: this script resolves the
champion alias, pulls the model plus its preprocessing contract, and writes a
frozen bundle into deploy_bundle/. The app loads only from there.

The bundle records which registry version it came from, so a deployed demo can
always be traced back to a specific run.

    python scripts/export_for_deploy.py
    python scripts/export_for_deploy.py --alias champion
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mlflow                                    # noqa: E402
import mlflow.xgboost                            # noqa: E402

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
MODEL_NAME = os.getenv("MODEL_NAME", "fraud-xgboost")
ARTIFACT_DIR = ROOT / os.getenv("ARTIFACT_DIR", "artifacts")
BUNDLE = ROOT / "deploy_bundle"

CONTRACT_FILES = [
    "cat_maps.json",
    "feature_columns.json",
    "encoding_config.json",
    "threshold_config.json",
]


def main():
    p = argparse.ArgumentParser(description="Freeze a serving bundle for deployment")
    p.add_argument("--alias", default="champion")
    p.add_argument("--version", default=None, help="Pin an exact version instead")
    args = p.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.MlflowClient()

    if args.version:
        uri, version = f"models:/{MODEL_NAME}/{args.version}", args.version
    else:
        try:
            mv = client.get_model_version_by_alias(MODEL_NAME, args.alias)
            uri, version = f"models:/{MODEL_NAME}@{args.alias}", mv.version
        except Exception as e:
            sys.exit(
                f"Alias '{args.alias}' is not set on '{MODEL_NAME}' ({e}).\n"
                f"Promote a version first:\n"
                f"  python scripts/promote_model.py --list\n"
                f"  python scripts/promote_model.py --version N --alias {args.alias}"
            )

    print(f"Resolving {uri} -> version {version}")
    model = mlflow.xgboost.load_model(uri)

    BUNDLE.mkdir(exist_ok=True)

    # Save the native booster. Portable across xgboost builds and ~2 MB, versus
    # dragging the whole MLflow model directory into the image.
    model_path = BUNDLE / "model.xgb"
    model.get_booster().save_model(str(model_path))
    print(f"  model     -> {model_path} ({model_path.stat().st_size/1024/1024:.1f} MB)")

    for name in CONTRACT_FILES:
        src = ARTIFACT_DIR / name
        if not src.exists():
            if name == "threshold_config.json":
                sys.exit(
                    f"{src} missing. Run scripts/select_threshold.py, or retrain "
                    f"(training now writes it automatically)."
                )
            sys.exit(f"Required contract file missing: {src}")
        shutil.copy2(src, BUNDLE / name)
        print(f"  contract  -> {BUNDLE / name}")

    # Pull the run's metrics so the dashboard can show real, traceable numbers
    # rather than figures typed into the frontend by hand.
    run_metrics, run_params, run_id = {}, {}, None
    try:
        mv = client.get_model_version(MODEL_NAME, str(version))
        run_id = mv.run_id
        run = client.get_run(run_id)
        run_metrics = {k: float(v) for k, v in run.data.metrics.items()}
        run_params = dict(run.data.params)
    except Exception as e:
        print(f"  (could not read run metrics: {e})")

    feature_columns = json.loads((BUNDLE / "feature_columns.json").read_text())
    manifest = {
        "model_name": MODEL_NAME,
        "model_version": str(version),
        "resolved_from": uri,
        "run_id": run_id,
        "n_features": len(feature_columns),
        "split_strategy": run_params.get("split_strategy", "unknown"),
        "metrics": run_metrics,
        "params": run_params,
        "threshold_config": json.loads((BUNDLE / "threshold_config.json").read_text()),
    }
    (BUNDLE / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  manifest  -> {BUNDLE / 'manifest.json'}")

    print(f"\nBundle ready: {BUNDLE}")
    print(f"  version {version} | {len(feature_columns)} features "
          f"| split={manifest['split_strategy']}")
    if run_metrics:
        auc = run_metrics.get("auc_roc")
        ap = run_metrics.get("avg_precision")
        print(f"  AUC={auc:.4f} AP={ap:.4f}" if auc and ap else f"  metrics: {run_metrics}")


if __name__ == "__main__":
    main()

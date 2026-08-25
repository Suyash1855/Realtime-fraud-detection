"""
Promote a registered model version — the replacement for MLflow stages
=====================================================================
MLflow 3.x removed model stages. consumer.py used to request
`models:/fraud-xgboost/Production`, which always threw because no version was
ever promoted; a bare `except` then fell back to "latest", so production
silently served whatever had been trained most recently.

Aliases are the supported mechanism. This script is the deliberate gate.

    python scripts/promote_model.py --list
    python scripts/promote_model.py --version 5 --alias champion
    python scripts/promote_model.py --version 5 --alias champion --min-auc 0.94
"""

import argparse
import os
import sys

import mlflow

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
MODEL_NAME = os.getenv("MODEL_NAME", "fraud-xgboost")


def main():
    p = argparse.ArgumentParser(description="Promote a model version via alias")
    p.add_argument("--version", help="Registry version number to promote")
    p.add_argument("--alias", default="champion", help="Alias to assign (default: champion)")
    p.add_argument("--model", default=MODEL_NAME, help=f"Model name (default: {MODEL_NAME})")
    p.add_argument("--list", action="store_true", help="List versions and exit")
    p.add_argument(
        "--min-auc", type=float, default=None,
        help="Refuse to promote unless the run's auc_roc metric is at least this",
    )
    args = p.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.MlflowClient()

    versions = sorted(
        client.search_model_versions(f"name='{args.model}'"),
        key=lambda v: int(v.version),
    )
    if not versions:
        sys.exit(f"No versions of '{args.model}' found at {TRACKING_URI}")

    if args.list or not args.version:
        print(f"\nVersions of '{args.model}' at {TRACKING_URI}:\n")
        print(f"  {'ver':<5} {'aliases':<24} {'auc_roc':<9} {'run_id'}")
        print(f"  {'-'*5} {'-'*24} {'-'*9} {'-'*32}")
        for v in versions:
            try:
                auc = client.get_run(v.run_id).data.metrics.get("auc_roc")
                auc_s = f"{auc:.4f}" if auc is not None else "-"
            except Exception:
                auc_s = "-"
            aliases = ",".join(v.aliases) if v.aliases else "-"
            print(f"  {v.version:<5} {aliases:<24} {auc_s:<9} {v.run_id}")
        print()
        if not args.version:
            print("Pass --version N --alias champion to promote.\n")
        return

    target = next((v for v in versions if v.version == str(args.version)), None)
    if target is None:
        sys.exit(f"Version {args.version} not found for '{args.model}'")

    if args.min_auc is not None:
        auc = client.get_run(target.run_id).data.metrics.get("auc_roc")
        if auc is None:
            sys.exit(f"Version {args.version} has no auc_roc metric — refusing to promote")
        if auc < args.min_auc:
            sys.exit(
                f"Refusing to promote: version {args.version} auc_roc={auc:.4f} "
                f"< required {args.min_auc:.4f}"
            )
        print(f"Quality gate passed: auc_roc={auc:.4f} >= {args.min_auc:.4f}")

    # Record what the alias pointed at before, so a rollback is one command.
    previous = next((v.version for v in versions if args.alias in v.aliases), None)

    client.set_registered_model_alias(args.model, args.alias, str(args.version))
    print(f"\n  '{args.alias}' -> {args.model} v{args.version}")
    if previous and previous != str(args.version):
        print(f"  (was v{previous}; roll back with "
              f"--version {previous} --alias {args.alias})")
    print(f"\nconsumer.py will now resolve models:/{args.model}@{args.alias}\n")


if __name__ == "__main__":
    main()

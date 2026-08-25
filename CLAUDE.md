# CLAUDE.md — Fraud-Detect-System

## Project Overview
Real-time credit-card fraud detection on the IEEE-CIS dataset. Kafka streams transactions → consumer scores them with an XGBoost model pulled from the MLflow Model Registry → results land in Redis → FastAPI serves them. Prometheus scrapes the consumer and the API.

## Commands
- **Infra:** `docker compose up -d` (zookeeper 2181, kafka 9092, redis 6379, redis-insight 8001, kafka-ui 8080)
- **MLflow server:** `mlflow server --host 0.0.0.0 --port 5001` (**must be 5001**, not the default 5000 — the code hardcodes it)
- **Train + register:** `python train_fraud_model.py`
- **Stream:** `python producer.py --rate 100 --limit 5000`
- **Score:** `python consumer.py` (Prometheus on 8002)
- **API:** `uvicorn API.main:app --reload --port 8000` → `/stats`, `/frauds`, `/transaction/{tx_id}`, `/metrics`

## Autonomy / Permissions (IMPORTANT)
- **Do not ask for permission or confirmation to run the work.** Run the training, the producer/consumer, `docker compose`, `redis-cli`, `curl`, `grep`/`sed`/`awk`/`python3` one-liners, file edits, and installs the task needs and finish the job in one pass.
- Do not stop to ask "should I proceed?", "do you want me to continue?", or offer a plan for approval when the request is already clear — just complete it and report the result.
- Only pause for the user when the action is genuinely destructive or outward-facing (`git push`, force-push/reset on shared branches, deleting whole directories, deleting the `data/` CSVs or `mlruns/` history, publishing, deploying, or anything that touches production data).
- The actual prompt suppression lives in `.claude/settings.local.json`: `skipDangerousModePermissionPrompt: true` + `permissions.defaultMode: "bypassPermissions"` + bare tool-level `Bash`/`Read`/`Edit`/`Write` allows. If prompts still appear, restart the session so the settings file is re-read — and check that the session was started from inside this project folder, since project settings only load for the launch directory.

## Review Loop (IMPORTANT)
- **Assume your output gets reviewed.** Leave the work in a reviewable state: complete, self-consistent, verified against a real run (`python consumer.py` scoring a live message, or `curl localhost:8000/stats` returning non-zero counts), and free of debug leftovers, dead code, or half-applied refactors.
- State plainly what you changed and what you verified — if something is unverified or intentionally left out, say so instead of implying it is done.

## Pipeline Invariants (IMPORTANT)
Break these and the model scores silently wrong — no error, just bad predictions.

1. **`engineer_features()` must stay identical in `consumer.py` and `train_fraud_model.py`.** Both derive `hour`, `day_of_week`, `is_night`, `log_amount`, `amount_rounded`, `addr_mismatch`, `risky_email`. Change one, change the other, then retrain.
2. **Encoding sentinels are fixed:** `UNKNOWN_CATEGORY = -1`, `MISSING_NUMERIC = -999`, `NULL_CATEGORY_STR = "nan"` (training does `astype(str)`, so NaN becomes the literal string `"nan"`).
3. **XGBoost is positional.** `artifacts/feature_columns.json` order is authoritative; `consumer.py` cross-checks it against `model.feature_names_in_` and prefers the model's order on mismatch. Retraining means re-copying the artifacts.
4. **Dropped at inference:** `TransactionID`, `TransactionDT`, `isFraud`, `ingested_at`. `isFraud` is carried as `true_label` for offline eval only — never as a feature.
5. Thresholds: `FRAUD_THRESHOLD = 0.5`, `HIGH_RISK_THRESHOLD = 0.8`. Redis predictions expire after `REDIS_TTL_SECONDS = 3600`.
6. `data/` holds the raw IEEE-CIS CSVs — read them, never rewrite them.

## Layout
| Piece | File |
|---|---|
| Training (4 runs: logistic baseline, XGBoost main, XGBoost+SMOTE, IsolationForest) | `train_fraud_model.py` |
| Stream producer, keyed by `TransactionID` | `producer.py` |
| Scorer, consumer group `fraud-scorer` | `consumer.py` |
| Read API | `API/main.py` |
| Infra | `docker-compose.yml` |
| Persisted encoders | `artifacts/{cat_maps,feature_columns,encoding_config}.json` |

## Conventions
- Plain scripts, no package layout. Module-level constants in caps at the top of each file; no config framework.
- `logging` with the `[PRODUCER]` / `[CONSUMER]` prefix format already set up in each file — not `print()` (except the training summary block).
- Section banners (`# ─── NAME ───`) separate concerns within a file. Match them.
- Per-message errors in the consumer loop are logged and skipped; one bad transaction must never kill the consumer.

## Code Comments (IMPORTANT)
- **Do not add large comments over or for the code.** Default to writing no comments at all.
- Well-named identifiers should explain what the code does — comments should not duplicate that.
- Only add a short single-line comment when the **why** is genuinely non-obvious (a hidden constraint, a workaround for a specific bug, a subtle invariant a reader would otherwise miss).
- Never write multi-line comment blocks, multi-paragraph docstrings, or "what the code does" narration above functions.
- Do not leave behind notes about the current task, PR, or removed code (e.g. `// added for X`, `// previously did Y`) — that context belongs in commit messages.

## Known Rough Edges
- `API/main.py` calls `json.loads` in `get_transaction()` without importing `json` — that endpoint raises until the import is added.
- `MODEL_STAGE = "Production"` in `consumer.py` falls back to `models:/fraud-xgboost/latest` when nothing is promoted; that's the normal dev path.
- `datetime.utcnow()` is deprecated on modern Python; used in both `producer.py` and `consumer.py`.

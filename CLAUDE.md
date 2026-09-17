# CLAUDE.md — Fraud-Detect-System

## Project Overview
Real-time credit-card fraud detection on the IEEE-CIS dataset. Kafka streams transactions → consumer scores them with an XGBoost model pulled from the MLflow Model Registry → results land in Redis → FastAPI serves them. Prometheus scrapes the consumer and the API.

## Commands
Secrets come from `.env`, which is gitignored. `cp .env.example .env` and fill it in first — compose fails to start rather than falling back to defaults.

- **Whole stack:** `docker compose up -d` — zookeeper, kafka, redis, kafka-ui, prometheus, grafana, postgres, minio, mlflow, `fraud-consumer`, `fraud-api`. Every port binds to `${BIND_ADDRESS}` (default `127.0.0.1`) except the API on 8200.
- **Scale the scorer:** `docker compose up -d --scale fraud-consumer=6` (6 topic partitions is the ceiling)
- **Replay the dataset:** `docker compose --profile load run --rm fraud-producer` (needs `data/`, mounted read-only)
- **MLflow:** http://localhost:5001 — Postgres backend, MinIO artifacts. Do **not** start a second `mlflow server` by hand; it would shadow this one with an empty file store.
- **Train + register:** `MLFLOW_TRACKING_URI=http://localhost:5001 python train_fraud_model.py`
- **API:** http://localhost:8200. Every endpoint except `/health` and `/metrics` needs `X-API-Key: $API_KEY`.
  - `POST /transactions`, `POST /transactions/batch` — ingest onto the topic
  - `GET /stats`, `/frauds`, `/transaction/{tx_id}`, `/health`, `/metrics`
- **Run locally without compose:** set `KAFKA_BROKER`, `REDIS_HOST`, `REDIS_PASSWORD`, `MLFLOW_TRACKING_URI`, `API_KEY`, then `python consumer.py` / `uvicorn API.main:app --port 8000`.

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
3. **XGBoost is positional.** `consumer.py` downloads `preprocessing/` from the MLflow run that produced the model, so the contract can never be stale relative to it. It cross-checks `feature_columns.json` against `model.feature_names_in_` and prefers the model's order on mismatch. Nothing cross-checks `cat_maps.json` — a wrong one just encodes categories to different integers, silently. Never point serving at a hand-copied `artifacts/` directory.
4. **Dropped at inference:** `TransactionID`, `TransactionDT`, `isFraud`, `ingested_at`. `isFraud` is carried as `true_label` for offline eval only — never as a feature.
5. Thresholds come from `threshold_config.json` in the model's run (currently 0.7933 / 0.9173), falling back to env then to the measured defaults in `consumer.py`. 0.5 is **not** a sane default — `scale_pos_weight` ≈ 27.5 inflates the probabilities. Redis predictions expire after `REDIS_TTL_SECONDS = 3600`.
6. `data/` holds the raw IEEE-CIS CSVs — read them, never rewrite them.

## Layout
| Piece | File |
|---|---|
| Training (4 runs: logistic baseline, XGBoost main, XGBoost+SMOTE, IsolationForest) | `train_fraud_model.py` |
| Stream producer, keyed by `TransactionID` | `producer.py` |
| Scorer, consumer group `fraud-scorer` | `consumer.py` |
| Ingest + read API | `API/main.py` |
| Infra | `docker-compose.yml`, `.env` (from `.env.example`) |
| Pipeline image (producer/consumer/API) | `Dockerfile.pipeline` |
| MLflow image (adds psycopg2 + boto3) | `Dockerfile.mlflow` |
| Demo console image | `Dockerfile` + `app/` + `deploy_bundle/` |
| Serving contract | MLflow run artifacts under `preprocessing/`; `artifacts/` is local scratch |
| Store migration (file store → Postgres/MinIO) | `scripts/migrate_mlflow_store.py` |
| CI | `.github/workflows/ci.yml` |

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
- **Single-broker Kafka, replication factor 1, Zookeeper mode.** No redundancy; losing the broker loses undelivered messages. Real HA means managed Kafka, not more config here.
- **Kafka and Redis speak plaintext with no auth inside the docker network.** The boundary is `BIND_ADDRESS=127.0.0.1`, not authentication — do not set it to `0.0.0.0` on an untrusted network.
- **Postgres and MinIO volumes are local.** They survive container restarts, not host loss. Rotating `POSTGRES_PASSWORD` after first init needs `ALTER USER` inside the DB; the env var only applies to an empty volume.
- **Ingestion is not idempotent.** Re-POSTing a `TransactionID` re-scores it, overwrites the Redis key and double-counts `stats:*`.
- **`TransactionDT` is a dataset-specific integer offset and is required on ingest.** A real gateway sends a wall-clock timestamp; there is no mapping yet.
- **The API takes ~45s to start when Kafka is unreachable** — kafka-python retries bootstrap and `max_block_ms` does not bound the constructor. Hence `start_period: 75s`.
- `train_fraud_model.py` and `scripts/export_for_deploy.py` still default to `http://localhost:5001`, which is correct only from the host.

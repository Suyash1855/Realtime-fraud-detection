# Real-Time Fraud Detection System

Streaming card-fraud detection on the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/data)
dataset. Kafka carries the transaction stream, an XGBoost model served from the
MLflow Model Registry scores each one, Redis holds the results, and FastAPI
exposes them. Prometheus and Grafana watch the whole thing.

```
 train_transaction.csv ─┐
                        ├─► train_fraud_model.py ─► MLflow Registry ─┐
 train_identity.csv  ───┘         │                                  │
                                  └─► artifacts/  ────────────┐      │
                                      (train/serve contract)  │      │
                                                              ▼      ▼
 producer.py ──► Kafka "transactions" ──► consumer.py ──► Redis ──► API/main.py
   (replays          6 partitions          batch score       │        /stats
    the CSV)                               + threshold       │        /frauds
                                                             │        /transaction/{id}
                                           Prometheus :9090 ─┴──► Grafana :3000
```

---

> **Just want to run it?** See **[RUNBOOK.md](RUNBOOK.md)** for
> step-by-step start instructions, a port map, and troubleshooting.
>
> **Putting the full pipeline on a server?** See **[DEPLOY.md](DEPLOY.md)**.
> **Deploying or updating the demo console?** See **[DEPLOY_DEMO_CONSOLE.md](DEPLOY_DEMO_CONSOLE.md)**.
> **How the full pipeline was deployed to Hetzner, and why?** See **[DEPLOYMENT_STORY.md](DEPLOYMENT_STORY.md)** ([PDF](DEPLOYMENT_STORY.pdf)).

---

## Quick start

```bash
pip install -r requirements.txt

# 1. Infrastructure (Kafka, Redis, Prometheus, Grafana, Kafka UI)
docker compose up -d

# 2. MLflow tracking server.
#    The --artifacts-destination and --serve-artifacts flags are REQUIRED.
#    Plain `mlflow ui --port 5001` cannot serve model artifacts and every
#    model load fails with a 500.
mlflow server \
  --backend-store-uri "file://$PWD/mlruns" \
  --artifacts-destination "file://$PWD/mlartifacts" \
  --serve-artifacts --host 127.0.0.1 --port 5001

# 3. Train (writes artifacts/, registers a new model version)
python train_fraud_model.py

# 4. Promote a version — registering does NOT deploy it
python scripts/promote_model.py --list
python scripts/promote_model.py --version 6 --alias champion --min-auc 0.85

# 5. Stream and score
python producer.py --rate 100 --limit 5000
python consumer.py

# 6. Read API
uvicorn API.main:app --port 8000
```

| Service | URL |
|---|---|
| FastAPI | http://localhost:8000/docs |
| MLflow | http://localhost:5001 |
| Grafana | http://localhost:3000 (anonymous viewer) |
| Prometheus | http://localhost:9090 |
| Kafka UI | http://localhost:8080 |
| RedisInsight | http://localhost:8001 |
| Consumer metrics | http://localhost:8002/metrics |

> Port 8001 belongs to RedisInsight. The consumer's metrics server therefore
> uses **8002** — do not move it to 8001, it cannot bind there.

---

## The demo console

A deployable single-container app: FastAPI serving both the JSON API and a React
frontend. It runs the **real** model and the **real** feature pipeline; only the
transport is swapped — an in-process clock instead of Kafka, memory instead of
Redis. The UI says so explicitly on its overview screen.

```bash
# 1. Freeze the champion model + contract into deploy_bundle/  (~2 MB)
python scripts/export_for_deploy.py --alias champion

# 2. Carve a deployable sample out of the chronological TEST period  (~1.0 MB)
#    Written as gzipped CSV, not Parquet -- Parquet needs pyarrow (105 MB of
#    compiled Arrow libs) to read one small file once.
python scripts/build_demo_data.py --rows 12000

# 3. Build the frontend
cd web && npm install && npm run build && cd ..

# 4. Serve both on one port
uvicorn app.main:app --port 8000     # → http://localhost:8000
```

Three screens, deep-linkable via hash routes:

| Route | What it shows |
|---|---|
| `#/overview` | What the system is, the honest metrics, the leakage comparison |
| `#/live` | Streaming scoring, live throughput/latency, alert queue, threshold slider |
| `#/analytics` | PR curve, score separation, confusion matrix, interactive cost model |

`#/live?autostart=1` opens straight into a running simulation.

Clicking any row in the alert queue opens a SHAP breakdown of *why* that
transaction scored the way it did, computed with XGBoost's exact tree
contributions — so the deployed image needs no `shap` dependency.

**Deploy:**

```bash
fly launch --no-deploy   # first time
fly deploy
```

`min_machines_running = 1` in `fly.toml` keeps it warm — a scale-to-zero cold
start is 20-30s, which is unacceptable if someone opens the link during an
interview.

The image is 691 MB and the container idles at ~170 MB RAM, booting in ~3s.

**Performance note.** All 12,000 demo scores are computed once at startup, so the
threshold slider is an `O(log n)` binary search over a pre-sorted array rather
than re-inference. The live stream still scores in real time, so the latency shown
is genuinely measured.

---

## The train/serve contract

The single most important idea in this repo. Training writes four files into
`artifacts/`, and the consumer reads them at startup instead of hardcoding
anything:

| File | Purpose |
|---|---|
| `cat_maps.json` | category → integer, per categorical column |
| `feature_columns.json` | the exact feature **order** (XGBoost is positional) |
| `encoding_config.json` | sentinels: unknown category, missing numeric, null string |
| `threshold_config.json` | the decision threshold and what it is expected to achieve |

If retraining changes the encoding policy, serving follows automatically. The
consumer additionally cross-checks the feature list against the model's own
`feature_names_in_` and prefers the model's ordering, because a silent order
mismatch produces confident, wrong scores.

`tests/test_train_serve_parity.py` asserts that the batch path (`features.py`)
and the streaming path (`consumer.py`) produce **identical feature vectors** for
the same rows. Run it after touching any feature:

```bash
python -m pytest tests/ -v
```

---

## Choosing the decision threshold

`0.5` is not a meaningful operating point. `scale_pos_weight` (~27.5) inflates the
predicted probabilities; at 0.5 precision measured **0.286**, meaning 71% of all
fraud alerts were false positives.

Measured on the full 118,108-row chronological holdout (model v6):

| threshold | precision | recall | alerts | false positives |
|---|---|---|---|---|
| 0.50 | 0.247 | 0.661 | 10,872 | 8,185 |
| 0.70 | 0.416 | 0.504 | 4,927 | 2,878 |
| 0.83 (F1-optimal on test) | 0.633 | 0.394 | 2,534 | 931 |
| 0.90 | 0.759 | 0.336 | 1,796 | 432 |
| 0.95 | 0.852 | 0.275 | 1,312 | 194 |

At the default 0.5, three quarters of every alert is a false positive: 8,185
legitimate customers blocked to catch 2,687 frauds.

The **deployed** threshold is `0.7933`, chosen by maximising F1 on the *validation*
split during training — not on test. Picking it on test would be the same category
of mistake as the leaky split.

Same model under different cost assumptions:

| Assumption | Optimal threshold | Fraud $ caught | Customers blocked |
|---|---|---|---|
| $5 false alarm / $100 missed fraud | 0.38 | $462,408 | 13,707 |
| $25 false alarm / $100 missed fraud | 0.75 | $253,058 | 2,006 |
| "90% of alerts must be correct" | 0.98 | $67,632 | 92 |

(Total fraud exposure in that holdout: $609,934.)

Training picks the F1-optimal point on the validation split automatically and
writes it to `artifacts/threshold_config.json`. To
choose a different policy:

```bash
python scripts/select_threshold.py --policy precision --target 0.90
python scripts/select_threshold.py --policy cost --fp-cost 5 --fn-cost 100
```

The right answer depends on your cost ratio — what a chargeback costs versus a
manual review versus blocking a good customer. `--policy cost` takes those
numbers directly.

---

## Configuration

Everything is environment-driven; the defaults are what you want locally.

| Variable | Default | Used by |
|---|---|---|
| `KAFKA_BROKER` | `localhost:9092` | producer, consumer |
| `KAFKA_TOPIC` | `transactions` | producer, consumer |
| `CONSUMER_GROUP` | `fraud-scorer` | consumer |
| `DLQ_TOPIC` | `transactions-dlq` | consumer |
| `REDIS_HOST` / `REDIS_PORT` | `localhost` / `6379` | consumer, API |
| `REDIS_TTL_SECONDS` | `3600` | consumer |
| `MLFLOW_TRACKING_URI` | `http://localhost:5001` | training, consumer, scripts |
| `MODEL_NAME` | `fraud-xgboost` | everywhere |
| `MODEL_ALIAS` | `champion` | consumer |
| `MODEL_VERSION` | *(unset)* | consumer — pins an exact version |
| `METRICS_PORT` | `8002` | consumer |
| `BATCH_MAX_RECORDS` | `256` | consumer |
| `SPLIT_STRATEGY` | `chronological` | training |
| `FRAUD_THRESHOLD` | from artifact | consumer (fallback only) |

Inside Docker, set `REDIS_HOST=redis-stack` and `KAFKA_BROKER=kafka:29092`.

---

## Model promotion

Registering a model does not deploy it. MLflow 3 removed stages, so promotion
uses an **alias**:

```bash
python scripts/promote_model.py --list
python scripts/promote_model.py --version 7 --alias champion --min-auc 0.85
```

The consumer resolves `models:/fraud-xgboost@champion`. Rollback is the same
command pointed at the previous version. If the alias is unset the consumer falls
back to the highest version number and logs a loud warning — that is a safety net,
not a deployment strategy.

---

## Layout

```
train_fraud_model.py   training: LR baseline → XGBoost → Isolation Forest
features.py            feature engineering, batch side (single source of truth)
producer.py            replays the CSV into Kafka (joins identity)
consumer.py            batch scoring, Redis writes, Prometheus metrics
API/main.py            FastAPI read API
API/prometheus.yml     scrape config
scripts/promote_model.py    alias-based promotion gate
scripts/select_threshold.py threshold calibration
tests/                 train/serve parity tests
artifacts/             the train/serve contract (generated)
PROJECT_ANALYSIS.md    full codebase analysis and findings
```

---

## Known limitations

* **Metrics depend heavily on the split.** Model v5 (random split) reported
  AUC 0.9469; v6 (chronological, currently serving) reports **0.8932**. The
  0.0537 gap was leakage, not skill. `SPLIT_STRATEGY=random` reproduces the
  optimistic number if you want to see it for yourself.
* **Isolation Forest is not wired into serving.** It trains and logs, but the
  consumer only uses XGBoost.
* **The API has no authentication.** It exposes fraud scores by transaction ID
  and must not be public as-is.
* **`producer.py --no-label`** strips `isFraud`; without it the ground-truth label
  travels in-band, which is fine for offline evaluation and wrong for production.
* **Zookeeper-backed Kafka** is deprecated upstream; KRaft would remove a service.

See `PROJECT_ANALYSIS.md` for the full findings list with file/line references.

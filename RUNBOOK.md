# Runbook — how to start this locally

Every command here was run and verified on this machine (macOS, Python 3.9.6,
Docker Desktop, Node 20). Copy-paste them as written.

There are two ways to run this project. Pick based on what you want:

| | **Path A — Demo console** | **Path B — Full pipeline** |
|---|---|---|
| What it is | One process. The interview artifact. | Kafka + Redis + MLflow + Prometheus + Grafana |
| Start time | ~5 seconds | ~2 minutes |
| Commands | 1 | ~6 across 4 terminals |
| Use when | Showing someone the project | Proving the streaming architecture |

---

## Before anything: a PATH note

`mlflow` and `uvicorn` are installed to `~/Library/Python/3.9/bin`, which is **not
on your PATH**. Every command below therefore uses `python3 -m <tool>`, which
always works.

If you'd rather type `mlflow` and `uvicorn` directly, add this to `~/.zshrc`:

```bash
export PATH="$HOME/Library/Python/3.9/bin:$PATH"
```

Then `source ~/.zshrc`. The rest of this runbook still works either way.

---

# Path A — Demo console (start here)

The self-contained app: FastAPI serving the API and the React frontend together.
No Kafka, no Redis, no MLflow needed — the model and its preprocessing contract
are frozen into `deploy_bundle/`.

## Start it

```bash
cd /Users/apple/Desktop/Fraud-Detect-System
python3 -m uvicorn app.main:app --port 8000
```

Open **http://localhost:8000**

You should see startup logs like:

```
[APP] Loading serving bundle from .../deploy_bundle
[APP] Scoring engine ready | model v6 | 439 features | threshold 0.7933
[APP] Loaded 12000 demo transactions
[APP] Precomputed 12000 scores | fraud=393
```

## What to click

1. **Overview** — the leakage story (random split 0.9469 vs chronological 0.8932)
2. **Start real-time simulation** → jumps to the live console and begins streaming
3. Click any row in the **Alert queue** → SHAP breakdown of why it scored that way
4. **Analytics** tab → drag the threshold slider, then edit the cost model inputs

Deep links: `#/overview`, `#/live`, `#/analytics`, and `#/live?autostart=1` to
open straight into a running simulation.

## Stop it

`Ctrl+C` in that terminal.

## Rebuild (only if you change something)

```bash
# Frontend changed
cd web && npm install && npm run build && cd ..

# Promoted a new model version
python3 scripts/export_for_deploy.py --alias champion

# Want a different demo sample
python3 scripts/build_demo_data.py --rows 12000
```

All three outputs are already committed to disk, so a fresh clone only needs
`npm install && npm run build`.

---

# Path B — Full pipeline

Four terminals. Do them in order; each waits on the previous.

## Step 1 — Infrastructure

```bash
cd /Users/apple/Desktop/Fraud-Detect-System
docker compose up -d
```

Starts Zookeeper, Kafka, Redis Stack, kafka-ui, Prometheus, Grafana, and a
`kafka-init` job that creates the topics with 6 partitions.

Verify:

```bash
docker compose ps
docker exec kafka kafka-topics --bootstrap-server kafka:29092 --list
```

Expect `transactions`, `transactions-dlq`, `__consumer_offsets`. If `kafka-init`
exited non-zero, re-run just that job:

```bash
docker compose up kafka-init --force-recreate
```

> **Redis data is ephemeral.** No named volume is mounted, so `docker compose
> down` or recreating the container wipes all predictions and `stats:*` counters.
> That is usually what you want between demo runs.

## Step 2 — MLflow tracking server

**Terminal 1** (leave running):

```bash
cd /Users/apple/Desktop/Fraud-Detect-System
python3 -m mlflow server \
  --backend-store-uri "file://$PWD/mlruns" \
  --artifacts-destination "file://$PWD/mlartifacts" \
  --serve-artifacts \
  --host 127.0.0.1 --port 5001
```

**`--serve-artifacts` and `--artifacts-destination` are not optional.** Plain
`mlflow ui --port 5001` starts fine and serves the web UI, but every attempt to
download a model artifact returns HTTP 500 — so `consumer.py` cannot load a model
at all. This failure is silent until something tries.

Verify artifact serving actually works (not just the UI):

```bash
curl -s -o /dev/null -w "%{http_code}\n" \
  "http://127.0.0.1:5001/api/2.0/mlflow-artifacts/artifacts/661888854093502547/models/m-a51f1a228b0a455c949a6022787e5f86/artifacts/MLmodel"
```

Must print `200`. If it prints `500`, the flags are wrong.

Open http://localhost:5001 to browse runs.

## Step 3 — Consumer (the scorer)

**Terminal 2**:

```bash
cd /Users/apple/Desktop/Fraud-Detect-System
python3 consumer.py
```

Expected startup:

```
[CONSUMER] Loading alias 'champion' -> version 6
[CONSUMER] Loaded model: fraud-xgboost v6
[CONSUMER] Prometheus metrics on :8002/metrics
[CONSUMER] Thresholds from artifact | fraud=0.7933 high_risk=0.9173
[CONSUMER] Subscribed to topic 'transactions' as group 'fraud-scorer'
[CONSUMER] Listening for transactions...
```

It idles indefinitely until the producer sends something. Stop with `Ctrl+C` —
it drains the current batch, commits offsets, and prints a summary.

## Step 4 — Producer (the transaction stream)

**Terminal 3**:

```bash
cd /Users/apple/Desktop/Fraud-Detect-System
python3 producer.py --rate 200 --limit 5000
```

| Flag | Meaning |
|---|---|
| `--rate N` | transactions per second |
| `--limit N` | stop after N (omit to stream all 590,540) |
| `--no-label` | strip `isFraud` from the payload, like a real production stream |

Watch the consumer terminal light up with `🚨 FRAUD DETECTED` lines.

## Step 5 — Read API

**Terminal 4**:

```bash
cd /Users/apple/Desktop/Fraud-Detect-System
python3 -m uvicorn API.main:app --host 127.0.0.1 --port 8200
```

> Port **8200**, not 8000 (the demo console owns that) and not 8100 — another
> project on this machine binds `[::1]:8100` on IPv6, so `localhost:8100` is
> ambiguous and resolves to the wrong app.

```bash
curl http://127.0.0.1:8200/stats
curl "http://127.0.0.1:8200/frauds?limit=5"
curl http://127.0.0.1:8200/transaction/<some-id>
```

Interactive docs: http://127.0.0.1:8200/docs

---

## Port map

| Port | Service | Started by |
|---|---|---|
| **8000** | **Demo console** | `uvicorn app.main:app` |
| 5001 | MLflow UI + registry | `python3 -m mlflow server` |
| 8200 | Read API | `uvicorn API.main:app` |
| 8002 | Consumer Prometheus metrics | `consumer.py` (auto) |
| 9090 | Prometheus | docker compose |
| 3000 | Grafana (anonymous viewer) | docker compose |
| 8080 | kafka-ui | docker compose |
| 8001 | RedisInsight | docker compose |
| 9092 | Kafka (host listener) | docker compose |
| 6379 | Redis | docker compose |
| 2181 | Zookeeper | docker compose |

**Ports to avoid:** `8001` belongs to RedisInsight — do not move the consumer's
metrics there, it cannot bind. `8100` is taken by an unrelated local project.

---

## Verify the whole thing is healthy

```bash
# Prometheus scrape targets — all three should be "up"
curl -s http://localhost:9090/api/v1/targets \
  | python3 -c "import json,sys; [print(f\"{t['labels']['job']:16s} {t['health']}\") for t in json.load(sys.stdin)['data']['activeTargets']]"

# Consumer metrics are actually being collected
curl -s 'http://localhost:9090/api/v1/query?query=transactions_processed_total'

# p95 scoring latency.
# Only meaningful WHILE the producer is running: rate() over a 5m window is zero
# once traffic stops, and histogram_quantile of all-zero rates is NaN. A real
# number here (e.g. 7.1) confirms the histogram buckets are in milliseconds --
# they used to default to seconds, which sent every sample into +Inf and made
# every quantile useless.
curl -s --get http://localhost:9090/api/v1/query \
  --data-urlencode 'query=histogram_quantile(0.95, rate(model_latency_milliseconds_bucket[5m]))'

# Kafka reachable from inside the docker network
curl -s http://localhost:8080/api/clusters | python3 -m json.tool

# Redis eviction policy must be volatile-lru, not allkeys-lru
docker exec redis-stack redis-cli CONFIG GET maxmemory-policy

# Train/serve parity — the regression guard for the whole contract
python3 -m pytest tests/ -v
```

The `fastapi` and `fraud-consumer` Prometheus targets read `down` whenever those
processes aren't running. That's expected, not a fault.

---

## Retraining and promoting

```bash
# Train (chronological split; ~10 min on the full 590k rows)
python3 train_fraud_model.py

# Faster iteration: skip the SMOTE comparison run
RUN_SMOTE=0 python3 train_fraud_model.py

# Reproduce the old optimistic (leaky) numbers for comparison
SPLIT_STRATEGY=random python3 train_fraud_model.py
```

Training **registers** a version but does not deploy it. Promotion is deliberate:

```bash
python3 scripts/promote_model.py --list
python3 scripts/promote_model.py --version 7 --alias champion --min-auc 0.85
```

`--min-auc` refuses the promotion if the run's AUC is below the bar.

Then refresh what the demo console serves:

```bash
python3 scripts/export_for_deploy.py --alias champion
```

Re-tune the threshold under a different policy:

```bash
python3 scripts/select_threshold.py --policy precision --target 0.90
python3 scripts/select_threshold.py --policy cost --fp-cost 5 --fn-cost 100
```

---

## Shutting down

```bash
# Host processes (Ctrl+C in each terminal, or:)
pkill -f "uvicorn app.main:app"     # demo console
pkill -f "uvicorn API.main:app"     # read API
pkill -f consumer.py
pkill -f producer.py
pkill -f "mlflow server"

# Containers — keeps volumes
docker compose stop

# Containers + network + Prometheus/Grafana volumes
docker compose down -v
```

---

## Troubleshooting

**`consumer.py`: "Alias 'champion' not set — falling back to highest version"**
No version is promoted. Run `python3 scripts/promote_model.py --version N --alias
champion`. The fallback still works but is not a deployment gate.

**`consumer.py`: MlflowException 500 downloading artifacts**
MLflow was started without `--serve-artifacts --artifacts-destination`. See Step 2.

**`consumer.py`: `OSError: [Errno 48] Address already in use`**
Something holds the metrics port. Since the fix this is non-fatal — the consumer
logs and keeps scoring without metrics. To restore them:
`METRICS_PORT=8003 python3 consumer.py` (and update `API/prometheus.yml`).

**Consumer starts but never scores anything**
The producer isn't running, or it already streamed everything and the consumer
group has committed past it. Use a fresh group:
`CONSUMER_GROUP=fresh-$(date +%s) python3 consumer.py`

**`docker compose up` → "container name already in use"**
An orphaned container from outside the compose project. Inspect, then remove:
`docker inspect <name> --format '{{.State.Status}}'` then `docker rm <name>`.
Do **not** add a `name:` key to `docker-compose.yml` — the project name must stay
`fraud-detect-system` (the directory name) to match existing containers.

**kafka-ui shows the cluster offline**
Kafka needs both listeners. `KAFKA_ADVERTISED_LISTENERS` must include
`INTERNAL://kafka:29092` and kafka-ui must bootstrap `kafka:29092`, not
`localhost:9092`.

**`/stats` numbers look far too high**
`auto_offset_reset="earliest"` means a new consumer group re-reads the whole
topic, double-counting. Reset with `docker exec redis-stack redis-cli FLUSHDB`.

**Grafana has no dashboards**
None are provisioned yet. The Prometheus datasource *is* pre-wired, so use
**Explore** and query e.g. `rate(transactions_processed_total[1m])`.

**Frontend changes don't show up**
The FastAPI app serves the built bundle from `web/dist`, not your source. Run
`cd web && npm run build`. For hot reload during development, run
`npm run dev` (port 5173) alongside the API on 8000 — Vite proxies `/api` across.

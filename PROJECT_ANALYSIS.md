# Fraud-Detect-System — Codebase Analysis

> Reference document. Every number below was measured on this machine against the
> real data and the real registered model — not estimated. Written 2026-08-11.
>
> **Fixes were applied on 2026-08-12.** See §9 for what was changed, what was
> deliberately left, and three additional issues found while fixing. Findings
> below are annotated `[FIXED]`, `[FIXED — needs retrain]`, or `[OPEN]`.

---

## 1. What this project is

A real-time card-fraud detection pipeline built on the **IEEE-CIS Fraud Detection**
Kaggle dataset. Four moving parts:

```
data/train_transaction.csv (652 MB)          artifacts/
data/train_identity.csv    (25 MB)           ├── cat_maps.json         (31 cat cols → int)
        │                                    ├── feature_columns.json  (438 names, ordered)
        │  merge on TransactionID            └── encoding_config.json  (sentinels)
        ▼                                              │
┌───────────────────────┐                              │ written at train time
│ train_fraud_model.py  │  LR baseline → XGBoost → IsolationForest
│                       │  logs params/metrics/SHAP to MLflow @ :5001
│                       │  registers "fraud-xgboost" in Model Registry
└───────────────────────┘                              │
                                                       │ read at serve time
┌───────────────────────┐   Kafka topic     ┌───────────▼───────────┐
│     producer.py       │  "transactions"   │     consumer.py       │
│ replays CSV rows as   │ ────────────────▶ │ engineer → encode →   │
│ a live tx stream      │   (1 partition)   │ predict_proba → Redis │
│ --rate N --limit N    │                   │ + Prometheus :8002    │
└───────────────────────┘                   └───────────┬───────────┘
                                                        │ prediction:<id>  (TTL 1h)
                                                        │ flagged_transactions (zset)
                                                        │ stats:* counters
┌───────────────────────┐                   ┌───────────▼───────────┐
│  docker-compose.yml   │                   │      API/main.py      │
│ zookeeper, kafka,     │                   │ FastAPI :8000         │
│ redis-stack, kafka-ui │                   │ /stats /frauds        │
└───────────────────────┘                   │ /transaction/{id}     │
                                            └───────────────────────┘
```

**Run order:** `docker-compose up -d` → `mlflow ui --port 5001` →
`python train_fraud_model.py` → `python producer.py --rate 100` → `python consumer.py`
→ `uvicorn API.main:app`.

---

## 2. Verified environment

| Component | Version | Note |
|---|---|---|
| Python | 3.9.6 | system Python, no venv |
| mlflow | 3.1.4 | stages deprecated in 3.x |
| xgboost | 2.1.4 | `use_label_encoder` removed in 2.x |
| scikit-learn | 1.6.1 | |
| imbalanced-learn | 0.12.4 | |
| shap | 0.49.1 | |
| pandas / numpy | 2.3.3 / 2.0.2 | |
| redis-py | 7.0.1 | |
| kafka-python | 3.0.1 | |
| fastapi | 0.128.8 | |
| prometheus-fastapi-instrumentator | 7.1.0 | |

Feature space: **438 features**, 31 categorical (label-encoded), 407 numeric.
Highest cardinality: `DeviceInfo` (1787), `id_33` (261), `id_31` (131).

---

## 3. Verified model results

Reproduced the exact `train_test_split(test_size=0.2, random_state=42, stratify=y)`
split (472,432 train / 118,108 test) and re-scored registered **version 5** through
the training preprocessing path:

```
AUC = 0.9469   AP = 0.6924      (MLflow logged run: AUC 0.9464, AP 0.6923)
```

The reconstruction matches the logged run to 4 decimal places — my model of the
pipeline is faithful.

**MLflow run history** (20 runs = 4 configs × 5 repeats, experiment `661888854093502547`):

| run | AUC-ROC | AP | precision@0.5 | recall@0.5 | F1 |
|---|---|---|---|---|---|
| `xgboost-main` ← registered | **0.946** | **0.692** | 0.286 | 0.825 | 0.425 |
| `xgboost-smote` | 0.912 | 0.614 | 0.791 | 0.462 | 0.583 |
| `baseline-logistic-regression` | 0.808 | 0.198 | 0.110 | 0.664 | 0.189 |
| `isolation-forest-unsupervised` | 0.746 | 0.090 | 0.119 | 0.205 | 0.150 |

Model registry: 5 versions of `fraud-xgboost`. **All have `current_stage: None` and
`aliases: []`** — nothing was ever promoted.

Model internals: `XGBClassifier`, `best_iteration=498` of 500 (early stopping barely
fired), **417 of 438 features receive at least one split** — 21 features are dead.

---

## 4. Threshold calibration (measured on the held-out 118,108 rows)

`FRAUD_THRESHOLD = 0.5` (consumer.py:48) is the single most costly choice in the repo.
`scale_pos_weight ≈ 27.5` inflates the output probabilities, so 0.5 is not a
meaningful operating point.

| threshold | precision | recall | F1 | alerts | false pos | FP per TP |
|---|---|---|---|---|---|---|
| **0.50** ← current | 0.286 | 0.826 | 0.425 | 11,922 | 8,509 | 2.5 |
| 0.60 | 0.382 | 0.772 | 0.511 | 8,339 | 5,150 | 1.6 |
| 0.70 | 0.508 | 0.711 | 0.593 | 5,784 | 2,846 | 1.0 |
| 0.80 | 0.667 | 0.621 | 0.643 | 3,847 | 1,280 | 0.5 |
| **0.844** ← F1-optimal | 0.754 | 0.573 | **0.651** | — | — | — |
| 0.90 | 0.853 | 0.493 | 0.625 | 2,389 | 350 | 0.2 |
| 0.95 | 0.920 | 0.388 | 0.546 | 1,743 | 140 | 0.1 |
| 0.99 | 0.965 | 0.215 | 0.352 | 922 | 32 | 0.0 |

At 0.5, **71% of every fraud alert is a false positive** — 8,509 legitimate customers
blocked to catch 3,413 frauds. Moving to 0.844 raises F1 from 0.425 → 0.651.

Dollar view (held-out fraud exposure = **$639,135**):

| threshold | fraud $ caught | good customers blocked |
|---|---|---|
| 0.50 | $517,208 | 8,509 |
| 0.90 | $274,907 | 350 |

The right threshold depends on your cost ratio (chargeback loss vs. review cost vs.
customer friction). It should be **chosen from the PR curve and persisted as an
artifact**, not hardcoded at 0.5. Note `precision_recall_curve` is already imported
in `train_fraud_model.py:24` and never used.

---

## 5. Findings

Ordered by severity. `file:line` references are clickable.

### 5.1 Blockers — code is broken as written

**B1. `[FIXED]` `API/main.py:62` — `json` is never imported → `/transaction/{tx_id}` always 500s.**
`json.loads(data)` raises `NameError`. Imports at lines 1-3 are only `fastapi`,
`redis`, `Instrumentator`. Any request for an existing transaction crashes; only the
not-found path works (which itself is wrong — see D3). One-line fix: `import json`.

**B2. `[FIXED]` `consumer.py:264` exposes metrics on :8002, `API/prometheus.yml:15` scrapes :8001.**
Prometheus never collects a single consumer metric. `TRANSACTIONS_TOTAL`,
`FRAUDS_TOTAL`, `LATENCY` are all recorded into the void.

**B3. `[FIXED]` Prometheus and Grafana are not in `docker-compose.yml` at all.**
`grep -c "prometheus\|grafana" docker-compose.yml` → **0**. `API/prometheus.yml`
exists but nothing consumes it. The monitoring layer is wired at the code level and
absent at the infra level.

**B4. `[FIXED]` `consumer.py:46` — `MODEL_STAGE = "Production"` can never resolve.**
No registry version was ever promoted (all `current_stage: None`, `aliases: []`), so
`models:/fraud-xgboost/Production` throws and the `except` at line 150 silently falls
back to `models:/fraud-xgboost/latest`. **Production always serves whatever was
trained last, with no promotion gate.** MLflow 3.x deprecates stages entirely — the
correct mechanism now is aliases: `client.set_registered_model_alias("fraud-xgboost",
"champion", 5)` and `models:/fraud-xgboost@champion`.

**B5. `[FIXED]` `docker-compose.yml:32` + `:67` — kafka-ui cannot reach Kafka.**
`KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092` while kafka-ui bootstraps
`kafka:9092`. Metadata comes back advertising `localhost:9092`, which inside the
kafka-ui container resolves to kafka-ui itself. Needs dual listeners:
```yaml
KAFKA_LISTENERS: INTERNAL://0.0.0.0:29092,EXTERNAL://0.0.0.0:9092
KAFKA_ADVERTISED_LISTENERS: INTERNAL://kafka:29092,EXTERNAL://localhost:9092
KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: INTERNAL:PLAINTEXT,EXTERNAL:PLAINTEXT
KAFKA_INTER_BROKER_LISTENER_NAME: INTERNAL
```
(kafka-ui then points at `kafka:29092`.)

### 5.2 Correctness — silently wrong results

**C1. `[FIXED]` `producer.py:34` never merges `train_identity.csv` → 41 features always missing.**
Measured: identity coverage is **32.9% in training data, 0.0% in the stream.** All of
`id_01`–`id_38`, `DeviceType`, `DeviceInfo` arrive as `None` and get filled with the
missing sentinels. Scoring the same 3,000 transactions through both paths:

```
mean score   train-path 0.1806   serve-path 0.1870
flagged @0.5 train-path 7.63%    serve-path 8.03%
mean |delta| 0.0110   max |delta| 0.5108
DECISION FLIPS: 28/3000 = 0.93% of transactions get a different verdict
columns differing: 41 (exactly the identity block)
```

~1 in 107 transactions is misclassified purely because the producer feeds a different
feature distribution than the model was trained on. Fix: merge identity in the
producer, or train without identity features so train and serve agree.

**C2. `[FIXED — needs retrain]` `train_fraud_model.py:90` — `addr_mismatch` is a dead constant, and skewed.**
`(df["addr1"] != df["addr2"])` compares a **region code** to a **country code**
(`addr2` is 87.0 for 90% of rows). They are never equal — measured `addr1 == addr2`
on 50,000 rows: **0 times**. So the feature is **constant 1 across 100% of training
rows**, and `get_score(importance_type="gain")` confirms **zero splits** — the model
ignores it entirely.

Worse, the two implementations disagree. Training: `NaN != NaN` is `True` in pandas,
so both-missing → 1. Serving (`consumer.py:97`): `a1 != a2 and a1 and a2` requires
both truthy, so both-missing → 0. **5.2% of rows get a different value.** Harmless
today only because the model ignores the feature. The intended signal — billing vs.
shipping mismatch — is not what's being computed.

**C3. `[FIXED — needs retrain]` `train_fraud_model.py:234` — early stopping evaluates on the test set.**
`eval_set=[(X_test, y_test)]` with `early_stopping_rounds=30`, and then
`compute_metrics(y_test, ...)` reports on that same set. The stopping iteration was
selected using test labels, so the reported 0.9469 AUC is optimistic. Needs a
three-way train/valid/test split.

**C4. `[FIXED — needs retrain]` `train_fraud_model.py:348` — random split on time-series data.**
`TransactionDT` is a seconds offset; IEEE-CIS is explicitly temporal. A random
stratified split puts future transactions in train and past in test, and scatters the
same card/device across both sides. This leaks, and it is the main reason 0.9469 looks
much better than what this feature set achieves on a true forward-in-time holdout.
Fix: sort by `TransactionDT` and cut chronologically.

**C5. `[FIXED — needs retrain]` `train_fraud_model.py:117-122` — encoders fit on the full dataset before splitting.**
`cat_maps` is built from `X[col].unique()` over train *and* test. Test categories leak
into the mapping. Minor next to C4, but it means `UNKNOWN_CATEGORY = -1` is never
exercised during evaluation while it fires constantly in production.

**C6. `[FIXED]` `consumer.py:60-63` — latency histogram buckets are wrong by 1000×.**
`Histogram("model_latency_ms", ...)` uses `DEFAULT_BUCKETS = (0.005 … 10.0, inf)`,
which are **seconds**. The code observes milliseconds (`LATENCY.observe(latency_ms)`,
line 307), typically 2-30. Every observation above 10 lands in `+Inf`, so all
quantiles are unusable. Either rename to `_seconds` and divide by 1000 (Prometheus
convention), or pass explicit `buckets=[1,2,5,10,25,50,100,250,500]`.

**C7. `[FIXED — needs retrain]` `train_fraud_model.py:218` — `use_label_encoder` is not a valid XGBoost 2.x param.**
Confirmed absent from `XGBClassifier.__init__` signature. It flows into `**kwargs`,
gets forwarded to the booster as an unknown parameter, and is silently discarded with
a warning. Dead config that looks meaningful. Delete it.

**C8. `[FIXED — needs retrain]` `train_fraud_model.py:176-192` — the logistic regression baseline is not scaled.**
`LogisticRegression(C=0.1, max_iter=1000)` on 438 unstandardized features where
missing values are `-999` and `DeviceInfo` is an ordinal 0-1786. Without a
`StandardScaler` this cannot converge meaningfully — hence AUC 0.808 / AP 0.198. As a
"did XGBoost beat the baseline" gate, it is not a fair comparison. `LabelEncoder` is
imported at line 19 and never used.

**C9. `[FIXED — needs retrain]` `train_fraud_model.py:124` — `-999` sentinel defeats XGBoost's native NaN handling.**
XGBoost learns an optimal default direction per split for genuine `NaN`. Replacing
missing with `-999` forces it to spend splits separating the sentinel and conflates
"missing" with "very negative". For the tree models, pass `NaN` through.

**C10. `[FIXED — needs retrain]` `train_fraud_model.py:146-148` — `classification_report` computed three times,
and `["1"]` raises `KeyError` if no positive is predicted.** Call it once; or better,
use `precision_recall_fscore_support`.

### 5.3 Production-readiness

**P1. `[FIXED]` `consumer.py:214` — `enable_auto_commit=True` gives at-most-once delivery.**
Offsets commit on a 1s timer regardless of whether scoring succeeded. A crash between
commit and Redis write loses those transactions permanently. For fraud, you want
manual commit *after* the write, plus idempotent Redis keys (which `prediction:<id>`
already is).

**P2. `[FIXED]` `consumer.py:339-342` — failures are logged and dropped, no dead-letter queue.**
A malformed message vanishes with one `log.error`. No DLQ topic, no failure counter,
so a systematic deserialization bug would show up as silence, not an alert.

**P3. `[FIXED]` Kafka topic is auto-created with 1 partition (`docker-compose.yml:34`).**
`KAFKA_AUTO_CREATE_TOPICS_ENABLE: "true"` with no partition config. The consumer
group can never scale past one member — a second `consumer.py` sits idle. Create the
topic explicitly with 6-12 partitions.

**P4. `[FIXED]` `consumer.py:237` — one `predict_proba` call per message.**
Single-row DataFrame construction plus a 500-tree traversal per transaction. Both
`pd.DataFrame([row])` and the XGBoost call have fixed overhead that amortizes across a
batch. Micro-batching via `consumer.poll(timeout_ms=..., max_records=256)` and one
`predict_proba` on the stacked frame is worth roughly an order of magnitude.

**P5. `[FIXED]` `consumer.py:190-201` — up to 4 sequential Redis round-trips per transaction.**
`setex` + `zadd` + `incr` × 3, each a separate network call. Wrap in
`r.pipeline()` → one round trip.

**P6. `[FIXED]` `consumer.py:194` — `flagged_transactions` zset grows without bound.**
No TTL, no `ZREMRANGEBYRANK` trim. Meanwhile `redis-stack` is configured
`--maxmemory 256mb --maxmemory-policy allkeys-lru` (`docker-compose.yml:50`), so under
pressure Redis will evict **the zset and the stats counters** — LRU does not respect
their importance. `/frauds` and `/stats` silently return garbage.

**P7. `[PARTIAL]` `consumer.py:197-201` — stats counters have no TTL while predictions expire in 1 hour.**
`stats:total_scored` accumulates forever across every run, but `prediction:*` keys die
after `REDIS_TTL_SECONDS = 3600`. `/stats` therefore reports lifetime totals against a
1-hour data window, and there is no reset path. The counters also double-count on
Kafka replay since `auto_offset_reset="earliest"` re-reads from the beginning.

**P8. `[FIXED]` `consumer.py:218` — `consumer_timeout_ms=600000` ends the process after 10 idle minutes.**
The `for message in consumer` loop raises `StopIteration`, `finally` prints the summary
and exits. Looks like a clean shutdown; is actually an unintended stop. A long-running
service should use `poll()` in a `while True`.

**P9. `[FIXED]` `producer.py:113` loads the full 652 MB CSV into memory, then `iterrows()`.**
~590k rows × 394 columns. Use `pd.read_csv(..., chunksize=10000)` and
`itertuples()`/dict iteration.

**P10. `[FIXED]` `producer.py:145` ships the `isFraud` label inside the transaction payload.**
The consumer correctly excludes it from features (`consumer.py:114`) and stores it as
`true_label` for offline eval — but a real ingestion stream has no label, and having it
in-band is exactly how leakage gets introduced later. Send it on a separate
side-channel topic keyed by `TransactionID`.

**P11. `[OPEN]` No schema validation on inbound messages.** `consumer.py:299` trusts
`message.value` completely. A Pydantic model or Avro/Protobuf schema registry belongs
here.

**P12. `[FIXED]` Hardcoded endpoints everywhere.** `localhost:9092`, `localhost:6379`,
`http://localhost:5001`, `localhost` in `API/main.py:9`. Nothing is environment-driven,
so none of it runs in Docker or CI. Move to `os.getenv` with defaults.

**P13. `[OPEN]` Zookeeper-based Kafka.** `cp-kafka:7.5.0` with Zookeeper — deprecated upstream.
KRaft mode drops a whole service.

**P14. `[FIXED]` `docker-compose.yml:35` — `KAFKA_LOG_RETENTION_HOURS: 1`.** Stream data is
deleted after an hour, so replays past that window silently return nothing.

### 5.4 Design & hygiene

**D1. `[FIXED]` Missing `requirements.txt` / `README.md` / `.gitignore` / `Dockerfile` / tests.**
All confirmed absent. **The directory is not a git repository** — there is no version
history for any of this code.

**D2. `[FIXED]` `data/` is 1.3 GB inside the project directory** with no `.gitignore`. The moment
`git init` happens, this is a landmine.

**D3. `[FIXED]` `API/main.py:60` returns `{"error": "Not found"}` with HTTP 200.**
Should be `raise HTTPException(status_code=404, detail="Not found")`. Clients cannot
distinguish "missing" from "found" by status code.

**D4. `[PARTIAL]` `API/main.py` has no response models, no auth, no rate limiting, no pagination.**
`/frauds` hardcodes `0, 9` (`zrevrange(..., 0, 9)`) with no `limit`/`offset` params. An
endpoint exposing fraud scores by transaction ID with zero authentication is a
production non-starter.

**D5. `[OPEN]` Redis client is module-level and synchronous in an async framework.**
`API/main.py:8` creates a blocking `redis.Redis` used from `def` (not `async def`)
handlers. FastAPI will thread-pool them, so it works, but `redis.asyncio` plus a
lifespan-managed pool is the correct shape. There is also no reconnect handling —
`r.ping()` in `consumer.py:178` is checked once at startup and never again.

**D6. `[FIXED — needs retrain]` `train_fraud_model.py:167-169` writes `shap_xgboost.png` to the repo root and
never deletes it.** That stray 140 KB file is sitting in the project root right now.
Write to a temp dir.

**D7. `[FIXED — needs retrain]` Duplicate MLflow runs — 20 runs across 4 unique configs.** The script was run 5
times with identical parameters. Nothing tags git SHA, data version, or feature-set
hash, so the runs are indistinguishable. `mlflow.set_tag("git_sha", ...)` and
`mlflow.log_input()` would fix this.

**D8. `[FIXED — needs retrain]` `mlflow.xgboost.log_model(model, "model")` uses the deprecated positional
`artifact_path`.** MLflow 3.x wants `name="model"`. Works, warns.

**D9. `[FIXED — needs retrain]` `run_xgboost_smote` and `run_isolation_forest` do not log the preprocessing
artifacts** — only `run_xgboost` does (lines 226-228). Those two runs are therefore
not reproducible at serving time.

**D10. `[FIXED]` `datetime.utcnow()` is deprecated** (`producer.py:67`, `consumer.py:255`). Use
`datetime.now(timezone.utc)`.

**D11. `[FIXED]` Unused imports**: `precision_recall_curve` and `confusion_matrix`
(`train_fraud_model.py:23-24`), `LabelEncoder` (line 19). `producer.py:44`
`delivery_report()` is defined and never wired to a callback — `producer.send()` at
line 138 passes no callback, so delivery is never actually confirmed despite
`acks="all"`.

**D12. `[FIXED]` `warnings.filterwarnings("ignore")` at `train_fraud_model.py:30`** hides exactly
the signals that would have surfaced C7 and D8.

**D13. `[FIXED — needs retrain]` Isolation Forest normalization is fit on the test set.**
`consumer`-independent, but `train_fraud_model.py:324` min-max normalizes using
`proba.min()`/`proba.max()` **of the test predictions**. Those bounds are unavailable
at serving time, so the score is not reproducible online.

**D14. `[KEPT]` `consumer.py:158-171` `get_feature_columns()` — this one is done well.**
It cross-checks the artifact feature list against `model.feature_names_in_` and prefers
the model's ordering (lines 272-281), with the right reasoning in the comment:
XGBoost is positional, so a mismatch means silently wrong scores. Keep this pattern.

---

## 6. What is genuinely good here

- **The train/serve contract is explicit.** Persisting `cat_maps.json`,
  `feature_columns.json`, and `encoding_config.json` (`train_fraud_model.py:127-134`)
  and reloading them in the consumer is the correct architecture. Most projects at this
  level re-fit a `LabelEncoder` at inference and never notice.
- **Feature-order defense** (D14) — catches the failure mode that is hardest to debug.
- **`scale_pos_weight` vs. SMOTE as a logged experiment**, not an argument. The
  comparison is in MLflow and the numbers tell a real story: SMOTE trades recall
  (0.826 → 0.462) for precision (0.286 → 0.791).
- **`aucpr` as the eval metric**, correct for 3.5% positives where AUC-ROC flatters.
- **SHAP logged per run** — explainability treated as a first-class artifact.
- **Isolation Forest as an unsupervised complement** — the right instinct for catching
  novel fraud patterns a supervised model has never seen.
- **Per-message exception isolation** (`consumer.py:339`) — one bad message cannot kill
  the stream.
- **Kafka key = `TransactionID`** (`producer.py:136`) — correct partitioning choice for
  per-transaction ordering.

The architecture is sound. Nearly every finding above is a wiring or calibration
defect, not a structural one.

---

## 7. Fix order

**Tier 1 — 20 minutes, unblocks the demo**
1. `import json` in `API/main.py` (B1)
2. Consumer metrics port 8002 → 8001, or fix `prometheus.yml` (B2)
3. Add prometheus + grafana services to `docker-compose.yml` (B3)
4. Fix Kafka dual-listener config so kafka-ui connects (B5)
5. Switch to registry **aliases** — `models:/fraud-xgboost@champion` (B4)

**Tier 2 — correctness, half a day**
6. Merge `train_identity.csv` in the producer (C1) — removes 0.93% decision flips
7. Fix or delete `addr_mismatch`; compute real billing-vs-shipping mismatch (C2)
8. Fix latency histogram buckets (C6)
9. Delete `use_label_encoder` (C7)
10. Tune the threshold from the PR curve and persist it as an artifact (§4)

**Tier 3 — methodology, and this changes your headline number**
11. Chronological split on `TransactionDT` (C4)
12. Three-way split so early stopping stops using test labels (C3)
13. Fit encoders on train only (C5)
14. Scale the LR baseline (C8); pass `NaN` instead of `-999` to XGBoost (C9)

**Tier 4 — production**
15. Manual offset commit after the Redis write (P1) + DLQ (P2)
16. Micro-batch inference (P4) + Redis pipelining (P5)
17. Explicit topic with 6-12 partitions (P3)
18. Trim/TTL the zset; protect stats keys from LRU eviction (P6, P7)
19. `poll()` loop instead of `consumer_timeout_ms` (P8)
20. Env-driven config (P12)

**Tier 5 — hygiene**
21. `git init`, `.gitignore` (exclude `data/`, `mlruns/`, `mlartifacts/`, `*.png`)
22. `requirements.txt` pinned, `README.md` with the run order from §1
23. Tests — the highest-value one asserts train-path and serve-path produce identical
    feature vectors for the same row (this is the C1/C2 regression guard)

---

## 8. Reproducing the measurements in this document

The threshold table (§4) and skew numbers (C1) came from reconstructing the pipeline
against the local artifacts:

```python
# exact test split — depends only on n_samples, y, random_state
y_all = pd.read_csv("data/train_transaction.csv", usecols=["isFraud"])["isFraud"]
tr_i, te_i = train_test_split(np.arange(len(y_all)), test_size=0.2,
                              random_state=42, stratify=y_all)

# registered version 5, loaded straight off disk without a tracking server
model = mlflow.xgboost.load_model(
    "mlartifacts/661888854093502547/models/"
    "m-a51f1a228b0a455c949a6022787e5f86/artifacts")

# dead-feature check
model.get_booster().get_score(importance_type="gain").get("addr_mismatch")  # -> None
```

Registry version → run mapping:

| version | run_id | model_id |
|---|---|---|
| 1 | `b4330bf1661e44a08b9b8db007451db8` | `m-e7c4b3f1f26343c782b149ad97c2819e` |
| 2 | `0a609c4fc5a144b7a23b55291c672bca` | `m-a1e9718468a74907b744570ef81be97f` |
| 3 | `1e28c9ed01ad4426a562e46820819a36` | `m-fd4419e05bcd4b25b2293a2318658254` |
| 4 | `e379b2e683c745abaef3a43eb3a696f8` | `m-a10ba94baae44c37a64c95309001df0b` |
| 5 (latest, served) | `918be045dbbd41c9a56681bbca7e8c0f` | `m-a51f1a228b0a455c949a6022787e5f86` |

---

## 9. Fixes applied (2026-08-12)

### 9.1 Three issues found *while* fixing, not in the original review

**N1. The MLflow server cannot serve artifacts — the consumer could not load a
model at all.** `mlflow.xgboost.load_model` failed with repeated HTTP 500s from
`/api/2.0/mlflow-artifacts/artifacts/...`. The files were present on disk and the
server's CWD was correct; the cause is that the server was started without
`--serve-artifacts --artifacts-destination`. Verified by starting a correctly
configured server on port 5002, where the identical request returns 200. The
required invocation is now in the README. **This blocks the whole pipeline and
was invisible until something actually tried to load a model.**

**N2. Port 8001 is occupied by RedisInsight, so the consumer can never bind there.**
`docker-compose.yml` publishes `8001:8001` for redis-stack's web UI. My first pass
at fixing B2 moved the consumer's metrics server from 8002 to 8001 to match
`prometheus.yml`; it died instantly with `OSError: [Errno 48] Address already in
use`. This is almost certainly why the original code used 8002. **The correct fix
is the opposite direction**: the scrape config moves to 8002, and the code stays.
Binding the metrics port is also no longer fatal — a busy port used to abort
startup before a single transaction was scored.

**N3. `SIGTERM` killed the consumer without running its shutdown path.** Docker and
Kubernetes stop containers with SIGTERM, and Python's default handler terminates
immediately — so the final offset commit, the DLQ flush and the summary were all
skipped, and anything mid-batch was lost. The consumer now installs handlers for
both SIGINT and SIGTERM that set a flag the poll loop checks, so it finishes the
current batch, commits, and exits cleanly. Verified: SIGTERM now produces
`Received SIGTERM — finishing current batch...` followed by the full summary.

Also worth recording: `MlflowClient.delete_model_version` is **broken in MLflow
3.1.4**. It raises `yaml.representer.RepresenterError: cannot represent an object`
because it tries to YAML-serialise `Metric` objects into the version's meta file.
Deleting a registry version currently requires removing
`mlruns/models/<name>/version-N/` by hand.

### 9.2 What changed, by file

| File | Change |
|---|---|
| `API/main.py` | `import json` (B1); 404 via `HTTPException` (D3); connection pool; `/health`; pagination on `/frauds` (D4); 503 when Redis is down; env-driven host/port (P12) |
| `consumer.py` | Alias-based model resolution (B4); artifact-driven sentinels; `addr_mismatch` matched to training (C2); millisecond histogram buckets (C6); manual commit after Redis write (P1); DLQ + failure counter (P2); micro-batched scoring (P4); Redis pipelining (P5); zset trimming (P6); `poll()` loop (P8); timezone-aware timestamps (D10); signal handling (N3); non-fatal metrics bind (N2) |
| `producer.py` | Left-joins `train_identity.csv` (C1); chunked CSV reads (P9); `--no-label` flag (P10); errback wired to `send` (D11); numpy-aware NaN handling; env config (P12) |
| `train_fraud_model.py` | Chronological three-way split (C3, C4); encoders fit on train only (C5); scaled LR baseline (C8); NaN instead of `-999` (C9); `use_label_encoder` removed (C7); single-pass metrics (C10); SHAP to a temp dir (D6); git SHA tags (D7); `name=` kwarg (D8); artifacts logged on every run (D9); Isolation Forest normalised with train bounds (D13); threshold selected on validation |
| `features.py` | **New.** Single source of truth for batch-side feature engineering |
| `docker-compose.yml` | Dual Kafka listeners (B5); Prometheus + Grafana (B3); `kafka-init` creating 6-partition topics (P3); `volatile-lru` (P6, P7); 24h retention (P14) |
| `API/prometheus.yml` | Scrape target corrected to 8002 (B2, N2) |
| `scripts/promote_model.py` | **New.** Alias-based promotion with an optional `--min-auc` gate |
| `scripts/select_threshold.py` | **New.** PR-curve calibration writing `threshold_config.json` |
| `tests/test_train_serve_parity.py` | **New.** 6 tests asserting batch and streaming paths agree |
| `requirements.txt`, `.gitignore`, `README.md` | **New.** (D1, D2) |

### 9.3 Verified working

* `pytest tests/ -v` → **6 passed**. The parity test caught a real bug during
  development: `None != None` is `False` in Python while `NaN != NaN` is `True` in
  pandas, so the first rewrite of `addr_mismatch` still disagreed with training.
* Producer now streams **22.7% of transactions carrying identity data** (was 0%).
* Consumer loads `models:/fraud-xgboost@champion` → v5, reads thresholds from the
  artifact (`fraud=0.8440 high_risk=0.9503`), scored 1,500 transactions with
  **0 failures**.
* **Latency: 0.34 ms/transaction, down from 48.69 ms** in the stored predictions —
  the micro-batching win, ~140×.
* Latency histogram now spreads across real buckets (1,438 of 1,500 under 0.5 ms)
  instead of every sample landing in `+Inf`.
* Flagged rate dropped from **8.3% to 1.53%** at the calibrated threshold.
* `promote_model.py --min-auc 0.99` correctly *refuses* to promote.
* SIGTERM shutdown drains, commits, and prints the summary.

### 9.4 Deliberately NOT done

* **No retraining.** All Tier-3 methodology fixes are in the code but take effect
  only on the next `train_fraud_model.py` run. The deployed v5 was trained with a
  random split, `-999` sentinels and the old feature set, and `artifacts/` still
  matches it exactly (438 features, `missing_numeric: -999`). Retraining is a
  deliberate decision because **the honest chronological number will be lower than
  0.9469** — that figure was inflated by leakage, not earned.
* **`addr_mismatch` still computed** in both paths, so v5 keeps scoring correctly.
  It is excluded from new runs via `LEGACY_FEATURES`. New runs produce 439
  features: `-addr_mismatch`, `+addr_missing`, `+email_mismatch`.
* **`git init` not run** — that is your call, and `.gitignore` is ready for it.
* **Docker stack not restarted.** The compose changes (partitions, listeners,
  Prometheus, Grafana, `volatile-lru`) require `docker compose up -d` to take
  effect. The running broker still has a 1-partition `transactions` topic.
* **Still open**: P11 (no message schema validation), P13 (Zookeeper → KRaft),
  D5 (sync Redis client in FastAPI), D4 partially (no authentication on the API).

### 9.5 Smoke-test result, for calibration not for reporting

A 60k-row subsample run through the rewritten training pipeline produced XGBoost
AUC 0.8884 with the chronological split. **Do not read that as the new headline
number** — it is 42k training rows against the real 472k, so it conflates the
subsample with the split change. The real figure needs a full run. What it does
confirm is that the pipeline is correct end to end: the splits are strictly
time-ordered, and the unseen-category rate is 0.00% on train against 0.04% on
valid/test, which is only possible if the encoders were fit on train alone.

---

## 10. A bug I introduced, and how it was caught (2026-08-14)

Worth recording because it is the same class of defect as C2 — the original
train/serve skew — reintroduced by me, in the demo app, three weeks later.

**What happened.** The deployable demo dataset was stored as Parquet. Parquet
round-trips a missing value in an object column as Python `None`. The batch
encoder did `X[col].astype(str).map(cat_maps[col])`, and `str(None)` is `"None"`.

Training had read CSVs, where missing was pandas `NaN`, and `str(NaN)` is `"nan"`.
So `"nan"` is the key actually stored in `cat_maps` — and `"None"` matched nothing,
falling through to `unknown_category` (-1) instead of the trained missing-value
index.

**Measured impact on the 12,000-row demo set:**

| | |
|---|---|
| Categorical cells mis-encoded | 223,556 of 372,000 — **60.1%** |
| Rows affected | **12,000 of 12,000** |
| Mean score | 0.1444 (buggy) vs 0.1885 (correct) |
| Precision at the deployed threshold | 0.644 (buggy) vs 0.523 (correct) |
| Decision flips | **125 of 12,000 — 1.04%** |

The buggy version looked *better* on precision, which is exactly why it went
unnoticed: mis-encoding missing categoricals as "unseen" made the model more
conservative, so it raised fewer, higher-quality alerts. A metric moving in a
flattering direction is not evidence of correctness.

**How it surfaced.** Not from a test — from measuring a supposedly unrelated
change. While evaluating whether to drop Parquet for image size, I compared the
feature matrices produced from Parquet and from CSV. Thirty of thirty-one
categorical columns disagreed. The size question was cosmetic; the answer wasn't.

**The real fix.** Dropping Parquet would have masked it, not fixed it. The defect
was that `features.encode_categoricals()` trusted the *storage format's* null
representation instead of the training contract. It now normalises every flavour of
missing — `None`, `float('nan')`, `pd.NA`, `pd.NaT` — to the `null_category_str`
recorded in `encoding_config.json` before the string cast. Parquet and CSV now
produce byte-identical matrices.

**Also done, separately:** demo data moved to gzipped CSV, which removed `pyarrow`
(105 MB of compiled Arrow libraries, used for one read of one 1 MB file). Image
891 MB → **691 MB**, container RAM 244 MB → **170 MB**. The CSV is *smaller* than
the Parquet was (1.03 MB vs 1.52 MB) and costs ~140 ms more at startup.

**The lesson that generalises:** the serialisation format is part of the train/serve
contract whether you treat it as such or not. Anything that can represent "missing"
in more than one way is a place where training and serving can silently diverge.

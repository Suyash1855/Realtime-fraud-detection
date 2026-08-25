"""
Kafka Consumer — Real-Time Fraud Scorer
=======================================
Reads transactions from Kafka, engineers features, scores with the
XGBoost model loaded from MLflow, and stores results in Redis.

Design notes worth knowing before editing:

* The train/serve contract lives in artifacts/. Sentinel values, the categorical
  maps and the feature ORDER are all read from disk rather than hardcoded here,
  so retraining with different preprocessing cannot silently desync serving.
* XGBoost is positional. Feature order comes from the model itself whenever it
  exposes names, and the artifact list is only a cross-check.
* Offsets are committed AFTER the Redis write, giving at-least-once delivery.
  Redis keys are idempotent (prediction:<id>), so replays are harmless.
* Messages are scored in micro-batches: one predict_proba call per poll rather
  than one per transaction.

Run AFTER the producer is running.
Usage: python consumer.py
"""

import json
import logging
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import redis
import mlflow.xgboost
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaConnectionError
from prometheus_client import Counter, Histogram, Gauge, start_http_server

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [CONSUMER] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


# ─── CONFIG ──────────────────────────────────────────────────────────────────
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "artifacts"))

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
TOPIC = os.getenv("KAFKA_TOPIC", "transactions")
CONSUMER_GROUP = os.getenv("CONSUMER_GROUP", "fraud-scorer")
DLQ_TOPIC = os.getenv("DLQ_TOPIC", "transactions-dlq")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_TTL_SECONDS = int(os.getenv("REDIS_TTL_SECONDS", "3600"))   # predictions live 1h
FLAGGED_ZSET_MAX = int(os.getenv("FLAGGED_ZSET_MAX", "10000"))    # cap the leaderboard

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
MODEL_NAME = os.getenv("MODEL_NAME", "fraud-xgboost")
# MLflow 3.x removed stages. Promotion is done with an alias:
#   python scripts/promote_model.py --version 5 --alias champion
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
MODEL_VERSION = os.getenv("MODEL_VERSION")   # explicit pin; overrides the alias

# Scoring thresholds. Loaded from artifacts/threshold_config.json when present so
# the operating point is a versioned artifact, not a magic number in code.
# 0.5 is NOT a sensible default here: scale_pos_weight (~27.5) inflates the
# probabilities, and at 0.5 measured precision is 0.286 -- 71% of alerts are
# false positives. See PROJECT_ANALYSIS.md section 4.
DEFAULT_FRAUD_THRESHOLD = 0.844      # F1-optimal on the held-out split
DEFAULT_HIGH_RISK_THRESHOLD = 0.95   # ~0.92 precision

# 8002, not 8001: redis-stack publishes its RedisInsight UI on 8001, so binding
# there fails with EADDRINUSE. Keep this in sync with API/prometheus.yml.
METRICS_PORT = int(os.getenv("METRICS_PORT", "8002"))
BATCH_MAX_RECORDS = int(os.getenv("BATCH_MAX_RECORDS", "256"))
POLL_TIMEOUT_MS = int(os.getenv("POLL_TIMEOUT_MS", "1000"))

# ─── METRICS ─────────────────────────────────────────────────────────────────
TRANSACTIONS_TOTAL = Counter(
    "transactions_processed_total",
    "Total transactions processed"
)

FRAUDS_TOTAL = Counter(
    "frauds_detected_total",
    "Total frauds detected"
)

FAILURES_TOTAL = Counter(
    "scoring_failures_total",
    "Transactions that could not be scored"
)

# Explicit millisecond buckets. prometheus_client's defaults top out at 10 and are
# meant for SECONDS -- observing milliseconds against them put every single
# sample in the +Inf bucket, making all quantiles meaningless.
LATENCY = Histogram(
    "model_latency_milliseconds",
    "End-to-end scoring latency per transaction (ms)",
    buckets=(0.5, 1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, float("inf")),
)

BATCH_SIZE = Histogram(
    "consumer_batch_size",
    "Records scored per poll",
    buckets=(1, 2, 5, 10, 25, 50, 100, 256, float("inf")),
)

MODEL_INFO = Gauge(
    "model_version_info",
    "Currently served model version",
    ["model_name", "version"],
)


# ─── GRACEFUL SHUTDOWN ───────────────────────────────────────────────────────
# Docker and Kubernetes stop containers with SIGTERM, which by default kills the
# process outright -- skipping the final offset commit and the DLQ flush, and
# losing whatever was mid-batch. kafka-python also swallows SIGINT while it is
# blocked inside poll(), so Ctrl+C alone did not stop the loop either.
# Both signals now just set a flag that the poll loop checks each pass.
_shutdown = False


def _request_shutdown(signum, _frame):
    global _shutdown
    if _shutdown:
        log.warning("Second signal received — exiting immediately")
        raise SystemExit(1)
    _shutdown = True
    log.info(f"Received {signal.Signals(signum).name} — finishing current batch...")


def install_signal_handlers():
    signal.signal(signal.SIGINT, _request_shutdown)
    signal.signal(signal.SIGTERM, _request_shutdown)


# ─── PREPROCESSING CONTRACT ──────────────────────────────────────────────────

def load_preprocessing_artifacts():
    """
    Load the encoders and sentinel policy persisted at training time.

    encoding_config.json is the authority on how missing values are encoded. It
    used to be duplicated as module constants in this file, so changing the
    training policy would have silently broken serving.
    """
    cat_maps = json.loads((ARTIFACT_DIR / "cat_maps.json").read_text())
    feature_columns = json.loads((ARTIFACT_DIR / "feature_columns.json").read_text())

    cfg_path = ARTIFACT_DIR / "encoding_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
    else:
        log.warning("encoding_config.json missing — falling back to legacy defaults")
        cfg = {}

    policy = {
        "unknown_category": cfg.get("unknown_category", -1),
        # None means "pass NaN through and let XGBoost route it", which is what a
        # model trained without a numeric sentinel expects.
        "missing_numeric": cfg.get("missing_numeric", -999),
        "null_category_str": cfg.get("null_category_str", "nan"),
    }

    log.info(
        f"Loaded encoders | {len(feature_columns)} features, "
        f"{len(cat_maps)} categorical | missing_numeric={policy['missing_numeric']}"
    )
    return cat_maps, feature_columns, policy


def load_threshold_config():
    """
    Read the decision thresholds from a versioned artifact if training produced
    one, otherwise fall back to env vars and then to the measured defaults.
    """
    path = ARTIFACT_DIR / "threshold_config.json"
    if path.exists():
        cfg = json.loads(path.read_text())
        fraud = float(cfg["fraud_threshold"])
        high = float(cfg.get("high_risk_threshold", DEFAULT_HIGH_RISK_THRESHOLD))
        log.info(
            f"Thresholds from artifact | fraud={fraud:.4f} high_risk={high:.4f} "
            f"(policy={cfg.get('policy', '?')}, "
            f"expected precision={cfg.get('expected_precision', '?')}, "
            f"recall={cfg.get('expected_recall', '?')})"
        )
        return fraud, high

    fraud = float(os.getenv("FRAUD_THRESHOLD", DEFAULT_FRAUD_THRESHOLD))
    high = float(os.getenv("HIGH_RISK_THRESHOLD", DEFAULT_HIGH_RISK_THRESHOLD))
    log.warning(
        f"No threshold_config.json — using fraud={fraud} high_risk={high}. "
        "Run scripts/select_threshold.py to calibrate against a holdout set."
    )
    return fraud, high


# ─── FEATURE ENGINEERING ─────────────────────────────────────────────────────
# Must exactly match what was done during training (train_fraud_model.py).
# Anything added here is only USED if it appears in feature_columns.json, which
# keeps this function compatible with models trained before or after a change.

def engineer_features(tx: dict) -> dict:
    """Replicate the same feature engineering used at training time."""
    dt = tx.get("TransactionDT") or 0

    tx["hour"] = int(dt / 3600) % 24
    tx["day_of_week"] = int(dt / (3600 * 24)) % 7
    tx["is_night"] = 1 if (tx["hour"] < 6 or tx["hour"] > 22) else 0

    amount = float(tx.get("TransactionAmt") or 0)
    tx["log_amount"] = np.log1p(amount)
    tx["amount_rounded"] = 1 if (amount % 1 == 0) else 0

    addr1 = tx.get("addr1")
    addr2 = tx.get("addr2")

    # Reproduces training EXACTLY, including its quirk. pandas evaluates
    # NaN != NaN as True, so a row missing either address got a 1. Note that a
    # plain `addr1 != addr2` is WRONG here: missing arrives as None, and
    # None != None is False in Python while NaN != NaN is True in pandas. So the
    # condition is "not equal, treating any missing value as unequal".
    #
    # The previous serving logic required both to be truthy and returned 0 for
    # missing addresses, disagreeing with training on 5.2% of rows.
    #
    # This feature is informationally dead: addr1 is a region code and addr2 a
    # country code, so they are never equal (0 matches in 50k rows) and the
    # trained model gives it zero splits. Kept only for feature-order
    # compatibility with the deployed model. The replacements are below.
    both_addr_present = addr1 is not None and addr2 is not None
    tx["addr_mismatch"] = 0 if (both_addr_present and addr1 == addr2) else 1

    # Real signals to use once the model is retrained.
    tx["addr_missing"] = 1 if addr1 is None else 0
    p_email = tx.get("P_emaildomain")
    r_email = tx.get("R_emaildomain")
    tx["email_mismatch"] = (
        1 if (p_email is not None and r_email is not None and p_email != r_email)
        else 0
    )

    high_risk_domains = {"gmail.com", "yahoo.com", "hotmail.com"}
    email = tx.get("P_emaildomain", "") or ""
    tx["risky_email"] = 1 if email in high_risk_domains else 0

    return tx


DROP_COLS = frozenset({"TransactionID", "TransactionDT", "isFraud", "ingested_at"})


def build_feature_row(tx: dict, feature_columns: list, cat_maps: dict,
                      policy: dict) -> dict:
    """
    Turn one raw transaction into a single ordered feature row, applying the
    SAME encodings used at training time.
    """
    tx = engineer_features(tx)
    tx = {k: v for k, v in tx.items() if k not in DROP_COLS}

    unknown = policy["unknown_category"]
    missing = policy["missing_numeric"]
    null_str = policy["null_category_str"]

    row = {}
    for col in feature_columns:
        val = tx.get(col)

        if col in cat_maps:
            # Categorical: match training's astype(str) behaviour exactly
            key = null_str if val is None else str(val)
            row[col] = cat_maps[col].get(key, unknown)
        else:
            # Numeric. missing=None means the model was trained on real NaN.
            if val is None:
                row[col] = np.nan if missing is None else missing
            else:
                try:
                    row[col] = float(val)
                except (TypeError, ValueError):
                    row[col] = np.nan if missing is None else missing

    return row


def build_feature_frame(rows: list, feature_columns: list) -> pd.DataFrame:
    """One DataFrame per batch instead of one per transaction."""
    return pd.DataFrame(rows, columns=feature_columns, dtype=np.float32)


# ─── MODEL LOADER ────────────────────────────────────────────────────────────

def load_model():
    """
    Load the registered XGBoost model from the MLflow Model Registry.

    Resolution order: explicit MODEL_VERSION -> alias -> highest version number.

    The previous implementation asked for stage "Production", which MLflow 3.x no
    longer supports and which no version here was ever promoted to. It therefore
    always threw, and a bare `except` silently fell through to "latest" -- so
    production served whatever had been trained most recently, with no gate.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()

    if MODEL_VERSION:
        uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
        resolved = MODEL_VERSION
        log.info(f"Loading pinned version: {uri}")
    else:
        try:
            mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
            uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
            resolved = mv.version
            log.info(f"Loading alias '{MODEL_ALIAS}' -> version {resolved}")
        except Exception as e:
            versions = client.search_model_versions(f"name='{MODEL_NAME}'")
            if not versions:
                raise RuntimeError(
                    f"No versions of '{MODEL_NAME}' in the registry at "
                    f"{MLFLOW_TRACKING_URI}. Run train_fraud_model.py first."
                ) from e
            latest = max(versions, key=lambda v: int(v.version))
            resolved = latest.version
            uri = f"models:/{MODEL_NAME}/{resolved}"
            log.warning(
                f"Alias '{MODEL_ALIAS}' not set ({type(e).__name__}) — falling back "
                f"to highest version {resolved}. This is NOT a promotion gate; run "
                f"scripts/promote_model.py to set the alias explicitly."
            )

    model = mlflow.xgboost.load_model(uri)
    MODEL_INFO.labels(model_name=MODEL_NAME, version=str(resolved)).set(1)
    log.info(f"Loaded model: {MODEL_NAME} v{resolved}")
    return model


def get_feature_columns(model) -> list:
    """
    Retrieve the feature names the model was trained on.
    XGBoost stores these in feature_names_in_ (sklearn API) or on the Booster.
    """
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    booster = model.get_booster() if hasattr(model, "get_booster") else None
    if booster is not None and booster.feature_names:
        return list(booster.feature_names)

    if hasattr(model, "feature_names") and model.feature_names:
        return list(model.feature_names)

    raise RuntimeError(
        "Cannot determine model feature columns. "
        "Make sure you saved feature names during training."
    )


# ─── REDIS CLIENT ────────────────────────────────────────────────────────────

def create_redis_client() -> redis.Redis:
    client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_keepalive=True,
        health_check_interval=30,   # reconnect transparently if the link drops
    )
    client.ping()  # raises if Redis is not running
    log.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
    return client


def store_predictions(r: redis.Redis, results: list):
    """
    Persist a batch of prediction results in one Redis round trip.

    Previously this was up to 4 sequential calls PER transaction (setex, zadd,
    and three incrs). A pipeline collapses the whole batch into one exchange.
    """
    frauds = [x for x in results if x["is_fraud"]]
    high_risk = sum(1 for x in results if x["risk_level"] == "HIGH")

    pipe = r.pipeline(transaction=False)
    for result in results:
        pipe.setex(
            f"prediction:{result['transaction_id']}",
            REDIS_TTL_SECONDS,
            json.dumps(result),
        )

    if frauds:
        pipe.zadd(
            "flagged_transactions",
            {x["transaction_id"]: x["fraud_score"] for x in frauds},
        )
        # Keep only the top N by score. The zset had no TTL and no trim, so it
        # grew without bound for as long as the consumer ran.
        pipe.zremrangebyrank("flagged_transactions", 0, -(FLAGGED_ZSET_MAX + 1))

    # Counters for the /stats endpoint
    pipe.incrby("stats:total_scored", len(results))
    if frauds:
        pipe.incrby("stats:total_fraud", len(frauds))
    if high_risk:
        pipe.incrby("stats:high_risk", high_risk)

    pipe.execute()


# ─── CONSUMER ────────────────────────────────────────────────────────────────

def create_consumer(retries: int = 5) -> KafkaConsumer:
    for attempt in range(1, retries + 1):
        try:
            consumer = KafkaConsumer(
                TOPIC,
                bootstrap_servers=KAFKA_BROKER,
                group_id=CONSUMER_GROUP,
                auto_offset_reset="earliest",   # read from start if no offset saved
                # At-least-once: we commit only after the Redis write succeeds.
                # Auto-commit ran on a 1s timer regardless of outcome, so a crash
                # between commit and write dropped those transactions for good.
                enable_auto_commit=False,
                max_poll_records=BATCH_MAX_RECORDS,
                value_deserializer=lambda b: json.loads(b.decode("utf-8")),
            )
            log.info(f"Subscribed to topic '{TOPIC}' as group '{CONSUMER_GROUP}'")
            return consumer
        except KafkaConnectionError:
            log.warning(f"Kafka not ready (attempt {attempt}/{retries}) — retrying in 3s...")
            time.sleep(3)

    raise RuntimeError("Could not connect to Kafka.")


def create_dlq_producer():
    """
    Dead-letter queue for messages we cannot score. Without it a systematic
    failure (a schema change, say) shows up as silence rather than an alert.
    """
    try:
        return KafkaProducer(
            bootstrap_servers=KAFKA_BROKER,
            value_serializer=lambda d: json.dumps(d, default=str).encode("utf-8"),
            acks="all",
        )
    except Exception as e:
        log.warning(f"DLQ producer unavailable ({e}) — failures will only be logged")
        return None


def score_batch(model, feature_columns: list, cat_maps: dict, policy: dict,
                txs: list, fraud_threshold: float, high_risk_threshold: float) -> list:
    """
    Score a batch of transactions with a single predict_proba call.
    Returns one result dict per input transaction, in order.
    """
    t0 = time.perf_counter()

    rows = [build_feature_row(dict(tx), feature_columns, cat_maps, policy) for tx in txs]
    X = build_feature_frame(rows, feature_columns)
    probas = model.predict_proba(X)[:, 1]

    # Latency is per-transaction wall time amortised across the batch.
    batch_latency_ms = (time.perf_counter() - t0) * 1000
    per_tx_latency_ms = batch_latency_ms / max(len(txs), 1)
    scored_at = datetime.now(timezone.utc).isoformat()

    results = []
    for tx, proba in zip(txs, probas):
        fraud_proba = float(proba)
        results.append({
            "transaction_id": str(tx.get("TransactionID")),
            "fraud_score": round(fraud_proba, 4),
            "is_fraud": bool(fraud_proba >= fraud_threshold),
            "risk_level": (
                "HIGH" if fraud_proba >= high_risk_threshold
                else "MEDIUM" if fraud_proba >= fraud_threshold
                else "LOW"
            ),
            "true_label": tx.get("isFraud"),        # for offline evaluation only
            "amount": tx.get("TransactionAmt"),
            "scored_at": scored_at,
            "latency_ms": round(per_tx_latency_ms, 3),
        })

    return results


def run_consumer():
    """Main consumer loop — polls, scores in batches, stores, then commits."""
    install_signal_handlers()

    log.info("Loading fraud model from MLflow...")
    model = load_model()

    # Metrics are secondary to scoring. A busy port used to abort startup with
    # OSError before a single transaction was processed -- which is easy to hit,
    # since redis-stack occupies the neighbouring 8001.
    try:
        start_http_server(METRICS_PORT)
        log.info(f"Prometheus metrics on :{METRICS_PORT}/metrics")
    except OSError as e:
        log.error(
            f"Could not bind metrics port {METRICS_PORT} ({e}). "
            f"Continuing WITHOUT metrics — set METRICS_PORT to a free port."
        )

    cat_maps, feature_columns, policy = load_preprocessing_artifacts()
    fraud_threshold, high_risk_threshold = load_threshold_config()

    # Sanity check the artifact feature list against the model's own.
    # XGBoost is positional, so a mismatch means silently wrong scores.
    try:
        model_features = get_feature_columns(model)
        if model_features != list(feature_columns):
            log.warning(
                f"Feature mismatch! artifacts={len(feature_columns)} "
                f"model={len(model_features)} — using the model's order"
            )
            missing = set(model_features) - set(feature_columns)
            if missing:
                log.warning(f"  in model but not artifacts: {sorted(missing)[:10]}")
            feature_columns = model_features
    except RuntimeError as e:
        log.warning(f"Could not verify feature order against model: {e}")

    log.info(f"Model ready | Features: {len(feature_columns)}")
    log.info("Connecting to Redis...")
    r = create_redis_client()

    consumer = create_consumer()
    dlq = create_dlq_producer()

    # Tracking stats
    processed = 0
    fraud_caught = 0
    failed = 0
    total_latency = 0.0
    start_time = time.time()

    log.info("Listening for transactions... (Ctrl+C to stop)\n")

    try:
        while not _shutdown:
            # poll() instead of iterating the consumer: the old code set
            # consumer_timeout_ms=600000, so the for-loop raised StopIteration
            # after 10 idle minutes and the process exited looking like a clean
            # shutdown. A service should idle indefinitely.
            batches = consumer.poll(timeout_ms=POLL_TIMEOUT_MS,
                                    max_records=BATCH_MAX_RECORDS)
            if not batches:
                continue

            txs = [msg.value for records in batches.values() for msg in records]
            BATCH_SIZE.observe(len(txs))

            try:
                results = score_batch(
                    model, feature_columns, cat_maps, policy, txs,
                    fraud_threshold, high_risk_threshold,
                )
                store_predictions(r, results)

                # Commit only once the batch is durably in Redis.
                consumer.commit()

            except Exception as e:
                # Batch-level failure: retry each row alone so one poison message
                # cannot discard up to BATCH_MAX_RECORDS good transactions.
                log.error(f"Batch of {len(txs)} failed ({e}) — retrying individually")
                results = []
                for tx in txs:
                    try:
                        one = score_batch(
                            model, feature_columns, cat_maps, policy, [tx],
                            fraud_threshold, high_risk_threshold,
                        )
                        store_predictions(r, one)
                        results.extend(one)
                    except Exception as inner:
                        failed += 1
                        FAILURES_TOTAL.inc()
                        log.error(
                            f"Failed to score transaction "
                            f"{tx.get('TransactionID')}: {inner}"
                        )
                        if dlq is not None:
                            dlq.send(DLQ_TOPIC, {
                                "error": str(inner),
                                "failed_at": datetime.now(timezone.utc).isoformat(),
                                "transaction": tx,
                            })
                consumer.commit()

            # Metrics + logging for whatever succeeded
            for result in results:
                TRANSACTIONS_TOTAL.inc()
                LATENCY.observe(result["latency_ms"])
                processed += 1
                total_latency += result["latency_ms"]

                if result["is_fraud"]:
                    FRAUDS_TOTAL.inc()
                    fraud_caught += 1
                    log.warning(
                        f"🚨 FRAUD DETECTED | "
                        f"ID: {result['transaction_id']} | "
                        f"Score: {result['fraud_score']:.3f} | "
                        f"Risk: {result['risk_level']} | "
                        f"Amount: ${result['amount']}"
                    )

            # Progress summary every 500 transactions
            if processed and processed // 500 != (processed - len(results)) // 500:
                elapsed = time.time() - start_time
                log.info(
                    f"Scored {processed:,} txns | "
                    f"Fraud: {fraud_caught:,} ({fraud_caught/processed:.2%}) | "
                    f"Failed: {failed} | "
                    f"Avg latency: {total_latency/processed:.2f}ms | "
                    f"Throughput: {processed/max(elapsed,1e-9):.0f} tx/sec"
                )

    except KeyboardInterrupt:
        log.info("\nStopped by user")

    finally:
        try:
            consumer.commit()
        except Exception:
            pass
        consumer.close()
        if dlq is not None:
            dlq.flush()
            dlq.close()

        elapsed = max(time.time() - start_time, 1e-9)
        log.info(f"\n{'='*50}")
        log.info(f"Total processed : {processed:,}")
        log.info(f"Fraud caught    : {fraud_caught:,} ({fraud_caught/max(processed,1):.2%})")
        log.info(f"Failed          : {failed:,}")
        log.info(f"Avg latency     : {total_latency/max(processed,1):.2f}ms")
        log.info(f"Throughput      : {processed/elapsed:.0f} tx/sec")
        log.info(f"{'='*50}")


# ─── ENTRYPOINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_consumer()

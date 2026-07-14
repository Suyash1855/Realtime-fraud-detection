"""
Kafka Consumer — Real-Time Fraud Scorer
=======================================
Reads transactions from Kafka, engineers features, scores with the
XGBoost model loaded from MLflow, and stores results in Redis.

Run AFTER the producer is running.
Usage: python consumer.py
"""

import json
import time
import logging
import numpy as np
import pandas as pd
import redis
import mlflow.xgboost
from datetime import datetime
from kafka import KafkaConsumer
from kafka.errors import KafkaConnectionError
from prometheus_client import Counter, Histogram, start_http_server
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [CONSUMER] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


# ─── CONFIG ──────────────────────────────────────────────────────────────────

KAFKA_BROKER = "localhost:9092"
TOPIC = "transactions"
CONSUMER_GROUP = "fraud-scorer"

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_TTL_SECONDS = 3600       # keep predictions for 1 hour

MLFLOW_TRACKING_URI = "http://localhost:5001"
MODEL_NAME = "fraud-xgboost"
MODEL_STAGE = "Production"     # or "latest" while developing

FRAUD_THRESHOLD = 0.5          # score above this → flagged as fraud
HIGH_RISK_THRESHOLD = 0.8      # score above this → HIGH RISK alert
TRANSACTIONS_TOTAL = Counter(
    "transactions_processed_total",
    "Total transactions processed"
)

FRAUDS_TOTAL = Counter(
    "frauds_detected_total",
    "Total frauds detected"
)

LATENCY = Histogram(
    "model_latency_ms",
    "Model inference latency"
)

# ─── FEATURE ENGINEERING ─────────────────────────────────────────────────────
# Must exactly match what was done during training (train_fraud_model.py)

def engineer_features(tx: dict) -> dict:
    """Replicate the same feature engineering used at training time."""
    dt = tx.get("TransactionDT", 0)

    tx["hour"] = int(dt / 3600) % 24
    tx["day_of_week"] = int(dt / (3600 * 24)) % 7
    tx["is_night"] = 1 if (tx["hour"] < 6 or tx["hour"] > 22) else 0

    amount = float(tx.get("TransactionAmt", 0) or 0)
    tx["log_amount"] = np.log1p(amount)
    tx["amount_rounded"] = 1 if (amount % 1 == 0) else 0

    addr1 = tx.get("addr1")
    addr2 = tx.get("addr2")
    tx["addr_mismatch"] = 1 if (addr1 != addr2 and addr1 and addr2) else 0

    high_risk_domains = {"gmail.com", "yahoo.com", "hotmail.com"}
    email = tx.get("P_emaildomain", "") or ""
    tx["risky_email"] = 1 if email in high_risk_domains else 0

    return tx


def preprocess_transaction(tx: dict, feature_columns: list) -> pd.DataFrame:
    """
    Convert a raw transaction dict into a single-row DataFrame
    with the exact columns the model was trained on.
    """
    tx = engineer_features(tx)

    # Drop columns not used during training
    drop_cols = {"TransactionID", "TransactionDT", "isFraud", "ingested_at"}
    tx = {k: v for k, v in tx.items() if k not in drop_cols}

    # Encode strings as integers (same strategy as training)
    for k, v in tx.items():
        if isinstance(v, str):
            tx[k] = hash(v) % 10000  # simple consistent encoding
        elif v is None:
            tx[k] = -999             # sentinel for missing values

    # Build a DataFrame with the exact training columns
    row = {col: tx.get(col, -999) for col in feature_columns}
    return pd.DataFrame([row])


# ─── MODEL LOADER ────────────────────────────────────────────────────────────

def load_model():
    """
    Load the registered XGBoost model from MLflow Model Registry.
    Falls back to loading the latest run if no Production model exists.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    try:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        model = mlflow.xgboost.load_model(model_uri)
        log.info(f"Loaded model: {MODEL_NAME} ({MODEL_STAGE})")
    except Exception:
        log.warning(f"No '{MODEL_STAGE}' model found — loading latest run instead")
        model_uri = f"models:/{MODEL_NAME}/latest"
        model = mlflow.xgboost.load_model(model_uri)

    return model


def get_feature_columns(model) -> list:
    """
    Retrieve the feature names the model was trained on.
    XGBoost stores these in model.feature_names.
    """
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    elif hasattr(model, "feature_names"):
        return model.feature_names
    else:
        raise RuntimeError(
            "Cannot determine model feature columns. "
            "Make sure you saved feature names during training."
        )


# ─── REDIS CLIENT ────────────────────────────────────────────────────────────

def create_redis_client() -> redis.Redis:
    client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    client.ping()  # raises if Redis is not running
    log.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
    return client


def store_prediction(r: redis.Redis, transaction_id: str, result: dict):
    """
    Store the prediction result in Redis.
    Key format: prediction:<TransactionID>
    Also push flagged transactions to a sorted set for easy retrieval.
    """
    key = f"prediction:{transaction_id}"
    r.setex(key, REDIS_TTL_SECONDS, json.dumps(result))

    # If fraud, add to sorted set (score = fraud probability for ranking)
    if result["is_fraud"]:
        r.zadd("flagged_transactions", {transaction_id: result["fraud_score"]})

    # Increment counters for the /stats endpoint
    r.incr("stats:total_scored")
    if result["is_fraud"]:
        r.incr("stats:total_fraud")
    if result["risk_level"] == "HIGH":
        r.incr("stats:high_risk")


# ─── CONSUMER ────────────────────────────────────────────────────────────────

def create_consumer(retries: int = 5) -> KafkaConsumer:
    for attempt in range(1, retries + 1):
        try:
            consumer = KafkaConsumer(
                TOPIC,
                bootstrap_servers=KAFKA_BROKER,
                group_id=CONSUMER_GROUP,
                auto_offset_reset="earliest",   # read from start if no offset saved
                enable_auto_commit=True,
                auto_commit_interval_ms=1000,
                value_deserializer=lambda b: json.loads(b.decode("utf-8")),
                # How long to wait for new messages before polling again
                consumer_timeout_ms=600000,
            )
            log.info(f"Subscribed to topic '{TOPIC}' as group '{CONSUMER_GROUP}'")
            return consumer
        except KafkaConnectionError:
            log.warning(f"Kafka not ready (attempt {attempt}/{retries}) — retrying in 3s...")
            time.sleep(3)

    raise RuntimeError("Could not connect to Kafka.")


def score_transaction(model, feature_columns: list, tx: dict) -> dict:
    """
    Run the full inference pipeline for one transaction.
    Returns a result dict with score, label, latency.
    """
    t0 = time.perf_counter()

    X = preprocess_transaction(tx, feature_columns)
    fraud_proba = float(model.predict_proba(X)[0][1])

    latency_ms = (time.perf_counter() - t0) * 1000

    is_fraud = fraud_proba >= FRAUD_THRESHOLD
    risk_level = (
        "HIGH" if fraud_proba >= HIGH_RISK_THRESHOLD
        else "MEDIUM" if fraud_proba >= FRAUD_THRESHOLD
        else "LOW"
    )

    return {
        "transaction_id": str(tx.get("TransactionID")),
        "fraud_score": round(fraud_proba, 4),
        "is_fraud": is_fraud,
        "risk_level": risk_level,
        "true_label": tx.get("isFraud"),        # for offline evaluation only
        "amount": tx.get("TransactionAmt"),
        "scored_at": datetime.utcnow().isoformat(),
        "latency_ms": round(latency_ms, 2),
    }


def run_consumer():
    """Main consumer loop — reads, scores, stores."""
    log.info("Loading fraud model from MLflow...")
    model = load_model()
    start_http_server(8001)
    feature_columns = get_feature_columns(model)
    log.info(f"Model ready | Features: {len(feature_columns)}")

    log.info("Connecting to Redis...")
    r = create_redis_client()

    consumer = create_consumer()

    # Tracking stats
    processed = 0
    fraud_caught = 0
    total_latency = 0.0
    start_time = time.time()

    log.info("Listening for transactions... (Ctrl+C to stop)\n")

    try:
        for message in consumer:
            tx = message.value

            try:
                result = score_transaction(model, feature_columns, tx)
                store_prediction(r, result["transaction_id"], result)
                # Prometheus metrics
                TRANSACTIONS_TOTAL.inc()

                LATENCY.observe(result["latency_ms"])

                if result["is_fraud"]:
                    FRAUDS_TOTAL.inc()

                processed += 1
                total_latency += result["latency_ms"]
                if result["is_fraud"]:
                    fraud_caught += 1

                # Log every flagged transaction immediately
                if result["is_fraud"]:
                    log.warning(
                        f"🚨 FRAUD DETECTED | "
                        f"ID: {result['transaction_id']} | "
                        f"Score: {result['fraud_score']:.3f} | "
                        f"Risk: {result['risk_level']} | "
                        f"Amount: ${result['amount']}"
                    )

                # Progress summary every 500 transactions
                if processed % 500 == 0:
                    elapsed = time.time() - start_time
                    avg_latency = total_latency / processed
                    throughput = processed / elapsed
                    log.info(
                        f"Scored {processed:,} txns | "
                        f"Fraud: {fraud_caught:,} ({fraud_caught/processed:.2%}) | "
                        f"Avg latency: {avg_latency:.1f}ms | "
                        f"Throughput: {throughput:.0f} tx/sec"
                    )

            except Exception as e:
                log.error(f"Failed to score transaction {tx.get('TransactionID')}: {e}")
                # Continue processing — don't let one bad message crash the consumer
                continue

    except KeyboardInterrupt:
        log.info("\nStopped by user")

    finally:
        consumer.close()

        elapsed = time.time() - start_time
        log.info(f"\n{'='*50}")
        log.info(f"Total processed : {processed:,}")
        log.info(f"Fraud caught    : {fraud_caught:,} ({fraud_caught/max(processed,1):.2%})")
        log.info(f"Avg latency     : {total_latency/max(processed,1):.1f}ms")
        log.info(f"Throughput      : {processed/max(elapsed,1):.0f} tx/sec")
        log.info(f"{'='*50}")


# ─── ENTRYPOINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_consumer()

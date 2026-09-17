"""
Fraud Detection API
===================
Accepts transactions onto the Kafka topic and serves the predictions that
consumer.py writes into Redis.

Run: uvicorn API.main:app --port 8000
Metrics are exposed on /metrics for Prometheus (see API/prometheus.yml).
"""

import json
import logging
import os
import secrets
from contextlib import asynccontextmanager
from typing import List, Optional

import redis
from fastapi import Body, Depends, FastAPI, Header, HTTPException, Query, status
from kafka import KafkaProducer
from kafka.errors import KafkaError
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, ConfigDict, Field

from producer import build_transaction, serialize

log = logging.getLogger(__name__)

# ─── CONFIG ──────────────────────────────────────────────────────────────────
# Env-driven so this runs unchanged locally and inside Docker.
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")   # "redis-stack" inside Docker
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD") or None

API_KEY = os.getenv("API_KEY", "")

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
TOPIC = os.getenv("KAFKA_TOPIC", "transactions")
INGEST_MAX_BATCH = int(os.getenv("INGEST_MAX_BATCH", "500"))
INGEST_ACK_TIMEOUT = float(os.getenv("INGEST_ACK_TIMEOUT", "10"))

INGESTED_TOTAL = Counter(
    "transactions_ingested_total",
    "Transactions accepted onto the Kafka topic",
)
INGEST_REJECTED_TOTAL = Counter(
    "transactions_ingest_rejected_total",
    "Transactions the broker refused",
)

_kafka: Optional[KafkaProducer] = None

# Result of the last publish. /health reports this instead of probing: a probe
# that actually contacts a dead broker blocks for max_block_ms, which is longer
# than the healthcheck timeout, so it would fail the container over an
# unavailable Kafka that the read endpoints do not need.
_ingest_state = "unknown"


def _create_kafka_producer() -> Optional[KafkaProducer]:
    try:
        return KafkaProducer(
            bootstrap_servers=KAFKA_BROKER,
            value_serializer=serialize,
            acks="all",
            retries=3,
            linger_ms=5,
            compression_type="gzip",
            # Default is 60s. A request handler must not sit that long waiting
            # for metadata from a broker that is not coming back.
            max_block_ms=int(INGEST_ACK_TIMEOUT * 1000),
        )
    except Exception as e:
        log.error("Kafka unavailable, ingestion disabled: %s", e)
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _kafka
    # Reads must keep working when the broker is down, so a failure here is
    # logged and the ingest endpoints answer 503 rather than the process dying.
    _kafka = _create_kafka_producer()
    if _kafka:
        log.info("Kafka producer ready at %s", KAFKA_BROKER)
    yield
    if _kafka:
        _kafka.flush(timeout=INGEST_ACK_TIMEOUT)
        _kafka.close(timeout=INGEST_ACK_TIMEOUT)


app = FastAPI(
    title="Fraud Detection API",
    description="Transaction ingestion and real-time fraud scores.",
    version="1.1.0",
    lifespan=lifespan,
)
Instrumentator().instrument(app).expose(app)

# A connection pool, not a bare connection: FastAPI runs these sync handlers in a
# threadpool, so several requests can touch Redis concurrently.
_pool = redis.ConnectionPool(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    decode_responses=True,
    max_connections=20,
    socket_connect_timeout=2,
    socket_timeout=2,
)
r = redis.Redis(connection_pool=_pool)


def require_api_key(x_api_key: str = Header(default="")):
    """
    Applied to every endpoint that reads transaction data or writes to the
    topic. /health and /metrics stay open so probes and Prometheus keep working.
    """
    if not API_KEY:
        raise HTTPException(status_code=503, detail="API_KEY is not configured")
    if not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


def _redis_or_503():
    """
    Every endpoint depends on Redis. If it is down we must return 503, not a
    500 traceback -- the original code assumed Redis was always reachable.
    """
    try:
        r.ping()
    except redis.RedisError as e:
        log.error("Redis unavailable: %s", e)
        raise HTTPException(status_code=503, detail="Redis unavailable")


@app.get("/health")
def health():
    """Liveness + dependency check, for Docker/k8s probes."""
    try:
        r.ping()
    except redis.RedisError:
        raise HTTPException(status_code=503, detail="Redis unavailable")
    # Kafka being down does not make this unhealthy: reads are the API's main
    # job and they only need Redis. Ingestion reports its own 503.
    return {"status": "ok", "redis": "up", "kafka": _ingest_state}


# ─── INGESTION ───────────────────────────────────────────────────────────────

class Transaction(BaseModel):
    """
    The three fields the scorer cannot work without, plus anything else the
    caller sends. extra="allow" is deliberate: the model reads 439 columns and
    fills what is absent from the sentinels in encoding_config.json, so pinning
    a schema here would reject valid traffic every time training changes.
    """

    model_config = ConfigDict(extra="allow")

    TransactionID: int
    TransactionAmt: float = Field(ge=0)
    TransactionDT: int = Field(ge=0)


def _kafka_or_503() -> KafkaProducer:
    global _kafka, _ingest_state
    if _kafka is None:
        _kafka = _create_kafka_producer()
    if _kafka is None:
        _ingest_state = "down"
        raise HTTPException(status_code=503, detail="Kafka unavailable")
    return _kafka


def _publish(producer: KafkaProducer, payloads: List[dict]) -> List[dict]:
    """Send, then wait once for the whole batch rather than per message."""
    global _ingest_state
    futures = []
    try:
        for payload in payloads:
            # include_label=False: ground truth must not enter through the
            # ingress. Accepting isFraud from a caller is label leakage.
            tx = build_transaction(payload, include_label=False)
            key = str(tx["TransactionID"]).encode("utf-8")
            futures.append(
                (tx["TransactionID"], producer.send(TOPIC, key=key, value=tx))
            )
        producer.flush(timeout=INGEST_ACK_TIMEOUT)
    except KafkaError as e:
        # An unreachable broker surfaces here, on send or flush. Without this it
        # leaves the handler as a 500, which tells a caller to stop retrying.
        _ingest_state = "down"
        log.error("Publish to %s failed: %s", TOPIC, e)
        raise HTTPException(status_code=503, detail=f"Kafka unavailable: {e}")

    _ingest_state = "up"

    failures = []
    for tx_id, future in futures:
        if not future.succeeded():
            # An unresolved future means flush hit INGEST_ACK_TIMEOUT, and then
            # there is no exception to report.
            err = future.exception or "timed out waiting for broker ack"
            failures.append({"transaction_id": tx_id, "error": str(err)})
    return failures


@app.post("/transactions", status_code=status.HTTP_202_ACCEPTED,
          dependencies=[Depends(require_api_key)])
def ingest_transaction(transaction: Transaction):
    """
    Publish one transaction for scoring. Keyed by TransactionID so every event
    for a transaction lands on the same partition.
    """
    producer = _kafka_or_503()
    failures = _publish(producer, [transaction.model_dump()])

    if failures:
        INGEST_REJECTED_TOTAL.inc()
        raise HTTPException(status_code=502, detail=failures[0]["error"])

    INGESTED_TOTAL.inc()
    return {"accepted": 1, "transaction_id": transaction.TransactionID}


@app.post("/transactions/batch", status_code=status.HTTP_202_ACCEPTED,
          dependencies=[Depends(require_api_key)])
def ingest_batch(transactions: List[Transaction] = Body(..., min_length=1)):
    """Publish many transactions in one round trip."""
    if len(transactions) > INGEST_MAX_BATCH:
        raise HTTPException(
            status_code=413,
            detail=f"Batch of {len(transactions)} exceeds INGEST_MAX_BATCH "
                   f"({INGEST_MAX_BATCH})",
        )

    producer = _kafka_or_503()
    failures = _publish(producer, [t.model_dump() for t in transactions])

    accepted = len(transactions) - len(failures)
    INGESTED_TOTAL.inc(accepted)
    INGEST_REJECTED_TOTAL.inc(len(failures))
    return {"accepted": accepted, "rejected": len(failures), "failures": failures}


@app.get("/stats", dependencies=[Depends(require_api_key)])
def get_stats():
    """Aggregate counters maintained by the consumer."""
    _redis_or_503()

    # One round trip instead of three.
    total_scored, total_fraud, high_risk = (
        int(v or 0)
        for v in r.mget("stats:total_scored", "stats:total_fraud", "stats:high_risk")
    )

    fraud_rate = (
        round((total_fraud / total_scored) * 100, 2) if total_scored > 0 else 0.0
    )

    return {
        "total_scored": total_scored,
        "total_fraud": total_fraud,
        "high_risk": high_risk,
        "fraud_rate": fraud_rate,
    }


@app.get("/frauds", dependencies=[Depends(require_api_key)])
def get_frauds(
    limit: int = Query(10, ge=1, le=500, description="How many to return"),
    offset: int = Query(0, ge=0, description="Rank to start from"),
):
    """
    Highest-scoring flagged transactions, ranked by fraud probability.
    Paginated -- the original hardcoded the top 10 with no way to page.
    """
    _redis_or_503()

    frauds = r.zrevrange(
        "flagged_transactions",
        offset,
        offset + limit - 1,
        withscores=True,
    )

    return {
        "count": len(frauds),
        "offset": offset,
        "limit": limit,
        "results": [
            {"transaction_id": tx_id, "score": score} for tx_id, score in frauds
        ],
    }


@app.get("/transaction/{tx_id}", dependencies=[Depends(require_api_key)])
def get_transaction(tx_id: str):
    """
    Full scoring result for one transaction.

    Note: predictions carry a TTL (REDIS_TTL_SECONDS in consumer.py), so a 404
    here can mean either "never scored" or "scored but expired".
    """
    _redis_or_503()

    data = r.get(f"prediction:{tx_id}")

    if not data:
        # Was previously a 200 with {"error": ...}, which no HTTP client can
        # distinguish from a successful lookup.
        raise HTTPException(status_code=404, detail=f"No prediction for {tx_id}")

    try:
        return json.loads(data)
    except json.JSONDecodeError:
        log.error("Corrupt payload stored for %s", tx_id)
        raise HTTPException(status_code=500, detail="Corrupt prediction payload")

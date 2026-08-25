"""
Fraud Detection Read API
========================
Serves the predictions that consumer.py writes into Redis.

Run: uvicorn API.main:app --port 8000
Metrics are exposed on /metrics for Prometheus (see API/prometheus.yml).
"""

import json
import logging
import os

import redis
from fastapi import FastAPI, HTTPException, Query
from prometheus_fastapi_instrumentator import Instrumentator

log = logging.getLogger(__name__)

# ─── CONFIG ──────────────────────────────────────────────────────────────────
# Env-driven so this runs unchanged locally and inside Docker.
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")   # "redis-stack" inside Docker
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

app = FastAPI(
    title="Fraud Detection API",
    description="Read-only access to real-time fraud scores.",
    version="1.0.0",
)
Instrumentator().instrument(app).expose(app)

# A connection pool, not a bare connection: FastAPI runs these sync handlers in a
# threadpool, so several requests can touch Redis concurrently.
_pool = redis.ConnectionPool(
    host=REDIS_HOST,
    port=REDIS_PORT,
    decode_responses=True,
    max_connections=20,
    socket_connect_timeout=2,
    socket_timeout=2,
)
r = redis.Redis(connection_pool=_pool)


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
        return {"status": "ok", "redis": "up"}
    except redis.RedisError:
        raise HTTPException(status_code=503, detail="Redis unavailable")


@app.get("/stats")
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


@app.get("/frauds")
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


@app.get("/transaction/{tx_id}")
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

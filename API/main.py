from fastapi import FastAPI
import redis
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()
Instrumentator().instrument(app).expose(app)

r = redis.Redis(
    host="localhost",  # or "redis" if FastAPI runs inside Docker
    port=6379,
    decode_responses=True
)

@app.get("/stats")
def get_stats():

    total_scored = int(r.get("stats:total_scored") or 0)
    total_fraud = int(r.get("stats:total_fraud") or 0)
    high_risk = int(r.get("stats:high_risk") or 0)

    fraud_rate = (
        round((total_fraud / total_scored) * 100, 2)
        if total_scored > 0
        else 0
    )

    return {
        "total_scored": total_scored,
        "total_fraud": total_fraud,
        "high_risk": high_risk,
        "fraud_rate": fraud_rate
    }


@app.get("/frauds")
def get_frauds():

    frauds = r.zrevrange(
        "flagged_transactions",
        0,
        9,
        withscores=True
    )

    return [
        {
            "transaction_id": tx_id,
            "score": score
        }
        for tx_id, score in frauds
    ]


@app.get("/transaction/{tx_id}")
def get_transaction(tx_id: str):

    data = r.get(f"prediction:{tx_id}")

    if not data:
        return {"error": "Not found"}

    return json.loads(data)

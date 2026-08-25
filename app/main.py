"""
Fraud Detection — demo application
==================================
One deployable unit: FastAPI serving both the JSON API and the built frontend.
No Kafka, no Redis, no MLflow at runtime -- the model and its contract are frozen
into deploy_bundle/ at build time by scripts/export_for_deploy.py.

What is real here and what is simulated is stated explicitly at /api/info, and
surfaced in the UI. The model, the feature pipeline and the measured latency are
real; the transport is an in-process clock instead of Kafka.

Run locally:  uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.analytics import Analytics
from app.scoring import ScoringEngine
from app.simulator import SimulationSession

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [APP] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
BUNDLE_DIR = Path(os.getenv("BUNDLE_DIR", ROOT / "deploy_bundle"))
DEMO_DIR = Path(os.getenv("DEMO_DATA_DIR", ROOT / "demo_data"))
STATIC_DIR = Path(os.getenv("STATIC_DIR", ROOT / "web" / "dist"))

# Measured, not guessed. On a 10-core dev machine the aggregate scoring ceiling is
# ~936 tx/s, and API latency (which is what the threshold slider feels) stays under
# 200 ms up to about 4 concurrent sessions:
#
#   sessions   per-stream tx/s   aggregate   API p50
#          1               253         253       2 ms
#          2               252         504       2 ms
#          4               231         924     133 ms
#          8               116         928     468 ms   <- slider feels broken
#         12                78         936     713 ms
#
# A Fly shared-cpu-1x has roughly a tenth of this CPU, so the cap is deliberately
# conservative. Past the cap /api/stream returns 429 rather than degrading everyone.
MAX_CONCURRENT_SESSIONS = int(os.getenv("MAX_CONCURRENT_SESSIONS", "6"))

state: dict = {}
sessions: Dict[int, SimulationSession] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading serving bundle from %s", BUNDLE_DIR)
    engine = ScoringEngine(BUNDLE_DIR)

    # csv.gz is the current format (no pyarrow needed). Parquet is still read if
    # an older demo_data/ directory is present, so a stale bundle does not break
    # the app -- but new builds should be csv.gz.
    csv_path = DEMO_DIR / "demo_transactions.csv.gz"
    parquet_path = DEMO_DIR / "demo_transactions.parquet"
    if csv_path.exists():
        data = pd.read_csv(csv_path, low_memory=False)
    elif parquet_path.exists():
        log.warning("Reading legacy parquet demo data; re-run build_demo_data.py")
        data = pd.read_parquet(parquet_path)
    else:
        raise FileNotFoundError(
            f"{csv_path} missing. Run: python scripts/build_demo_data.py"
        )
    meta = json.loads((DEMO_DIR / "demo_metadata.json").read_text())
    log.info("Loaded %d demo transactions", len(data))

    # Score the whole demo set once. Everything on the analytics screen and the
    # threshold slider then derives from these numbers with no further inference,
    # which is what makes the slider feel instantaneous.
    scores = engine.score_frame(data)
    labels = data["isFraud"].to_numpy() if "isFraud" in data.columns else np.zeros(len(data))
    amounts = data["TransactionAmt"].to_numpy(dtype=float)
    analytics = Analytics(scores, labels, amounts)
    log.info(
        "Precomputed %d scores | AUC-equivalent separation ready | fraud=%d",
        len(scores), analytics.n_fraud,
    )

    state.update(
        engine=engine, data=data, meta=meta,
        scores=scores, analytics=analytics,
        index_by_id={str(tid): i for i, tid in enumerate(data["TransactionID"])},
    )
    yield

    for s in list(sessions.values()):
        s.stop()
    sessions.clear()


app = FastAPI(
    title="Fraud Detection — Real-Time Demo",
    description="Live transaction scoring with an XGBoost model served from a frozen MLflow bundle.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


# ─── META ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": "engine" in state}


@app.get("/api/info")
def info():
    """Everything the landing page needs, including what is real vs simulated."""
    engine: ScoringEngine = state["engine"]
    analytics: Analytics = state["analytics"]
    meta = state["meta"]

    return {
        "model": engine.model_info(),
        "dataset": {
            **analytics.summary(),
            "identity_coverage": meta.get("identity_coverage"),
            "provenance": meta.get("provenance"),
            "source": meta.get("source"),
        },
        "architecture": {
            "real": [
                "XGBoost model loaded from a frozen MLflow registry bundle",
                "Feature engineering shared with training (features.py)",
                "Per-batch inference, latency measured live",
                "Threshold loaded from a versioned artifact",
            ],
            "simulated": [
                "Transaction transport: in-process clock instead of Kafka",
                "Results held in memory instead of Redis",
            ],
            "local_only": [
                "Kafka + Zookeeper (producer.py -> consumer.py)",
                "Redis for prediction storage",
                "MLflow tracking + model registry",
                "Prometheus + Grafana",
            ],
        },
    }


# ─── ANALYTICS ───────────────────────────────────────────────────────────────

@app.get("/api/analytics/threshold")
def threshold_metrics(t: float = Query(..., ge=0.0, le=1.0)):
    return state["analytics"].at_threshold(t)


@app.get("/api/analytics/pr-curve")
def pr_curve(points: int = Query(120, ge=20, le=400)):
    a: Analytics = state["analytics"]
    return {
        "pr": a.pr_curve(points),
        "roc": a.roc_curve(points),
        "current_threshold": state["engine"].contract.fraud_threshold,
    }


@app.get("/api/analytics/distribution")
def distribution(bins: int = Query(40, ge=10, le=100)):
    return state["analytics"].score_distribution(bins)


@app.get("/api/analytics/cost")
def cost(fp_cost: float = Query(5.0, ge=0.0), fn_cost: float = Query(100.0, ge=0.0)):
    """
    Threshold sweep under an explicit cost model.
    fp_cost: cost of blocking a legitimate transaction.
    fn_cost: cost of missing a fraudulent one.
    """
    return state["analytics"].cost_sweep(fp_cost, fn_cost)


# ─── EXPLAINABILITY ──────────────────────────────────────────────────────────

@app.get("/api/transaction/{tx_id}")
def transaction_detail(tx_id: str, explain: bool = True):
    """One transaction, its score, and why the model scored it that way."""
    idx = state["index_by_id"].get(str(tx_id))
    if idx is None:
        raise HTTPException(status_code=404, detail=f"Transaction {tx_id} not in demo set")

    engine: ScoringEngine = state["engine"]
    data: pd.DataFrame = state["data"]
    row = data.iloc[[idx]]
    score = float(state["scores"][idx])
    is_fraud, risk = engine.classify(score)

    payload = {
        "transaction_id": str(tx_id),
        "score": round(score, 4),
        "is_flagged": is_fraud,
        "risk_level": risk,
        "true_label": (
            int(row["isFraud"].iloc[0]) if "isFraud" in row.columns else None
        ),
        "amount": float(row["TransactionAmt"].iloc[0]),
        "attributes": {
            k: (None if pd.isna(row[k].iloc[0]) else str(row[k].iloc[0]))
            for k in ["ProductCD", "card4", "card6", "P_emaildomain",
                      "R_emaildomain", "DeviceType", "DeviceInfo", "addr1"]
            if k in row.columns
        },
    }

    if explain:
        payload["explanation"] = engine.explain(row)

    return payload


# ─── LIVE SIMULATION ─────────────────────────────────────────────────────────

def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.get("/api/stream")
async def stream(
    request: Request,
    rate: int = Query(50, ge=1, le=2000, description="Transactions per second"),
    threshold: Optional[float] = Query(None, ge=0.0, le=1.0),
    loop: bool = Query(False, description="Restart when the dataset is exhausted"),
    shuffle: bool = Query(False, description="Randomise order for a varied demo"),
):
    """
    Server-Sent Events stream of live scoring.

    SSE rather than WebSockets: the flow is strictly server to client, it
    reconnects automatically, and it survives proxies that mangle upgrades.
    """
    if len(sessions) >= MAX_CONCURRENT_SESSIONS:
        raise HTTPException(
            status_code=429,
            detail="Too many concurrent simulations. Try again in a moment.",
        )

    data: pd.DataFrame = state["data"]

    # A shuffled INDEX, not a shuffled copy of the frame. `data.sample(frac=1.0)`
    # duplicated the whole ~52 MB DataFrame per session; a permutation of 12,000
    # int64 positions is ~94 KB, which is 571x smaller and behaves identically.
    order = np.random.permutation(len(data)) if shuffle else None

    session = SimulationSession(
        state["engine"], data, rate=rate, threshold=threshold, loop=loop,
        order=order,
    )
    sessions[session.id] = session

    async def generator():
        try:
            async for event in session.run():
                if await request.is_disconnected():
                    session.stop()
                    break
                yield _sse(event["event"], event["data"])
        except asyncio.CancelledError:
            session.stop()
            raise
        finally:
            sessions.pop(session.id, None)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",     # stop nginx/proxies buffering the stream
        },
    )


@app.post("/api/stream/{session_id}/threshold")
def update_session_threshold(session_id: int, value: float = Query(..., ge=0.0, le=1.0)):
    """Retune a running simulation without restarting it."""
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="No such active session")
    session.set_threshold(value)
    return {"session_id": session_id, "threshold": round(value, 4)}


@app.post("/api/stream/{session_id}/stop")
def stop_session(session_id: int):
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="No such active session")
    session.stop()
    return {"session_id": session_id, "stopped": True}


# ─── FRONTEND ────────────────────────────────────────────────────────────────
# Mounted last so it never shadows /api/*. html=True makes the SPA router work on
# deep links by falling back to index.html.

if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
    log.info("Serving frontend from %s", STATIC_DIR)
else:
    @app.get("/")
    def no_frontend():
        return {
            "message": "API is running; frontend not built yet.",
            "build": "cd web && npm install && npm run build",
            "docs": "/docs",
        }

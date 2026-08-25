"""
Transaction stream simulator
============================
Replays the demo dataset as a live stream and scores it in real time.

This is the one component the deployed demo swaps out. Locally, producer.py
publishes to Kafka and consumer.py scores from it; here the same rows are driven
by an in-process clock. The scoring path is identical -- same model, same
features.py, same micro-batching -- so the latency and throughput on screen are
genuinely measured, not replayed from a cache.

Each browser connection gets its own session, so two viewers do not fight over a
shared cursor.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# How often a tick fires. Each tick scores one micro-batch and emits one event,
# which keeps the browser from being flooded at high rates while still feeling
# continuous to the eye.
TICK_SECONDS = 0.2
MAX_FEED_ITEMS = 60          # rolling window kept per session for the live table

_session_ids = itertools.count(1)


@dataclass
class SessionStats:
    processed: int = 0
    flagged: int = 0
    high_risk: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    amount_total: float = 0.0
    amount_flagged: float = 0.0
    fraud_amount_caught: float = 0.0
    fraud_amount_missed: float = 0.0
    latency_samples: list = field(default_factory=list)
    started_at: float = field(default_factory=time.perf_counter)

    def latency_percentile(self, p: float) -> float:
        if not self.latency_samples:
            return 0.0
        return float(np.percentile(self.latency_samples, p))

    def as_dict(self, elapsed: float) -> dict:
        precision = (
            self.true_positives / self.flagged if self.flagged else 0.0
        )
        recall_denom = self.true_positives + self.false_negatives
        recall = self.true_positives / recall_denom if recall_denom else 0.0
        return {
            "processed": self.processed,
            "flagged": self.flagged,
            "high_risk": self.high_risk,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "flag_rate": round(self.flagged / self.processed, 4) if self.processed else 0.0,
            "throughput": round(self.processed / elapsed, 1) if elapsed > 0 else 0.0,
            "latency_p50_ms": round(self.latency_percentile(50), 3),
            "latency_p95_ms": round(self.latency_percentile(95), 3),
            "latency_p99_ms": round(self.latency_percentile(99), 3),
            "amount_total": round(self.amount_total, 2),
            "amount_flagged": round(self.amount_flagged, 2),
            "fraud_amount_caught": round(self.fraud_amount_caught, 2),
            "fraud_amount_missed": round(self.fraud_amount_missed, 2),
            "elapsed_seconds": round(elapsed, 1),
        }


class SimulationSession:
    """One replay of the demo dataset, driven by an async generator of events."""

    def __init__(self, engine, data: pd.DataFrame, rate: int = 50,
                 threshold: float | None = None, loop: bool = False,
                 order: np.ndarray | None = None):
        self.id = next(_session_ids)
        self.engine = engine
        # `data` is SHARED across sessions and must never be copied per session.
        # The demo frame is ~52 MB; copying it per viewer cost ~52 MB each and
        # measured 810 MB RSS at 12 concurrent sessions -- enough to OOM a 512 MB
        # box. Per-session ordering is carried by `order`, a 94 KB index
        # permutation, instead.
        self.data = data
        self.order = order
        self.rate = max(1, min(int(rate), 5000))
        self.threshold = (
            engine.contract.fraud_threshold if threshold is None else float(threshold)
        )
        self.high_risk = max(engine.contract.high_risk_threshold, self.threshold)
        self.loop = loop

        self.cursor = 0
        self.stats = SessionStats()
        self.feed: list = []
        self._stop = asyncio.Event()

    def stop(self):
        self._stop.set()

    def set_threshold(self, threshold: float):
        """
        Retune mid-stream. Only affects transactions scored from here on -- the
        already-emitted feed is not rewritten, which mirrors how a real deployment
        behaves when you change an operating point.
        """
        self.threshold = float(threshold)
        self.high_risk = max(self.engine.contract.high_risk_threshold, self.threshold)

    # ── the replay loop ──────────────────────────────────────────────────────

    async def run(self):
        """Yields event dicts. The caller serialises them as SSE frames."""
        per_tick = max(1, int(round(self.rate * TICK_SECONDS)))
        log.info(
            "session %d started | rate=%d/s batch=%d threshold=%.4f",
            self.id, self.rate, per_tick, self.threshold,
        )

        yield {
            "event": "started",
            "data": {
                "session_id": self.id,
                "rate": self.rate,
                "threshold": round(self.threshold, 4),
                "total_available": int(len(self.data)),
                "batch_size": per_tick,
            },
        }

        next_tick = time.perf_counter()

        while not self._stop.is_set():
            if self.cursor >= len(self.data):
                if self.loop:
                    self.cursor = 0
                else:
                    break

            if self.order is None:
                batch = self.data.iloc[self.cursor: self.cursor + per_tick]
            else:
                batch = self.data.iloc[self.order[self.cursor: self.cursor + per_tick]]
            self.cursor += len(batch)

            try:
                # Scoring is CPU-bound (pandas + XGBoost) and used to be called
                # directly here, inside the coroutine -- which blocked the whole
                # event loop for the duration. Measured: a trivial /api/health
                # call took 1.5 s with 8 concurrent sessions, so the threshold
                # slider and every other request stalled behind scoring.
                #
                # numpy and XGBoost release the GIL during compute, so handing
                # this to a worker thread genuinely parallelises it and keeps the
                # loop free to serve HTTP and other sessions' ticks.
                event = await asyncio.get_running_loop().run_in_executor(
                    None, self._score_and_record, batch
                )
            except Exception as e:                       # never kill the stream
                log.exception("scoring failed for session %d", self.id)
                yield {"event": "error", "data": {"message": str(e)}}
                break

            yield event

            # Pace the loop against a fixed schedule rather than sleeping a flat
            # interval, so scoring time does not cause the rate to drift.
            next_tick += TICK_SECONDS
            delay = next_tick - time.perf_counter()
            if delay > 0:
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=delay)
                    break                                # stop() fired during the wait
                except asyncio.TimeoutError:
                    pass
            else:
                next_tick = time.perf_counter()          # we fell behind; resync
                await asyncio.sleep(0)

        elapsed = time.perf_counter() - self.stats.started_at
        yield {
            "event": "complete",
            "data": {
                "session_id": self.id,
                "reason": "stopped" if self._stop.is_set() else "exhausted",
                "stats": self.stats.as_dict(elapsed),
            },
        }
        log.info("session %d finished after %d transactions", self.id, self.stats.processed)

    # ── scoring one micro-batch ──────────────────────────────────────────────

    def _score_and_record(self, batch: pd.DataFrame) -> dict:
        proba, elapsed_ms = self.engine.score_batch(batch)
        per_tx_ms = elapsed_ms / max(len(batch), 1)

        ids = batch["TransactionID"].to_numpy()
        amounts = np.nan_to_num(batch["TransactionAmt"].to_numpy(dtype=float))
        # isFraud is ground truth, used only to score the demo's own accuracy.
        # A production stream would not have it.
        labels = (
            batch["isFraud"].to_numpy() if "isFraud" in batch.columns
            else np.full(len(batch), -1)
        )

        flagged_items = []
        for i in range(len(batch)):
            p = float(proba[i])
            is_flagged = p >= self.threshold
            risk = "HIGH" if p >= self.high_risk else "MEDIUM" if is_flagged else "LOW"
            label = int(labels[i]) if labels[i] in (0, 1) else None
            amount = float(amounts[i])

            self.stats.processed += 1
            self.stats.amount_total += amount

            if is_flagged:
                self.stats.flagged += 1
                self.stats.amount_flagged += amount
                if risk == "HIGH":
                    self.stats.high_risk += 1
                if label == 1:
                    self.stats.true_positives += 1
                    self.stats.fraud_amount_caught += amount
                elif label == 0:
                    self.stats.false_positives += 1
            elif label == 1:
                self.stats.false_negatives += 1
                self.stats.fraud_amount_missed += amount

            if is_flagged:
                flagged_items.append({
                    "transaction_id": str(ids[i]),
                    "score": round(p, 4),
                    "risk_level": risk,
                    "amount": round(amount, 2),
                    "true_label": label,
                    "correct": (label == 1) if label is not None else None,
                    "product": _safe_str(batch, "ProductCD", i),
                    "card_type": _safe_str(batch, "card4", i),
                    "email_domain": _safe_str(batch, "P_emaildomain", i),
                    "device_type": _safe_str(batch, "DeviceType", i),
                })

        self.stats.latency_samples.append(per_tx_ms)
        if len(self.stats.latency_samples) > 2000:
            del self.stats.latency_samples[:1000]

        self.feed.extend(flagged_items)
        if len(self.feed) > MAX_FEED_ITEMS:
            self.feed = self.feed[-MAX_FEED_ITEMS:]

        elapsed = time.perf_counter() - self.stats.started_at
        return {
            "event": "batch",
            "data": {
                "batch_size": int(len(batch)),
                "batch_latency_ms": round(elapsed_ms, 3),
                "per_tx_latency_ms": round(per_tx_ms, 4),
                "flagged": flagged_items,
                "score_samples": [round(float(p), 4) for p in proba[:40]],
                "progress": round(self.cursor / len(self.data), 4),
                "stats": self.stats.as_dict(elapsed),
            },
        }


def _safe_str(df: pd.DataFrame, col: str, i: int):
    if col not in df.columns:
        return None
    val = df[col].iloc[i]
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return str(val)

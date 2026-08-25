"""
Kafka Producer — Transaction Stream Simulator
=============================================
Reads the IEEE-CIS dataset and publishes transactions to a Kafka topic
one by one, simulating a real bank's transaction stream.

Transactions are LEFT-JOINED against train_identity.csv before publishing.
This matters: the model was trained on the merged frame, where ~33% of rows
carry identity attributes (id_01..id_38, DeviceType, DeviceInfo). Streaming
the transaction file alone made those 41 features permanently missing at
serving time, which shifted scores and flipped ~0.9% of fraud verdicts.

Run this AFTER starting Kafka with docker-compose up.
Usage: python producer.py --rate 100
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from kafka import KafkaProducer
from kafka.errors import KafkaError, KafkaConnectionError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [PRODUCER] %(message)s",
    datefmt="%H:%M:%S"
)

log = logging.getLogger(__name__)


# ─── CONFIG ──────────────────────────────────────────────────────────────────

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
TOPIC = os.getenv("KAFKA_TOPIC", "transactions")
DATASET_PATH = os.getenv("DATASET_PATH", "data/train_transaction.csv")
IDENTITY_PATH = os.getenv("IDENTITY_PATH", "data/train_identity.csv")

# Rows pulled off disk at a time. The transaction CSV is 652 MB; reading it
# whole cost ~1.5 GB of RAM before a single message was sent.
CHUNK_SIZE = 20_000


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def serialize(data: dict) -> bytes:
    """JSON-encode a dict to bytes for Kafka."""
    return json.dumps(data, default=str).encode("utf-8")


def on_send_error(exc):
    """
    Errback for failed sends. Previously delivery_report() was defined but never
    wired to producer.send(), so acks="all" was requested and never verified --
    a broker rejecting messages would have been completely silent.
    """
    log.error(f"Delivery failed: {exc}")


def build_transaction(row: dict, include_label: bool) -> dict:
    """
    Convert a row into a clean transaction event.

    NaN -> None so the JSON is valid. This handles numpy scalar types too; the
    previous isinstance(v, float) check missed np.float32/np.float64 values.
    """
    transaction = {}
    for k, v in row.items():
        if v is None or (isinstance(v, (float, np.floating)) and np.isnan(v)):
            transaction[k] = None
        elif isinstance(v, np.generic):
            transaction[k] = v.item()      # numpy scalar -> native Python
        else:
            transaction[k] = v

    if not include_label:
        # A real ingestion stream has no ground truth. Keeping isFraud in-band is
        # how label leakage sneaks in later.
        transaction.pop("isFraud", None)

    transaction["ingested_at"] = datetime.now(timezone.utc).isoformat()
    return transaction


# ─── PRODUCER ────────────────────────────────────────────────────────────────

def create_producer(retries: int = 5) -> KafkaProducer:
    """
    Create a KafkaProducer with retry logic.
    Kafka sometimes takes a few seconds to be ready after Docker starts.
    """
    for attempt in range(1, retries + 1):
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BROKER,
                value_serializer=serialize,
                # Reliability settings
                acks="all",              # wait for all replicas to acknowledge
                retries=3,
                # Performance settings
                batch_size=16384,        # batch up to 16KB before sending
                linger_ms=5,             # wait 5ms to fill batches
                compression_type="gzip", # compress messages
            )
            log.info(f"Connected to Kafka at {KAFKA_BROKER}")
            return producer
        except KafkaConnectionError:
            log.warning(f"Kafka not ready (attempt {attempt}/{retries}) — retrying in 3s...")
            time.sleep(3)

    raise RuntimeError(
        "Could not connect to Kafka. "
        "Make sure you ran: docker-compose up -d"
    )


def load_identity() -> pd.DataFrame:
    """
    Load the identity table once and keep it indexed by TransactionID.
    It is only ~25 MB, so holding it in memory to join against each chunk is
    cheap and mirrors the `how="left"` merge done at training time.
    """
    if not os.path.exists(IDENTITY_PATH):
        log.warning(
            f"{IDENTITY_PATH} not found — streaming WITHOUT identity features. "
            "Scores will drift from training; see PROJECT_ANALYSIS.md (C1)."
        )
        return None

    ident = pd.read_csv(IDENTITY_PATH)
    log.info(f"Loaded identity table: {len(ident):,} rows, {ident.shape[1] - 1} features")
    return ident


def stream_transactions(rate_per_sec: int = 100, limit: int = None,
                        include_label: bool = True):
    """
    Main streaming loop.

    Args:
        rate_per_sec  : how many transactions to publish per second
        limit         : stop after N transactions (None = stream entire dataset)
        include_label : keep isFraud in the payload (needed for offline eval)
    """
    log.info(f"Streaming from {DATASET_PATH} in chunks of {CHUNK_SIZE:,}...")
    identity = load_identity()

    producer = create_producer()
    sleep_interval = 1.0 / rate_per_sec

    sent = 0
    fraud_sent = 0
    with_identity = 0
    start_time = time.time()

    log.info(f"Streaming at {rate_per_sec} tx/sec → topic '{TOPIC}'")
    log.info("Press Ctrl+C to stop\n")

    try:
        for chunk in pd.read_csv(DATASET_PATH, chunksize=CHUNK_SIZE):
            if limit and sent >= limit:
                break

            # Same left join as training: most transactions have no identity row.
            if identity is not None:
                chunk = chunk.merge(identity, on="TransactionID", how="left")

            # to_dict("records") is far cheaper than iterrows(), which builds a
            # fully-typed Series per row.
            for row in chunk.to_dict("records"):
                if limit and sent >= limit:
                    break

                transaction = build_transaction(row, include_label)

                # Use TransactionID as the Kafka message key
                # This ensures all events for the same transaction go to
                # the same partition (preserving order per transaction)
                key = str(transaction["TransactionID"]).encode("utf-8")

                producer.send(
                    TOPIC,
                    key=key,
                    value=transaction,
                ).add_errback(on_send_error)

                sent += 1
                if row.get("isFraud") == 1:
                    fraud_sent += 1
                if transaction.get("DeviceType") is not None:
                    with_identity += 1

                # Progress log every 1000 messages
                if sent % 1000 == 0:
                    elapsed = time.time() - start_time
                    actual_rate = sent / elapsed
                    log.info(
                        f"Sent {sent:,} transactions "
                        f"({fraud_sent:,} fraud, {with_identity/sent:.1%} w/ identity) | "
                        f"Rate: {actual_rate:.0f} tx/sec"
                    )

                time.sleep(sleep_interval)

    except KeyboardInterrupt:
        log.info("\nStopped by user")

    except KafkaError as e:
        log.error(f"Kafka error, aborting stream: {e}")

    finally:
        # Flush ensures all buffered messages are sent before exit
        log.info("Flushing remaining messages...")
        producer.flush()
        producer.close()

        elapsed = max(time.time() - start_time, 1e-9)
        log.info(f"\n{'='*45}")
        log.info(f"Total sent    : {sent:,} transactions")
        log.info(f"Fraud sent    : {fraud_sent:,} ({fraud_sent/max(sent,1):.2%})")
        log.info(f"With identity : {with_identity:,} ({with_identity/max(sent,1):.1%})")
        log.info(f"Elapsed       : {elapsed:.1f}s")
        log.info(f"Avg rate      : {sent/elapsed:.0f} tx/sec")
        log.info(f"{'='*45}")


# ─── ENTRYPOINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fraud transaction stream producer")
    parser.add_argument(
        "--rate", type=int, default=100,
        help="Transactions per second (default: 100)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Stop after N transactions (default: stream full dataset)"
    )
    parser.add_argument(
        "--no-label", action="store_true",
        help="Strip isFraud from the payload (simulates a true production stream)"
    )
    args = parser.parse_args()

    stream_transactions(
        rate_per_sec=args.rate,
        limit=args.limit,
        include_label=not args.no_label,
    )

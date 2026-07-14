"""
Kafka Producer — Transaction Stream Simulator
=============================================
Reads the IEEE-CIS dataset and publishes transactions to a Kafka topic
one by one, simulating a real bank's transaction stream.

Run this AFTER starting Kafka with docker-compose up.
Usage: python producer.py --rate 100
"""

import json
import time
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from kafka import KafkaProducer
from kafka.errors import KafkaConnectionError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [PRODUCER] %(message)s",
    datefmt="%H:%M:%S"
)

log = logging.getLogger(__name__)


# ─── CONFIG ──────────────────────────────────────────────────────────────────

KAFKA_BROKER = "localhost:9092"
TOPIC = "transactions"
DATASET_PATH = "data/train_transaction.csv"


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def serialize(data: dict) -> bytes:
    """JSON-encode a dict to bytes for Kafka."""
    return json.dumps(data, default=str).encode("utf-8")


def delivery_report(record_metadata):
    """Called once per message to confirm delivery."""
    log.debug(
        f"Delivered to {record_metadata.topic} "
        f"[partition {record_metadata.partition}] "
        f"offset {record_metadata.offset}"
    )


def build_transaction(row: pd.Series) -> dict:
    """
    Convert a DataFrame row into a clean transaction event.
    We add a real timestamp so the consumer knows when it arrived.
    """
    transaction = row.to_dict()

    # Replace NaN with None so JSON serialization works
    transaction = {
        k: (None if isinstance(v, float) and np.isnan(v) else v)
        for k, v in transaction.items()
    }

    # Add ingestion timestamp
    transaction["ingested_at"] = datetime.utcnow().isoformat()

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


def stream_transactions(rate_per_sec: int = 100, limit: int = None):
    """
    Main streaming loop.

    Args:
        rate_per_sec : how many transactions to publish per second
        limit        : stop after N transactions (None = stream entire dataset)
    """
    log.info(f"Loading dataset from {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)
    log.info(f"Loaded {len(df):,} transactions")

    if limit:
        df = df.head(limit)

    producer = create_producer()
    sleep_interval = 1.0 / rate_per_sec

    sent = 0
    fraud_sent = 0
    start_time = time.time()

    log.info(f"Streaming at {rate_per_sec} tx/sec → topic '{TOPIC}'")
    log.info("Press Ctrl+C to stop\n")

    try:
        for _, row in df.iterrows():
            transaction = build_transaction(row)

            # Use TransactionID as the Kafka message key
            # This ensures all events for the same transaction go to
            # the same partition (preserving order per transaction)
            key = str(transaction["TransactionID"]).encode("utf-8")

            producer.send(
                TOPIC,
                key=key,
                value=transaction,
            )

            sent += 1
            if transaction.get("isFraud") == 1:
                fraud_sent += 1

            # Progress log every 1000 messages
            if sent % 1000 == 0:
                elapsed = time.time() - start_time
                actual_rate = sent / elapsed
                log.info(
                    f"Sent {sent:,} transactions "
                    f"({fraud_sent:,} fraud) | "
                    f"Rate: {actual_rate:.0f} tx/sec"
                )

            time.sleep(sleep_interval)

    except KeyboardInterrupt:
        log.info("\nStopped by user")

    finally:
        # Flush ensures all buffered messages are sent before exit
        log.info("Flushing remaining messages...")
        producer.flush()
        producer.close()

        elapsed = time.time() - start_time
        log.info(f"\n{'='*45}")
        log.info(f"Total sent  : {sent:,} transactions")
        log.info(f"Fraud sent  : {fraud_sent:,} ({fraud_sent/max(sent,1):.2%})")
        log.info(f"Elapsed     : {elapsed:.1f}s")
        log.info(f"Avg rate    : {sent/elapsed:.0f} tx/sec")
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
    args = parser.parse_args()

    stream_transactions(rate_per_sec=args.rate, limit=args.limit)

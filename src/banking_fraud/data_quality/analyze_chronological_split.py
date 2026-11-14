"""Evaluate a chronological train/test split for PaySim."""

import logging
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "paysim.csv"

CHUNK_SIZE = 250_000
TRAIN_RATIO = 0.80

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Analyze fraud distribution across chronological partitions."""

    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {INPUT_PATH}")

    summaries = []

    for chunk in pd.read_csv(
        INPUT_PATH,
        usecols=["step", "isFraud"],
        chunksize=CHUNK_SIZE,
    ):
        grouped = chunk.groupby("step").agg(
            transactions=("isFraud", "size"),
            fraud=("isFraud", "sum"),
        )
        summaries.append(grouped)

    step_summary = pd.concat(summaries).groupby(level=0).sum()
    step_summary = step_summary.sort_index()

    split_index = int(len(step_summary) * TRAIN_RATIO)
    cutoff_step = step_summary.index[split_index]

    train = step_summary.iloc[:split_index]
    test = step_summary.iloc[split_index:]

    for name, partition in [("TRAIN", train), ("TEST", test)]:
        transactions = partition["transactions"].sum()
        fraud = partition["fraud"].sum()
        fraud_rate = fraud / transactions * 100

        logger.info("%s results:", name)
        logger.info("Steps: %s to %s", partition.index.min(), partition.index.max())
        logger.info("Transactions: %s", f"{transactions:,}")
        logger.info("Fraud: %s", f"{fraud:,}")
        logger.info("Non-fraud: %s", f"{transactions - fraud:,}")
        logger.info("Fraud rate: %.4f%%", fraud_rate)

    logger.info("Total simulation steps: %s", len(step_summary))
    logger.info("Training steps: %s", len(train))
    logger.info("Testing steps: %s", len(test))
    logger.info("First test step: %s", cutoff_step)


if __name__ == "__main__":
    main()

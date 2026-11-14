"""Analyze fraud distribution across simulation periods."""

import logging
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "paysim.csv"

CHUNK_SIZE = 250_000
PERIOD_SIZE = 30

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Calculate fraud rates across simulation periods."""

    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {INPUT_PATH}")

    summaries = []

    for chunk in pd.read_csv(
        INPUT_PATH,
        usecols=["step", "isFraud"],
        chunksize=CHUNK_SIZE,
    ):
        chunk["period"] = (chunk["step"] - 1) // PERIOD_SIZE + 1

        grouped = chunk.groupby("period")["isFraud"].agg(
            transactions="count",
            fraud="sum",
        )

        summaries.append(grouped)

    result = pd.concat(summaries).groupby(level=0).sum()

    result["fraud_rate_pct"] = result["fraud"] / result["transactions"] * 100

    logger.info("Fraud distribution by period:\n%s", result.to_string())

    logger.info("Total transactions: %s", f"{result['transactions'].sum():,}")
    logger.info("Total fraud: %s", f"{result['fraud'].sum():,}")


if __name__ == "__main__":
    main()

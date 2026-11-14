"""Investigate transaction types in low-volume periods."""

import logging
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "paysim.csv"

CHUNK_SIZE = 250_000
PERIOD_SIZE = 30
LOW_VOLUME_PERIODS = [3, 25]

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Analyze transaction types in selected periods."""

    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {INPUT_PATH}")

    summaries = []

    for chunk in pd.read_csv(
        INPUT_PATH,
        usecols=["step", "type", "isFraud"],
        chunksize=CHUNK_SIZE,
    ):
        chunk["period"] = (chunk["step"] - 1) // PERIOD_SIZE + 1

        selected = chunk[chunk["period"].isin(LOW_VOLUME_PERIODS)]

        if selected.empty:
            continue

        grouped = selected.groupby(["period", "type"]).agg(
            transactions=("isFraud", "size"),
            fraud=("isFraud", "sum"),
        )

        summaries.append(grouped)

    if not summaries:
        logger.info("No transactions found in selected periods.")
        return

    result = pd.concat(summaries).groupby(level=[0, 1]).sum()

    logger.info(
        "Transaction distribution in low-volume periods:\n%s",
        result.to_string(),
    )


if __name__ == "__main__":
    main()

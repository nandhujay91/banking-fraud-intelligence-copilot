"""Analyze transaction volume and fraud across simulation steps."""

import logging
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "paysim.csv"

CHUNK_SIZE = 250_000

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Summarize transaction volume and fraud by simulation step."""

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

    result = pd.concat(summaries).groupby(level=0).sum()

    logger.info("Lowest-volume steps:\n%s", result.nsmallest(10, "transactions"))
    logger.info("Highest-volume steps:\n%s", result.nlargest(10, "transactions"))


if __name__ == "__main__":
    main()

"""Identify simulation periods with unusual transaction volumes."""

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
    """Summarize transaction volume by simulation period."""

    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {INPUT_PATH}")

    summaries = []

    for chunk in pd.read_csv(
        INPUT_PATH,
        usecols=["step", "isFraud"],
        chunksize=CHUNK_SIZE,
    ):
        chunk["period"] = (chunk["step"] - 1) // PERIOD_SIZE + 1

        grouped = chunk.groupby("period").agg(
            transactions=("isFraud", "size"),
            fraud=("isFraud", "sum"),
        )

        summaries.append(grouped)

    result = pd.concat(summaries).groupby(level=0).sum()

    median_volume = result["transactions"].median()

    low_volume = result[result["transactions"] < median_volume * 0.1]

    logger.info("Median period volume: %s", f"{median_volume:,.0f}")
    logger.info("Low-volume periods:\n%s", low_volume.to_string())

    logger.info(
        "Periods below 10%% of median volume: %s",
        len(low_volume),
    )


if __name__ == "__main__":
    main()

"""Analyze fraud distribution over PaySim simulation time."""

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
    """Calculate fraud distribution by simulation step."""

    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {INPUT_PATH}")

    summary = []

    for chunk in pd.read_csv(
        INPUT_PATH,
        usecols=["step", "isFraud"],
        chunksize=CHUNK_SIZE,
    ):
        grouped = chunk.groupby("step")["isFraud"].agg(["count", "sum"])
        summary.append(grouped)

    result = pd.concat(summary).groupby(level=0).sum()

    result["fraud_percentage"] = result["sum"] / result["count"] * 100

    logger.info("Simulation steps: %s", f"{len(result):,}")
    logger.info("First step: %s", result.index.min())
    logger.info("Last step: %s", result.index.max())
    logger.info("Steps containing fraud: %s", (result["sum"] > 0).sum())

    logger.info(
        "Top 10 steps by fraud count:\n%s",
        result.nlargest(10, "sum"),
    )


if __name__ == "__main__":
    main()

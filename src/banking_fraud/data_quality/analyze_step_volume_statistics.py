"""Calculate transaction volume statistics by simulation step."""

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
    """Calculate descriptive statistics for step volumes."""

    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {INPUT_PATH}")

    summaries = []

    for chunk in pd.read_csv(
        INPUT_PATH,
        usecols=["step"],
        chunksize=CHUNK_SIZE,
    ):
        summaries.append(chunk["step"].value_counts())

    step_counts = pd.concat(summaries).groupby(level=0).sum()

    logger.info(
        "Transaction volume statistics:\n%s",
        step_counts.describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.95]).to_string(),
    )


if __name__ == "__main__":
    main()

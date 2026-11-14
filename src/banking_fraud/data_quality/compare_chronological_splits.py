"""Compare chronological train/test split strategies."""

import logging
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "paysim.csv"

CHUNK_SIZE = 250_000
SPLIT_RATIOS = [0.70, 0.80, 0.90]

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Compare transaction and fraud distribution across splits."""

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

    steps = pd.concat(summaries).groupby(level=0).sum().sort_index()

    for ratio in SPLIT_RATIOS:
        split_index = int(len(steps) * ratio)

        train = steps.iloc[:split_index]
        test = steps.iloc[split_index:]

        train_rows = train["transactions"].sum()
        test_rows = test["transactions"].sum()

        train_fraud = train["fraud"].sum()
        test_fraud = test["fraud"].sum()

        logger.info(
            "%.0f/%.0f split | cutoff step: %s",
            ratio * 100,
            (1 - ratio) * 100,
            test.index.min(),
        )
        logger.info(
            "Train: %s rows | %s fraud | %.4f%% fraud rate",
            f"{train_rows:,}",
            f"{train_fraud:,}",
            train_fraud / train_rows * 100,
        )
        logger.info(
            "Test: %s rows | %s fraud | %.4f%% fraud rate",
            f"{test_rows:,}",
            f"{test_fraud:,}",
            test_fraud / test_rows * 100,
        )


if __name__ == "__main__":
    main()

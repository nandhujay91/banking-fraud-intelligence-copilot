"""Analyze fraud and non-fraud distribution by transaction type."""

import logging
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "fraud_features.parquet"

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Calculate class distribution by transaction type."""

    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_parquet(DATA_PATH)

    summary = (
        data.groupby(["type", "isFraud"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={0: "non_fraud", 1: "fraud"})
    )

    summary["total"] = summary["non_fraud"] + summary["fraud"]
    summary["fraud_percentage"] = summary["fraud"] / summary["total"] * 100

    logger.info("Class distribution by transaction type:\n%s", summary)

    logger.info("Total fraud: %s", f"{data['isFraud'].sum():,}")


if __name__ == "__main__":
    main()

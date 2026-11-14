"""Analyze duplicate feature records."""

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
    """Analyze exact duplicate feature records."""

    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_parquet(DATA_PATH)

    duplicate_count = data.duplicated().sum()
    duplicate_percentage = duplicate_count / len(data) * 100

    logger.info("Total transactions: %s", f"{len(data):,}")
    logger.info("Duplicate feature rows: %s", f"{duplicate_count:,}")
    logger.info("Duplicate percentage: %.4f%%", duplicate_percentage)

    logger.info("Duplicate analysis completed. No records were removed.")


if __name__ == "__main__":
    main()

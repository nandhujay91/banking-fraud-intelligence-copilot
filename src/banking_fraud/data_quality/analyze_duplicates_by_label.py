"""Analyze duplicate feature records by fraud label."""

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
    """Compare duplicate feature records by fraud label."""

    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_parquet(DATA_PATH)

    duplicate_rows = data[data.duplicated(keep=False)]

    summary = duplicate_rows.groupby("isFraud").size().rename("duplicate_rows")

    logger.info("Duplicate rows by fraud label:")

    for label, count in summary.items():
        logger.info(
            "Fraud label %s: %s",
            label,
            f"{count:,}",
        )


if __name__ == "__main__":
    main()

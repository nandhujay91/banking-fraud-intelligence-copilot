"""Identify feature combinations with conflicting fraud labels."""

import logging
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "fraud_features.parquet"

FEATURE_COLUMNS = [
    "type",
    "amount",
    "oldbalanceOrg",
    "log_amount",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Display feature combinations with conflicting labels."""

    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_parquet(DATA_PATH)

    label_counts = (
        data.groupby(FEATURE_COLUMNS, dropna=False)["isFraud"]
        .value_counts()
        .unstack(fill_value=0)
    )

    conflicts = label_counts[(label_counts[0] > 0) & (label_counts[1] > 0)]

    logger.info("Conflicting feature groups: %s", len(conflicts))

    if not conflicts.empty:
        logger.info("Conflicting feature details:\n%s", conflicts)


if __name__ == "__main__":
    main()

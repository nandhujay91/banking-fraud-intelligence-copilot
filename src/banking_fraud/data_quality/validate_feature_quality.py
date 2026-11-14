"""Validate the quality of the engineered fraud dataset."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "fraud_features.parquet"

EXPECTED_COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "log_amount",
    "isFraud",
]

NUMERIC_COLUMNS = [
    "step",
    "amount",
    "oldbalanceOrg",
    "log_amount",
]

MONETARY_COLUMNS = [
    "amount",
    "oldbalanceOrg",
]

VALID_TYPES = {
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
}

EXPECTED_ROWS = 6_362_620

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)


def validate_features(data: pd.DataFrame) -> None:
    """Check schema, values, and feature consistency."""

    if list(data.columns) != EXPECTED_COLUMNS:
        raise ValueError("Unexpected feature columns.")

    if len(data) != EXPECTED_ROWS:
        raise ValueError(f"Unexpected row count: {len(data):,}")

    if data.isna().any().any():
        raise ValueError("Missing values found.")

    if not np.isfinite(data[NUMERIC_COLUMNS]).all().all():
        raise ValueError("Infinite numeric values found.")

    if (data[MONETARY_COLUMNS] < 0).any().any():
        raise ValueError("Negative monetary values found.")

    if (data["step"] < 1).any():
        raise ValueError("Invalid step values found.")

    if not data["step"].mod(1).eq(0).all():
        raise ValueError("Non-integer step values found.")

    if not data["type"].isin(VALID_TYPES).all():
        raise ValueError("Invalid transaction type found.")

    if not data["isFraud"].isin([0, 1]).all():
        raise ValueError("Invalid fraud labels found.")

    expected_log = np.log1p(data["amount"])

    if not np.allclose(data["log_amount"], expected_log):
        raise ValueError("Log amount consistency check failed.")


def main() -> None:
    """Run feature dataset quality checks."""

    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    logger.info("Loading engineered feature dataset.")

    data = pd.read_parquet(DATA_PATH)

    validate_features(data)

    logger.info("Total rows: %s", f"{len(data):,}")
    logger.info("Total columns: %s", len(data.columns))
    logger.info("First step: %s", data["step"].min())
    logger.info("Last step: %s", data["step"].max())
    logger.info("Fraud transactions: %s", f"{data['isFraud'].sum():,}")
    logger.info(
        "Non-fraud transactions: %s",
        f"{(data['isFraud'] == 0).sum():,}",
    )
    logger.info("Feature quality validation passed.")


if __name__ == "__main__":
    main()

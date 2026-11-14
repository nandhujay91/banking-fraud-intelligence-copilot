"""Validate chronological training and testing datasets."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"

TRAIN_PATH = SPLIT_DIR / "train.parquet"
TEST_PATH = SPLIT_DIR / "test.parquet"

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

EXPECTED_TRAIN_ROWS = 6_082_007
EXPECTED_TEST_ROWS = 280_613
EXPECTED_TRAIN_FRAUD = 5_781
EXPECTED_TEST_FRAUD = 2_432

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)


def validate_dataset(data: pd.DataFrame, name: str) -> None:
    """Validate the schema and values of one split."""

    if list(data.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"{name}: unexpected columns.")

    if data.isna().any().any():
        raise ValueError(f"{name}: missing values found.")

    if not np.isfinite(data[NUMERIC_COLUMNS]).all().all():
        raise ValueError(f"{name}: infinite numeric values found.")

    if not data["isFraud"].isin([0, 1]).all():
        raise ValueError(f"{name}: invalid fraud labels found.")

    if not np.allclose(data["log_amount"], np.log1p(data["amount"])):
        raise ValueError(f"{name}: log_amount inconsistency.")


def main() -> None:
    """Validate both chronological splits."""

    if not TRAIN_PATH.is_file() or not TEST_PATH.is_file():
        raise FileNotFoundError("Training or testing dataset is missing.")

    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)

    validate_dataset(train, "Training")
    validate_dataset(test, "Testing")

    if len(train) != EXPECTED_TRAIN_ROWS:
        raise ValueError("Unexpected training row count.")

    if len(test) != EXPECTED_TEST_ROWS:
        raise ValueError("Unexpected testing row count.")

    if train["step"].max() >= test["step"].min():
        raise ValueError("Training and testing steps overlap.")

    if train["step"].min() != 1 or train["step"].max() != 520:
        raise ValueError("Unexpected training step range.")

    if test["step"].min() != 521 or test["step"].max() != 743:
        raise ValueError("Unexpected testing step range.")

    train_fraud = int(train["isFraud"].sum())
    test_fraud = int(test["isFraud"].sum())

    if train_fraud != EXPECTED_TRAIN_FRAUD:
        raise ValueError("Unexpected training fraud count.")

    if test_fraud != EXPECTED_TEST_FRAUD:
        raise ValueError("Unexpected testing fraud count.")

    logger.info("Training rows: %s", f"{len(train):,}")
    logger.info("Training fraud: %s", f"{train_fraud:,}")
    logger.info("Testing rows: %s", f"{len(test):,}")
    logger.info("Testing fraud: %s", f"{test_fraud:,}")
    logger.info("Training steps: %s–%s", train["step"].min(), train["step"].max())
    logger.info("Testing steps: %s–%s", test["step"].min(), test["step"].max())
    logger.info("Total rows: %s", f"{len(train) + len(test):,}")
    logger.info("Chronological split validation passed.")


if __name__ == "__main__":
    main()

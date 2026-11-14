"""Initial data profiling for the PaySim transaction dataset."""

import logging
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "paysim.csv"

CHUNK_SIZE = 250_000

EXPECTED_COLUMNS = {
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)


def validate_dataset_schema(columns: list[str]) -> None:
    """Check whether the dataset contains the expected columns."""

    actual_columns = set(columns)

    missing_columns = EXPECTED_COLUMNS - actual_columns
    unexpected_columns = actual_columns - EXPECTED_COLUMNS

    if missing_columns or unexpected_columns:
        raise ValueError(
            "Dataset schema validation failed. "
            f"Missing columns: {sorted(missing_columns)}. "
            f"Unexpected columns: {sorted(unexpected_columns)}."
        )


def profile_dataset() -> None:
    """Calculate basic dataset statistics using chunked processing."""

    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"PaySim dataset not found: {DATASET_PATH}")

    total_rows = 0
    missing_values = None
    column_dtypes = None

    fraud_counts = Counter()
    flagged_fraud_counts = Counter()
    transaction_type_counts = Counter()

    total_amount = 0.0
    minimum_amount = float("inf")
    maximum_amount = float("-inf")

    minimum_step = float("inf")
    maximum_step = float("-inf")

    logger.info("Starting PaySim dataset profiling.")
    logger.info("Dataset location: %s", DATASET_PATH)
    logger.info("File size: %.2f MB", DATASET_PATH.stat().st_size / 1_000_000)

    for chunk in pd.read_csv(
        DATASET_PATH,
        chunksize=CHUNK_SIZE,
    ):
        if column_dtypes is None:
            validate_dataset_schema(chunk.columns.tolist())

            column_dtypes = chunk.dtypes.astype(str)
            missing_values = pd.Series(
                0,
                index=chunk.columns,
                dtype="int64",
            )

        total_rows += len(chunk)

        missing_values += chunk.isna().sum()

        fraud_counts.update(chunk["isFraud"].value_counts().to_dict())

        flagged_fraud_counts.update(chunk["isFlaggedFraud"].value_counts().to_dict())

        transaction_type_counts.update(chunk["type"].value_counts().to_dict())

        total_amount += chunk["amount"].sum()

        minimum_amount = min(
            minimum_amount,
            chunk["amount"].min(),
        )
        maximum_amount = max(
            maximum_amount,
            chunk["amount"].max(),
        )

        minimum_step = min(
            minimum_step,
            chunk["step"].min(),
        )
        maximum_step = max(
            maximum_step,
            chunk["step"].max(),
        )

    if total_rows == 0:
        raise ValueError("The PaySim dataset contains no records.")

    logger.info("Dataset profiling completed.")

    print("\n" + "=" * 55)
    print("PAYSIM DATASET PROFILE")
    print("=" * 55)

    print("\nDATASET OVERVIEW")
    print(f"Total records       : {total_rows:,}")
    print(f"Total columns       : {len(column_dtypes)}")

    print("\nCOLUMN DATA TYPES")
    print(column_dtypes.to_string())

    print("\nMISSING VALUES")
    print(missing_values.to_string())
    print(f"Total missing values: {missing_values.sum():,}")

    print("\nFRAUD DISTRIBUTION")
    for label, count in sorted(fraud_counts.items()):
        percentage = count / total_rows * 100
        print(f"Fraud {label}: {count:,} ({percentage:.4f}%)")

    print("\nFLAGGED FRAUD DISTRIBUTION")
    for label, count in sorted(flagged_fraud_counts.items()):
        print(f"Flagged {label}: {count:,}")

    print("\nTRANSACTION TYPE DISTRIBUTION")
    for transaction_type, count in sorted(transaction_type_counts.items()):
        percentage = count / total_rows * 100
        print(f"{transaction_type}: {count:,} ({percentage:.2f}%)")

    print("\nTRANSACTION AMOUNT")
    print(f"Minimum amount: {minimum_amount:,.2f}")
    print(f"Maximum amount: {maximum_amount:,.2f}")
    print(f"Average amount: {total_amount / total_rows:,.2f}")

    print("\nSIMULATION TIME")
    print(f"First step: {minimum_step}")
    print(f"Last step : {maximum_step}")

    print("\n" + "=" * 55)


if __name__ == "__main__":
    profile_dataset()

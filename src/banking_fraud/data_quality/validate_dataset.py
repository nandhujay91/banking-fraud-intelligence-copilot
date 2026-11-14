"""Data quality checks for the PaySim transaction dataset."""

import logging
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

VALID_TRANSACTION_TYPES = {
    "PAYMENT",
    "TRANSFER",
    "CASH_OUT",
    "DEBIT",
    "CASH_IN",
}

VALID_BINARY_LABELS = {0, 1}

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)


def validate_schema(columns: list[str]) -> None:
    """Validate the dataset column names."""

    actual_columns = set(columns)

    missing_columns = EXPECTED_COLUMNS - actual_columns
    unexpected_columns = actual_columns - EXPECTED_COLUMNS

    if missing_columns or unexpected_columns:
        raise ValueError(
            "Dataset schema validation failed. "
            f"Missing columns: {sorted(missing_columns)}. "
            f"Unexpected columns: {sorted(unexpected_columns)}."
        )


def validate_dataset() -> None:
    """Run initial data quality checks using chunked processing."""

    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"PaySim dataset not found: {DATASET_PATH}")

    total_rows = 0
    duplicate_rows = 0
    negative_amount_rows = 0
    invalid_transaction_type_rows = 0
    invalid_fraud_label_rows = 0
    invalid_flag_label_rows = 0
    invalid_step_rows = 0

    missing_values = None
    previous_chunk = None

    transaction_type_counts = {}
    fraud_counts = {}
    flagged_fraud_counts = {}

    logger.info("Starting PaySim data quality checks.")
    logger.info("Dataset: %s", DATASET_PATH)

    for chunk in pd.read_csv(
        DATASET_PATH,
        chunksize=CHUNK_SIZE,
    ):
        if total_rows == 0:
            validate_schema(chunk.columns.tolist())

            missing_values = pd.Series(
                0,
                index=chunk.columns,
                dtype="int64",
            )

        total_rows += len(chunk)

        missing_values += chunk.isna().sum()

        # Check duplicate rows within the current chunk.
        duplicate_rows += int(chunk.duplicated().sum())

        # Check duplicates that cross chunk boundaries.
        if previous_chunk is not None:
            combined = pd.concat(
                [previous_chunk, chunk],
                ignore_index=True,
            )

            duplicate_rows += int(
                combined.duplicated().sum()
                - chunk.duplicated().sum()
                - previous_chunk.duplicated().sum()
            )

        previous_chunk = chunk.tail(1).copy()

        # Validate transaction amounts.
        negative_amount_rows += int((chunk["amount"] < 0).sum())

        # Validate transaction types.
        invalid_transaction_type_rows += int(
            (~chunk["type"].isin(VALID_TRANSACTION_TYPES)).sum()
        )

        # Validate fraud labels.
        invalid_fraud_label_rows += int(
            (~chunk["isFraud"].isin(VALID_BINARY_LABELS)).sum()
        )

        invalid_flag_label_rows += int(
            (~chunk["isFlaggedFraud"].isin(VALID_BINARY_LABELS)).sum()
        )

        # Validate simulation steps.
        invalid_step_rows += int((chunk["step"] < 1).sum())

        transaction_type_counts.update(
            chunk["type"].value_counts().to_dict()
        ) if hasattr(transaction_type_counts, "update") else None

        for label, count in chunk["isFraud"].value_counts().items():
            fraud_counts[label] = fraud_counts.get(label, 0) + count

        for label, count in chunk["isFlaggedFraud"].value_counts().items():
            flagged_fraud_counts[label] = flagged_fraud_counts.get(label, 0) + count

    if total_rows == 0:
        raise ValueError("The PaySim dataset contains no records.")

    print("\n" + "=" * 60)
    print("PAYSIM DATA QUALITY REPORT")
    print("=" * 60)

    print(f"\nTotal records: {total_rows:,}")

    print("\nMISSING VALUES")
    print(missing_values.to_string())
    print(f"Total missing values: {missing_values.sum():,}")

    print("\nDATA VALIDATION")
    print(f"Duplicate rows: {duplicate_rows:,}")
    print(f"Negative amounts: {negative_amount_rows:,}")
    print(f"Invalid transaction types: {invalid_transaction_type_rows:,}")
    print(f"Invalid fraud labels: {invalid_fraud_label_rows:,}")
    print(f"Invalid flagged-fraud labels: {invalid_flag_label_rows:,}")
    print(f"Invalid simulation steps: {invalid_step_rows:,}")

    print("\nFRAUD LABEL DISTRIBUTION")
    print(fraud_counts)

    print("\nFLAGGED FRAUD DISTRIBUTION")
    print(flagged_fraud_counts)

    print("\n" + "=" * 60)
    print("DATA QUALITY CHECKS COMPLETED")
    print("=" * 60)

    logger.info("Data quality checks completed.")


if __name__ == "__main__":
    validate_dataset()

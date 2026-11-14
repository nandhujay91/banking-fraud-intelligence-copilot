"""Build initial features for the PaySim fraud dataset."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "paysim.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "fraud_features.parquet"

CHUNK_SIZE = 250_000

REQUIRED_COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "isFraud",
]

VALID_TYPES = {
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)


def validate_data(data: pd.DataFrame) -> None:
    """Validate transaction data before feature engineering."""

    if data[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("Missing values found in input data.")

    if (data[["amount", "oldbalanceOrg"]] < 0).any().any():
        raise ValueError("Negative monetary values found.")

    if not np.isfinite(data[REQUIRED_COLUMNS[0]]).all():
        raise ValueError("Invalid simulation step values found.")

    if (data["step"] < 1).any() or not data["step"].mod(1).eq(0).all():
        raise ValueError("Invalid simulation step values found.")

    if not data["isFraud"].isin([0, 1]).all():
        raise ValueError("Invalid fraud labels found.")

    if not data["type"].isin(VALID_TYPES).all():
        raise ValueError("Unknown transaction type found.")


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create model features and preserve simulation metadata."""

    features = data[REQUIRED_COLUMNS].copy()

    features["log_amount"] = np.log1p(features["amount"])

    numeric_columns = [
        "amount",
        "oldbalanceOrg",
        "log_amount",
    ]

    if not np.isfinite(features[numeric_columns]).all().all():
        raise ValueError("Invalid or infinite feature values found.")

    return features[
        [
            "step",
            "type",
            "amount",
            "oldbalanceOrg",
            "log_amount",
            "isFraud",
        ]
    ]


def main() -> None:
    """Build features and save them as a Parquet file."""

    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {INPUT_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    writer = None

    logger.info("Starting feature engineering.")

    try:
        for chunk in pd.read_csv(
            INPUT_PATH,
            usecols=REQUIRED_COLUMNS,
            chunksize=CHUNK_SIZE,
        ):
            validate_data(chunk)
            features = build_features(chunk)

            table = pa.Table.from_pandas(
                features,
                preserve_index=False,
            )

            if writer is None:
                writer = pq.ParquetWriter(
                    OUTPUT_PATH,
                    table.schema,
                    compression="snappy",
                )

            writer.write_table(table)
            total_rows += len(features)

            if total_rows % 1_000_000 < len(features):
                logger.info("Processed %s transactions.", f"{total_rows:,}")

    except Exception:
        if writer is not None:
            writer.close()

        if OUTPUT_PATH.exists():
            OUTPUT_PATH.unlink()

        raise

    else:
        if writer is not None:
            writer.close()

    if total_rows == 0:
        raise ValueError("No transactions were processed.")

    logger.info("Feature engineering completed.")
    logger.info("Total transactions: %s", f"{total_rows:,}")
    logger.info("Output file: %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()

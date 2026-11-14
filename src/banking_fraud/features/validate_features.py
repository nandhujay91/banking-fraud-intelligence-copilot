"""Validate the engineered PaySim feature dataset."""

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

EXPECTED_ROWS = 6_362_620


def main() -> None:
    """Validate the Parquet dataset."""

    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_parquet(DATA_PATH)

    if list(data.columns) != EXPECTED_COLUMNS:
        raise ValueError("Unexpected feature columns.")

    if len(data) != EXPECTED_ROWS:
        raise ValueError("Unexpected row count.")

    if data.isna().any().any():
        raise ValueError("Missing values found.")

    if not np.isfinite(data[NUMERIC_COLUMNS]).all().all():
        raise ValueError("Invalid numeric values found.")

    if (data[["amount", "oldbalanceOrg"]] < 0).any().any():
        raise ValueError("Negative monetary values found.")

    if (data["step"] < 1).any() or not data["step"].mod(1).eq(0).all():
        raise ValueError("Invalid simulation steps found.")

    if not data["isFraud"].isin([0, 1]).all():
        raise ValueError("Invalid fraud labels found.")

    if not np.allclose(data["log_amount"], np.log1p(data["amount"])):
        raise ValueError("Log amount consistency check failed.")

    print(f"Total rows: {len(data):,}")
    print(f"Columns: {list(data.columns)}")
    print(f"First simulation step: {data['step'].min()}")
    print(f"Last simulation step: {data['step'].max()}")
    print(f"Fraud transactions: {data['isFraud'].sum():,}")
    print("Feature validation passed!")


if __name__ == "__main__":
    main()

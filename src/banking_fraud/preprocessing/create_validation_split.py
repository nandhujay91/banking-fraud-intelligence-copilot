
"""Create chronological training and validation datasets."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

SOURCE_PATH = PROJECT_ROOT / "data" / "splits" / "train.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "validation"

CUTOFF_STEP = 416


def main() -> None:
    """Split training data into chronological training and validation sets."""
    df = pd.read_parquet(SOURCE_PATH)

    train_df = df[df["step"] <= CUTOFF_STEP].copy()
    validation_df = df[df["step"] > CUTOFF_STEP].copy()

    if train_df.empty or validation_df.empty:
        raise ValueError("Training or validation dataset is empty.")

    if train_df["step"].max() >= validation_df["step"].min():
        raise ValueError("Chronological split overlap detected.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_path = OUTPUT_DIR / "train.parquet"
    validation_path = OUTPUT_DIR / "validation.parquet"

    train_df.to_parquet(train_path, index=False)
    validation_df.to_parquet(validation_path, index=False)

    print("Validation split created successfully.")
    print(f"Training rows: {len(train_df):,}")
    print(f"Validation rows: {len(validation_df):,}")
    print(f"Training fraud: {train_df['isFraud'].sum():,}")
    print(f"Validation fraud: {validation_df['isFraud'].sum():,}")
    print(f"Training file: {train_path}")
    print(f"Validation file: {validation_path}")


if __name__ == "__main__":
    main()

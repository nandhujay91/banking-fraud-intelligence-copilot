"""Analyse transaction amount distribution by type."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "paysim.csv"

amounts = []

for chunk in pd.read_csv(
    DATASET_PATH,
    usecols=["type", "amount"],
    chunksize=250_000,
):
    amounts.append(chunk.copy())

data = pd.concat(amounts, ignore_index=True)

report = data.groupby("type")["amount"].agg(
    transactions="count",
    mean="mean",
    median="median",
    minimum="min",
    maximum="max",
)

print("\nTRANSACTION AMOUNT DISTRIBUTION BY TYPE")
print("=" * 75)
print(report.round(2).to_string())

"""Analyse sender balance differences by transaction type."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "paysim.csv"

results = []

for chunk in pd.read_csv(
    DATASET_PATH,
    usecols=["type", "amount", "oldbalanceOrg"],
    chunksize=250_000,
):
    chunk["balance_difference"] = chunk["oldbalanceOrg"] - chunk["amount"]

    summary = chunk.groupby("type").agg(
        transactions=("balance_difference", "size"),
        total_difference=("balance_difference", "sum"),
        min_difference=("balance_difference", "min"),
        max_difference=("balance_difference", "max"),
        negative_count=(
            "balance_difference",
            lambda x: (x < 0).sum(),
        ),
        zero_count=(
            "balance_difference",
            lambda x: (x == 0).sum(),
        ),
    )

    results.append(summary)

report = (
    pd.concat(results)
    .groupby(level=0)
    .agg(
        {
            "transactions": "sum",
            "total_difference": "sum",
            "min_difference": "min",
            "max_difference": "max",
            "negative_count": "sum",
            "zero_count": "sum",
        }
    )
)

report["mean_difference"] = report["total_difference"] / report["transactions"]

report["negative_pct"] = report["negative_count"] / report["transactions"] * 100

report["zero_pct"] = report["zero_count"] / report["transactions"] * 100

print("\nSENDER BALANCE DIFFERENCE ANALYSIS")
print("=" * 75)

print(
    report[
        [
            "transactions",
            "mean_difference",
            "min_difference",
            "max_difference",
            "negative_pct",
            "zero_pct",
        ]
    ]
    .round(2)
    .to_string()
)

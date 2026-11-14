"""Analyse transaction amounts against sender balances."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "paysim.csv"

CHUNK_SIZE = 250_000

results = []

for chunk in pd.read_csv(
    DATASET_PATH,
    usecols=["type", "amount", "oldbalanceOrg"],
    chunksize=CHUNK_SIZE,
):
    chunk["amount_above_balance"] = chunk["amount"] > chunk["oldbalanceOrg"]

    summary = (
        chunk.groupby("type")
        .agg(
            transactions=("amount", "size"),
            zero_balance=("oldbalanceOrg", lambda x: (x == 0).sum()),
            amount_above_balance=("amount_above_balance", "sum"),
        )
        .reset_index()
    )

    results.append(summary)

report = pd.concat(results).groupby("type").sum().reset_index()

report["above_balance_pct"] = (
    report["amount_above_balance"] / report["transactions"] * 100
)

print("\nSENDER BALANCE ANALYSIS BY TRANSACTION TYPE")
print("=" * 65)
print(report.to_string(index=False, float_format="%.2f"))

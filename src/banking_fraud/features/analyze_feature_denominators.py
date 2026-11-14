"""Check zero balances before creating ratio features."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "paysim.csv"

CHUNK_SIZE = 250_000

columns = ["amount", "oldbalanceOrg", "oldbalanceDest"]

totals = {
    "transactions": 0,
    "zero_amount": 0,
    "zero_sender_balance": 0,
    "zero_recipient_balance": 0,
    "amount_above_sender_balance": 0,
}

for chunk in pd.read_csv(
    DATASET_PATH,
    usecols=columns,
    chunksize=CHUNK_SIZE,
):
    totals["transactions"] += len(chunk)
    totals["zero_amount"] += int((chunk["amount"] == 0).sum())
    totals["zero_sender_balance"] += int((chunk["oldbalanceOrg"] == 0).sum())
    totals["zero_recipient_balance"] += int((chunk["oldbalanceDest"] == 0).sum())
    totals["amount_above_sender_balance"] += int(
        (chunk["amount"] > chunk["oldbalanceOrg"]).sum()
    )

print("\nFEATURE DENOMINATOR ANALYSIS")
print("=" * 55)

for name, count in totals.items():
    if name == "transactions":
        print(f"{name}: {count:,}")
    else:
        percentage = count / totals["transactions"] * 100
        print(f"{name}: {count:,} ({percentage:.2f}%)")

"""Compare original and log-transformed transaction amounts."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "paysim.csv"

amounts = []

for chunk in pd.read_csv(
    DATASET_PATH,
    usecols=["amount"],
    chunksize=250_000,
):
    amounts.extend(chunk["amount"].tolist())

data = pd.Series(amounts)

log_amount = np.log1p(data)

print("\nAMOUNT TRANSFORMATION ANALYSIS")
print("=" * 55)

print(f"Original amount skewness: {skew(data):.4f}")
print(f"Log amount skewness:      {skew(log_amount):.4f}")

print("\nOriginal amount:")
print(data.describe().round(2))

print("\nLog-transformed amount:")
print(log_amount.describe().round(4))

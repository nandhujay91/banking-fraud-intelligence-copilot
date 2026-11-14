
import json
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

TEST_PATH = PROJECT_ROOT / "data" / "splits" / "test.parquet"
MODEL_PATH = (
    PROJECT_ROOT / "models" / "xgboost" / "xgboost_fraud_model.joblib"
)
THRESHOLD_PATH = (
    PROJECT_ROOT / "models" / "xgboost" / "decision_threshold.json"
)

FEATURES = ["type", "amount", "oldbalanceOrg", "log_amount"]


def describe_group(name: str, data: pd.DataFrame) -> None:
    print(f"\n{name}")
    print("-" * 55)

    if data.empty:
        print("No transactions found.")
        return

    for column in ["amount", "oldbalanceOrg", "log_amount", "probability"]:
        values = data[column]

        print(
            f"{column:18} "
            f"mean={values.mean():,.4f} | "
            f"median={values.median():,.4f} | "
            f"min={values.min():,.4f} | "
            f"max={values.max():,.4f}"
        )


def main() -> None:
    test_data = pd.read_parquet(TEST_PATH)

    model = joblib.load(MODEL_PATH)

    with THRESHOLD_PATH.open(encoding="utf-8") as file:
        threshold = json.load(file)["threshold"]

    probabilities = model.predict_proba(test_data[FEATURES])[:, 1]

    results = test_data.copy()
    results["probability"] = probabilities
    results["predicted"] = (probabilities >= threshold).astype(int)

    cash_out_fraud = results[
        (results["type"] == "CASH_OUT")
        & (results["isFraud"] == 1)
    ].copy()

    false_negatives = cash_out_fraud[
        cash_out_fraud["predicted"] == 0
    ]

    true_positives = cash_out_fraud[
        cash_out_fraud["predicted"] == 1
    ]

    print(f"Decision threshold: {threshold:.6f}")
    print(f"Total CASH_OUT fraud: {len(cash_out_fraud):,}")
    print(f"Correctly detected: {len(true_positives):,}")
    print(f"Missed fraud: {len(false_negatives):,}")

    describe_group("MISSED CASH_OUT FRAUD", false_negatives)
    describe_group("CORRECTLY DETECTED CASH_OUT FRAUD", true_positives)

    print("\nMISSED FRAUD PROBABILITY DISTRIBUTION")
    print("-" * 55)

    probability_bins = pd.cut(
        false_negatives["probability"],
        bins=[0, 0.25, 0.50, 0.75, threshold, 1.0],
        include_lowest=True,
    )

    print(probability_bins.value_counts().sort_index())

    print("\nMISSED FRAUD BY AMOUNT QUARTILE")
    print("-" * 55)

    false_negatives["amount_quartile"] = pd.qcut(
        false_negatives["amount"],
        q=4,
        duplicates="drop",
    )

    print(
        false_negatives.groupby(
            "amount_quartile",
            observed=True,
        ).agg(
            transaction_count=("amount", "size"),
            average_amount=("amount", "mean"),
            average_probability=("probability", "mean"),
        )
    )

    output_dir = PROJECT_ROOT / "docs" / "reports" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "cash_out_false_negatives.csv"

    false_negatives[
        ["step", "type", "amount", "oldbalanceOrg",
         "log_amount", "probability", "predicted", "isFraud"]
    ].to_csv(output_path, index=False)

    print(f"\nMissed fraud details saved to: {output_path}")


if __name__ == "__main__":
    main()

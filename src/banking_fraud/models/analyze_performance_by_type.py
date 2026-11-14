
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

TEST_PATH = PROJECT_ROOT / "data" / "splits" / "test.parquet"
MODEL_PATH = (
    PROJECT_ROOT / "models" / "xgboost" / "xgboost_fraud_model.joblib"
)
THRESHOLD_PATH = (
    PROJECT_ROOT / "models" / "xgboost" / "decision_threshold.json"
)

FEATURES = ["type", "amount", "oldbalanceOrg", "log_amount"]


def main() -> None:
    test_data = pd.read_parquet(TEST_PATH)

    model = joblib.load(MODEL_PATH)

    with THRESHOLD_PATH.open(encoding="utf-8") as file:
        threshold = json.load(file)["threshold"]

    X_test = test_data[FEATURES]
    y_test = test_data["isFraud"]

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    results = pd.DataFrame(
        {
            "type": test_data["type"].values,
            "actual": y_test.values,
            "predicted": predictions,
            "probability": probabilities,
        }
    )

    print(f"Decision threshold: {threshold:.6f}")
    print(f"Total test transactions: {len(results):,}")

    print("\nPERFORMANCE BY TRANSACTION TYPE")

    for transaction_type, group in results.groupby("type"):
        actual = group["actual"]
        predicted = group["predicted"]

        tn, fp, fn, tp = confusion_matrix(
            actual, predicted, labels=[0, 1]
        ).ravel()

        fraud_count = int(actual.sum())
        predicted_fraud = int(predicted.sum())

        print(f"\n{transaction_type}")
        print("-" * 45)
        print(f"Transactions:       {len(group):,}")
        print(f"Actual fraud:       {fraud_count:,}")
        print(f"Predicted fraud:    {predicted_fraud:,}")
        print(f"True positives:     {tp:,}")
        print(f"False positives:    {fp:,}")
        print(f"False negatives:    {fn:,}")
        print(f"True negatives:     {tn:,}")

        if fraud_count > 0:
            print(
                f"Precision:          "
                f"{precision_score(actual, predicted, zero_division=0):.4f}"
            )
            print(
                f"Recall:             "
                f"{recall_score(actual, predicted, zero_division=0):.4f}"
            )
            print(
                f"F1-score:           "
                f"{f1_score(actual, predicted, zero_division=0):.4f}"
            )
            print(
                f"PR-AUC:             "
                f"{average_precision_score(actual, group['probability']):.4f}"
            )
        else:
            print("Precision:          N/A (no actual fraud)")
            print("Recall:             N/A (no actual fraud)")
            print("F1-score:           N/A (no actual fraud)")
            print("PR-AUC:             N/A (no actual fraud)")


if __name__ == "__main__":
    main()

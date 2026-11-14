
"""Analyze XGBoost decision thresholds using validation data only."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

VALIDATION_PATH = (
    PROJECT_ROOT / "data" / "validation" / "validation.parquet"
)

MODEL_PATH = (
    PROJECT_ROOT / "models" / "xgboost" / "xgboost_fraud_model.joblib"
)

THRESHOLD_PATH = (
    PROJECT_ROOT / "models" / "xgboost" / "decision_threshold.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "reports"
    / "analysis"
    / "threshold_analysis.csv"
)

FEATURES = ["type", "amount", "oldbalanceOrg", "log_amount"]
TARGET = "isFraud"

CURRENT_THRESHOLD = 0.939572

THRESHOLDS = [
    0.50,
    0.70,
    0.80,
    0.85,
    0.90,
    0.92,
    CURRENT_THRESHOLD,
    0.95,
    0.97,
    0.99,
]


def main() -> None:
    """Evaluate candidate thresholds on validation data."""

    print("Loading saved XGBoost model...")
    model = joblib.load(MODEL_PATH)

    print("Loading validation dataset...")
    validation_df = pd.read_parquet(VALIDATION_PATH)

    X_validation = validation_df[FEATURES]
    y_validation = validation_df[TARGET]

    probabilities = model.predict_proba(X_validation)[:, 1]

    results = []

    for threshold in THRESHOLDS:
        predictions = (probabilities >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_validation,
            predictions,
            labels=[0, 1],
        ).ravel()

        results.append(
            {
                "threshold": threshold,
                "precision": precision_score(
                    y_validation, predictions, zero_division=0
                ),
                "recall": recall_score(
                    y_validation, predictions, zero_division=0
                ),
                "f1_score": f1_score(
                    y_validation, predictions, zero_division=0
                ),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_negatives": int(tn),
                "predicted_fraud": int(predictions.sum()),
            }
        )

    results_df = pd.DataFrame(results)

    print("\nVALIDATION THRESHOLD COMPARISON")
    print("=" * 110)

    print(
        results_df.to_string(
            index=False,
            formatters={
                "threshold": "{:.6f}".format,
                "precision": "{:.4f}".format,
                "recall": "{:.4f}".format,
                "f1_score": "{:.4f}".format,
            },
        )
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nResults saved to: {OUTPUT_PATH}")
    print(f"Current production candidate threshold: {CURRENT_THRESHOLD:.6f}")

    if THRESHOLD_PATH.exists():
        saved_threshold = json.loads(
            THRESHOLD_PATH.read_text(encoding="utf-8")
        )["threshold"]
        print(f"Saved model threshold: {saved_threshold:.6f}")


if __name__ == "__main__":
    main()

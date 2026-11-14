
"""Analyze fraud model performance across validation time periods."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[3]

VALIDATION_PATH = (
    PROJECT_ROOT / "data" / "validation" / "validation.parquet"
)
MODEL_PATH = (
    PROJECT_ROOT / "models" / "xgboost" / "xgboost_fraud_model.joblib"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "reports"
    / "analysis"
    / "temporal_performance.csv"
)

FEATURES = ["type", "amount", "oldbalanceOrg", "log_amount"]
TARGET = "isFraud"
THRESHOLD = 0.939572
PERIOD_SIZE = 10


def main() -> None:
    """Evaluate fraud detection metrics across time periods."""

    print("Loading saved XGBoost model...")
    model = joblib.load(MODEL_PATH)

    print("Loading validation dataset...")
    df = pd.read_parquet(VALIDATION_PATH)

    probabilities = model.predict_proba(df[FEATURES])[:, 1]
    df = df.copy()
    df["fraud_probability"] = probabilities
    df["prediction"] = (
        df["fraud_probability"] >= THRESHOLD
    ).astype(int)

    df["period_start"] = (
        ((df["step"] - 1) // PERIOD_SIZE) * PERIOD_SIZE + 1
    )
    df["period_end"] = df["period_start"] + PERIOD_SIZE - 1

    results = []

    for (period_start, period_end), group in df.groupby(
        ["period_start", "period_end"], sort=True
    ):
        y_true = group[TARGET]
        y_pred = group["prediction"]

        results.append(
            {
                "period": f"{period_start}-{period_end}",
                "start_step": int(period_start),
                "end_step": int(period_end),
                "transactions": len(group),
                "actual_fraud": int(y_true.sum()),
                "detected_fraud": int(
                    ((y_true == 1) & (y_pred == 1)).sum()
                ),
                "false_positives": int(
                    ((y_true == 0) & (y_pred == 1)).sum()
                ),
                "false_negatives": int(
                    ((y_true == 1) & (y_pred == 0)).sum()
                ),
                "precision": precision_score(
                    y_true, y_pred, zero_division=0
                ),
                "recall": recall_score(
                    y_true, y_pred, zero_division=0
                ),
                "f1_score": f1_score(
                    y_true, y_pred, zero_division=0
                ),
            }
        )

    results_df = pd.DataFrame(results)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print("\nTEMPORAL VALIDATION PERFORMANCE")
    print("=" * 100)
    print(
        results_df.to_string(
            index=False,
            formatters={
                "precision": "{:.4f}".format,
                "recall": "{:.4f}".format,
                "f1_score": "{:.4f}".format,
            },
        )
    )

    print(f"\nPeriods analyzed: {len(results_df)}")
    print(f"Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()


"""Compare XGBoost performance on seen and unseen test feature combinations."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

TRAIN_PATH = PROJECT_ROOT / "data" / "validation" / "train.parquet"
TEST_PATH = PROJECT_ROOT / "data" / "splits" / "test.parquet"

MODEL_PATH = (
    PROJECT_ROOT / "models" / "xgboost" / "xgboost_fraud_model.joblib"
)
THRESHOLD_PATH = (
    PROJECT_ROOT / "models" / "xgboost" / "decision_threshold.json"
)

FEATURES = ["type", "amount", "oldbalanceOrg", "log_amount"]
TARGET = "isFraud"


def load_artifacts() -> tuple:
    """Load the trained model, threshold, and datasets."""
    required_paths = [
        TRAIN_PATH,
        TEST_PATH,
        MODEL_PATH,
        THRESHOLD_PATH,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    train_df = pd.read_parquet(TRAIN_PATH, columns=FEATURES)
    test_df = pd.read_parquet(TEST_PATH, columns=FEATURES + [TARGET])

    model = joblib.load(MODEL_PATH)

    threshold_data = json.loads(
        THRESHOLD_PATH.read_text(encoding="utf-8")
    )
    threshold = float(threshold_data["threshold"])

    return train_df, test_df, model, threshold


def identify_seen_combinations(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> np.ndarray:
    """Identify test rows whose feature combinations occur in training."""
    train_hashes = pd.util.hash_pandas_object(
        X_train[FEATURES],
        index=False,
    )

    test_hashes = pd.util.hash_pandas_object(
        X_test[FEATURES],
        index=False,
    )

    seen_hashes = pd.Index(train_hashes.unique())

    return test_hashes.isin(seen_hashes).to_numpy()


def evaluate_subset(
    name: str,
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:
    """Calculate metrics for one test subset."""
    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "rows": len(y_true),
        "fraud_cases": int(y_true.sum()),
        "fraud_rate": float(y_true.mean()),
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "f1_score": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "pr_auc": (
            average_precision_score(y_true, probabilities)
            if y_true.nunique() == 2
            else float("nan")
        ),
        "roc_auc": (
            roc_auc_score(y_true, probabilities)
            if y_true.nunique() == 2
            else float("nan")
        ),
    }

    print(f"\n{name}")
    print("=" * 60)
    print(f"Rows:       {metrics['rows']:,}")
    print(f"Fraud:      {metrics['fraud_cases']:,}")
    print(f"Fraud rate: {metrics['fraud_rate']:.4%}")
    print(f"Accuracy:   {metrics['accuracy']:.4f}")
    print(f"Precision:  {metrics['precision']:.4f}")
    print(f"Recall:     {metrics['recall']:.4f}")
    print(f"F1-score:   {metrics['f1_score']:.4f}")
    print(f"PR-AUC:     {metrics['pr_auc']:.4f}")
    print(f"ROC-AUC:    {metrics['roc_auc']:.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, predictions, labels=[0, 1]))

    return metrics


def main() -> None:
    """Compare model performance on seen and unseen test combinations."""
    print("Loading saved model and datasets...")

    X_train, test_df, model, threshold = load_artifacts()

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    print(f"Training rows: {len(X_train):,}")
    print(f"Test rows:     {len(X_test):,}")
    print(f"Threshold:     {threshold:.6f}")

    print("\nIdentifying previously seen feature combinations...")

    seen_mask = identify_seen_combinations(X_train, X_test)
    unseen_mask = ~seen_mask

    print(f"Seen test rows:   {seen_mask.sum():,}")
    print(f"Unseen test rows: {unseen_mask.sum():,}")

    probabilities = model.predict_proba(X_test)[:, 1]

    seen_results = evaluate_subset(
        "TEST — SEEN FEATURE COMBINATIONS",
        y_test.loc[seen_mask].reset_index(drop=True),
        probabilities[seen_mask],
        threshold,
    )

    unseen_results = evaluate_subset(
        "TEST — UNSEEN FEATURE COMBINATIONS",
        y_test.loc[unseen_mask].reset_index(drop=True),
        probabilities[unseen_mask],
        threshold,
    )

    print("\nCOMPARISON")
    print("=" * 60)

    comparison = pd.DataFrame(
        {
            "Seen combinations": seen_results,
            "Unseen combinations": unseen_results,
        }
    )

    print(comparison.to_string(float_format=lambda value: f"{value:.4f}"))

    print("\nAnalysis completed. No model or dataset was modified.")


if __name__ == "__main__":
    main()

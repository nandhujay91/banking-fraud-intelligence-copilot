
"""Train and evaluate a baseline fraud detection model."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]

TRAIN_PATH = PROJECT_ROOT / "data" / "validation" / "train.parquet"
VALIDATION_PATH = PROJECT_ROOT / "data" / "validation" / "validation.parquet"
TEST_PATH = PROJECT_ROOT / "data" / "splits" / "test.parquet"

MODEL_DIR = PROJECT_ROOT / "models" / "baseline"
MODEL_PATH = MODEL_DIR / "logistic_regression.joblib"
THRESHOLD_PATH = MODEL_DIR / "decision_threshold.json"

CATEGORICAL_FEATURES = ["type"]
NUMERICAL_FEATURES = ["amount", "oldbalanceOrg", "log_amount"]
FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
TARGET = "isFraud"


def load_data() -> tuple:
    """Load chronological training, validation, and test datasets."""
    train_df = pd.read_parquet(TRAIN_PATH)
    validation_df = pd.read_parquet(VALIDATION_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_validation = validation_df[FEATURES]
    y_validation = validation_df[TARGET]

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    return X_train, y_train, X_validation, y_validation, X_test, y_test


def build_pipeline() -> Pipeline:
    """Build preprocessing and Logistic Regression pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            ("numerical", StandardScaler(), NUMERICAL_FEATURES),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def select_threshold(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> float:
    """Select the probability threshold that maximizes validation F1."""
    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities,
    )

    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-10)
    )

    best_index = int(np.argmax(f1_scores))
    best_threshold = float(thresholds[best_index])

    print("\nValidation Threshold Selection")
    print("=" * 45)
    print(f"Selected threshold: {best_threshold:.6f}")
    print(f"Validation F1:      {f1_scores[best_index]:.4f}")

    return best_threshold


def evaluate_model(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float,
    dataset_name: str,
) -> None:
    """Evaluate the model using the selected probability threshold."""
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    print(f"\n{dataset_name} Evaluation")
    print("=" * 45)
    print(f"Threshold: {threshold:.6f}")
    print(f"Precision: {precision_score(y, predictions):.4f}")
    print(f"Recall:    {recall_score(y, predictions):.4f}")
    print(f"F1-score:  {f1_score(y, predictions):.4f}")
    print(f"PR-AUC:    {average_precision_score(y, probabilities):.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y, predictions))

    print("\nClassification Report:")
    print(classification_report(y, predictions, zero_division=0))


def main() -> None:
    """Train, tune, evaluate, and save the baseline model."""
    print("Loading chronological datasets...")

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    ) = load_data()

    print(f"Training rows:   {len(X_train):,}")
    print(f"Validation rows: {len(X_validation):,}")
    print(f"Testing rows:    {len(X_test):,}")
    print(f"Input features: {FEATURES}")

    print("\nTraining Logistic Regression...")
    model = build_pipeline()
    model.fit(X_train, y_train)

    validation_probabilities = model.predict_proba(X_validation)[:, 1]

    threshold = select_threshold(
        y_validation,
        validation_probabilities,
    )

    evaluate_model(
        model,
        X_validation,
        y_validation,
        threshold,
        "Validation",
    )

    print("\nEvaluating on untouched final test data...")
    evaluate_model(
        model,
        X_test,
        y_test,
        threshold,
        "Final Test",
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    threshold_data = {
        "threshold": threshold,
        "selection_metric": "validation_f1",
        "features": FEATURES,
        "target": TARGET,
    }

    THRESHOLD_PATH.write_text(
        json.dumps(threshold_data, indent=4),
        encoding="utf-8",
    )

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Threshold saved to: {THRESHOLD_PATH}")


if __name__ == "__main__":
    main()

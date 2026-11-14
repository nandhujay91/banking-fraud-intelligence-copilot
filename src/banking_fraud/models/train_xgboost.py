
"""Train and evaluate an XGBoost fraud detection model."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[3]

TRAIN_PATH = PROJECT_ROOT / "data" / "validation" / "train.parquet"
VALIDATION_PATH = PROJECT_ROOT / "data" / "validation" / "validation.parquet"
TEST_PATH = PROJECT_ROOT / "data" / "splits" / "test.parquet"

MODEL_DIR = PROJECT_ROOT / "models" / "xgboost"
MODEL_PATH = MODEL_DIR / "xgboost_fraud_model.joblib"
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

    return (
        train_df[FEATURES],
        train_df[TARGET],
        validation_df[FEATURES],
        validation_df[TARGET],
        test_df[FEATURES],
        test_df[TARGET],
    )


def build_pipeline(scale_pos_weight: float) -> Pipeline:
    """Build the XGBoost preprocessing and classification pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            ("numerical", "passthrough", NUMERICAL_FEATURES),
        ]
    )

    classifier = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
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
    threshold = float(thresholds[best_index])

    print("\nValidation Threshold Selection")
    print("=" * 55)
    print(f"Selected threshold: {threshold:.6f}")
    print(f"Validation F1:      {f1_scores[best_index]:.4f}")

    return threshold


def evaluate_model(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float,
    dataset_name: str,
) -> dict:
    """Evaluate classification and probability-ranking performance."""
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "f1_score": f1_score(y, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y, probabilities),
        "pr_auc": average_precision_score(y, probabilities),
    }

    print(f"\n{dataset_name} Evaluation")
    print("=" * 55)
    print(f"Threshold:  {threshold:.6f}")
    print(f"Accuracy:   {metrics['accuracy']:.4f}")
    print(f"Precision:  {metrics['precision']:.4f}")
    print(f"Recall:     {metrics['recall']:.4f}")
    print(f"F1-score:   {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:    {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:     {metrics['pr_auc']:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y, predictions, labels=[0, 1]))

    print("\nClassification Report:")
    print(
        classification_report(
            y,
            predictions,
            labels=[0, 1],
            target_names=["Non-Fraud", "Fraud"],
            zero_division=0,
        )
    )

    return metrics


def compare_performance(results: dict) -> None:
    """Compare training, validation, and test performance."""
    print("\nPerformance Comparison")
    print("=" * 75)

    comparison = pd.DataFrame(results).T
    print(comparison.to_string(float_format=lambda value: f"{value:.4f}"))

    print("\nGeneralisation Gap (Training - Test)")
    print("=" * 55)

    for metric in ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]:
        gap = results["Training"][metric] - results["Final Test"][metric]
        print(f"{metric:<12}: {gap:.4f}")


def main() -> None:
    """Train, tune, evaluate, compare, and save the XGBoost model."""
    print("Loading chronological datasets...")

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    ) = load_data()

    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    scale_pos_weight = negative_count / positive_count

    print(f"Training rows:   {len(X_train):,}")
    print(f"Validation rows: {len(X_validation):,}")
    print(f"Testing rows:    {len(X_test):,}")
    print(f"Input features: {FEATURES}")
    print(f"Scale positive weight: {scale_pos_weight:.2f}")

    print("\nTraining XGBoost...")
    model = build_pipeline(scale_pos_weight)
    model.fit(X_train, y_train)

    validation_probabilities = model.predict_proba(X_validation)[:, 1]
    threshold = select_threshold(y_validation, validation_probabilities)

    # Evaluate all datasets using the same validation-selected threshold.
    results = {}

    results["Training"] = evaluate_model(
        model,
        X_train,
        y_train,
        threshold,
        "Training",
    )

    results["Validation"] = evaluate_model(
        model,
        X_validation,
        y_validation,
        threshold,
        "Validation",
    )

    print("\nEvaluating on untouched final test data...")
    results["Final Test"] = evaluate_model(
        model,
        X_test,
        y_test,
        threshold,
        "Final Test",
    )

    compare_performance(results)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    threshold_data = {
        "threshold": threshold,
        "selection_metric": "validation_f1",
        "features": FEATURES,
        "target": TARGET,
        "scale_pos_weight": scale_pos_weight,
    }

    THRESHOLD_PATH.write_text(
        json.dumps(threshold_data, indent=4),
        encoding="utf-8",
    )

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Threshold saved to: {THRESHOLD_PATH}")


if __name__ == "__main__":
    main()

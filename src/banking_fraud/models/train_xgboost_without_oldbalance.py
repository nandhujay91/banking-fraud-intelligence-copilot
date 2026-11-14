
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
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

OUTPUT_DIR = PROJECT_ROOT / "models" / "xgboost_ablation_without_oldbalance"
MODEL_PATH = OUTPUT_DIR / "xgboost_without_oldbalance.joblib"
THRESHOLD_PATH = OUTPUT_DIR / "decision_threshold.json"

FEATURES = ["type", "amount", "log_amount"]
TARGET = "isFraud"
RANDOM_STATE = 42


def load_split(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_parquet(path)

    X = data[FEATURES]
    y = data[TARGET]

    return X, y


def select_threshold(
    y_true: pd.Series,
    probabilities,
) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(
        y_true, probabilities
    )

    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )

    best_index = f1_scores.argmax()

    return float(thresholds[best_index]), float(f1_scores[best_index])


def evaluate_model(
    name: str,
    y_true: pd.Series,
    probabilities,
    threshold: float,
) -> dict:
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    metrics = {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y_true, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    print(f"\n{name}")
    print("-" * 50)

    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric:22}: {value:.4f}")
        else:
            print(f"{metric:22}: {value:,}")

    return metrics


def main() -> None:
    print("Loading chronological datasets...")

    X_train, y_train = load_split(TRAIN_PATH)
    X_validation, y_validation = load_split(VALIDATION_PATH)
    X_test, y_test = load_split(TEST_PATH)

    print(f"Training rows:   {len(X_train):,}")
    print(f"Validation rows: {len(X_validation):,}")
    print(f"Test rows:       {len(X_test):,}")

    positive_count = int(y_train.sum())
    negative_count = len(y_train) - positive_count
    scale_pos_weight = negative_count / positive_count

    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                ["type"],
            ),
            (
                "numerical",
                "passthrough",
                ["amount", "log_amount"],
            ),
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
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    print("\nTraining XGBoost without oldbalanceOrg...")

    pipeline.fit(X_train, y_train)

    validation_probabilities = pipeline.predict_proba(
        X_validation
    )[:, 1]

    threshold, validation_f1 = select_threshold(
        y_validation,
        validation_probabilities,
    )

    print(f"\nSelected validation threshold: {threshold:.6f}")
    print(f"Validation F1 at selected threshold: {validation_f1:.4f}")

    test_probabilities = pipeline.predict_proba(X_test)[:, 1]

    evaluate_model(
        "FINAL TEST — WITHOUT oldbalanceOrg",
        y_test,
        test_probabilities,
        threshold,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, MODEL_PATH)

    threshold_data = {
        "threshold": threshold,
        "selection_metric": "validation_f1",
        "features": FEATURES,
        "target": TARGET,
        "scale_pos_weight": scale_pos_weight,
        "excluded_feature": "oldbalanceOrg",
    }

    with THRESHOLD_PATH.open("w", encoding="utf-8") as file:
        json.dump(threshold_data, file, indent=4)

    print("\nSaved model:", MODEL_PATH)
    print("Saved threshold:", THRESHOLD_PATH)


if __name__ == "__main__":
    main()

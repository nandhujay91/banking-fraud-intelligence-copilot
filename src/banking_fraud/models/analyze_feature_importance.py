
"""Analyze feature importance from the trained XGBoost model."""

from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "xgboost"
    / "xgboost_fraud_model.joblib"
)


def main() -> None:
    """Extract and display XGBoost feature importance."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()
    booster = classifier.get_booster()

    importance_types = ["gain", "weight", "cover"]
    results = pd.DataFrame({"feature": feature_names})

    for importance_type in importance_types:
        scores = booster.get_score(importance_type=importance_type)

        results[importance_type] = [
            scores.get(f"f{index}", 0.0)
            for index in range(len(feature_names))
        ]

    results = results.sort_values(
        by="gain",
        ascending=False,
    ).reset_index(drop=True)

    print("\nXGBoost Feature Importance")
    print("=" * 70)
    print(results.to_string(index=False, float_format="%.4f"))

    print("\nMost influential feature by Gain:")
    print(results.iloc[0]["feature"])


if __name__ == "__main__":
    main()

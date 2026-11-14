
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = PROJECT_ROOT / "models/xgboost/xgboost_fraud_model.joblib"
TEST_DATA_PATH = PROJECT_ROOT / "data/splits/test.parquet"
OUTPUT_DIR = PROJECT_ROOT / "docs/reports/figures"

FEATURE_COLUMNS = [
    "type",
    "amount",
    "oldbalanceOrg",
    "log_amount",
]

SAMPLE_SIZE = 2_000
RANDOM_STATE = 42


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading trained XGBoost pipeline...")
    pipeline = joblib.load(MODEL_PATH)

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    print("Loading final test dataset...")
    test_data = pd.read_parquet(TEST_DATA_PATH)

    X_test = test_data[FEATURE_COLUMNS]

    # Select a reproducible sample for global SHAP analysis.
    X_sample = X_test.sample(
        n=min(SAMPLE_SIZE, len(X_test)),
        random_state=RANDOM_STATE,
    )

    print(f"Global SHAP sample: {len(X_sample):,} transactions")

    # Apply the same preprocessing used during model training.
    X_transformed = preprocessor.transform(X_sample)
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer(X_transformed)

    # Attach transformed feature names for readable SHAP plots.
    shap_values.feature_names = list(feature_names)

    # Global feature impact.
    shap.plots.beeswarm(
        shap_values,
        max_display=10,
        show=False,
    )

    plt.savefig(
        OUTPUT_DIR / "shap_summary.png",
        bbox_inches="tight",
        dpi=150,
    )
    plt.close()

    # Mean absolute SHAP importance.
    shap.plots.bar(
        shap_values,
        max_display=10,
        show=False,
    )

    plt.savefig(
        OUTPUT_DIR / "shap_feature_importance.png",
        bbox_inches="tight",
        dpi=150,
    )
    plt.close()

    print("\nGlobal SHAP analysis completed.")

    # Select three actual fraudulent transactions.
    fraud_data = test_data.loc[test_data["isFraud"] == 1]

    fraud_examples = fraud_data.sample(
        n=min(3, len(fraud_data)),
        random_state=RANDOM_STATE,
    )

    X_fraud = fraud_examples[FEATURE_COLUMNS]
    X_fraud_transformed = preprocessor.transform(X_fraud)

    fraud_shap_values = explainer(X_fraud_transformed)
    fraud_shap_values.feature_names = list(feature_names)

    # Generate individual fraud explanations.
    for index in range(len(fraud_examples)):
        shap.plots.waterfall(
            fraud_shap_values[index],
            max_display=10,
            show=False,
        )

        output_path = OUTPUT_DIR / f"shap_fraud_example_{index + 1}.png"

        plt.savefig(
            output_path,
            bbox_inches="tight",
            dpi=150,
        )
        plt.close()

        print(f"Saved: {output_path.name}")

    print("\nSHAP analysis completed successfully.")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

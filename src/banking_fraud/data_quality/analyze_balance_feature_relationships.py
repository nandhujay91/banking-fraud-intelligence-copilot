"""Analyze relationships between balance-related fraud features."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "paysim.csv"

CHUNK_SIZE = 250_000
BALANCE_TOLERANCE = 0.01

TRANSACTION_TYPES = {"TRANSFER", "CASH_OUT"}

REQUIRED_COLUMNS = {
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
}

FEATURE_COLUMNS = [
    "amount",
    "sender_balance_change",
    "recipient_balance_change",
    "sender_balance_difference",
    "recipient_balance_difference",
    "sender_unchanged",
    "recipient_unchanged",
    "sender_zero_before",
    "sender_zero_after",
    "recipient_zero_before",
    "recipient_zero_after",
    "sender_balance_mismatch",
    "recipient_balance_mismatch",
    "isFraud",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)


def create_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Create numerical balance-related features for analysis."""

    features = pd.DataFrame(index=transactions.index)

    sender_change = transactions["oldbalanceOrg"] - transactions["newbalanceOrig"]

    recipient_change = transactions["newbalanceDest"] - transactions["oldbalanceDest"]

    sender_difference = sender_change - transactions["amount"]
    recipient_difference = recipient_change - transactions["amount"]

    features["amount"] = transactions["amount"]
    features["sender_balance_change"] = sender_change
    features["recipient_balance_change"] = recipient_change
    features["sender_balance_difference"] = sender_difference
    features["recipient_balance_difference"] = recipient_difference

    features["sender_unchanged"] = (sender_change.abs() <= BALANCE_TOLERANCE).astype(
        "int8"
    )

    features["recipient_unchanged"] = (
        recipient_change.abs() <= BALANCE_TOLERANCE
    ).astype("int8")

    features["sender_zero_before"] = (transactions["oldbalanceOrg"] == 0).astype("int8")

    features["sender_zero_after"] = (transactions["newbalanceOrig"] == 0).astype("int8")

    features["recipient_zero_before"] = (transactions["oldbalanceDest"] == 0).astype(
        "int8"
    )

    features["recipient_zero_after"] = (transactions["newbalanceDest"] == 0).astype(
        "int8"
    )

    features["sender_balance_mismatch"] = (
        sender_difference.abs() > BALANCE_TOLERANCE
    ).astype("int8")

    features["recipient_balance_mismatch"] = (
        recipient_difference.abs() > BALANCE_TOLERANCE
    ).astype("int8")

    features["isFraud"] = transactions["isFraud"].astype("int8")

    return features[FEATURE_COLUMNS]


def update_statistics(
    statistics: dict[str, dict[str, object]],
    transaction_type: str,
    features: pd.DataFrame,
) -> None:
    """Accumulate sums and cross-products for correlation analysis."""

    values = features.to_numpy(dtype=np.float64)

    if transaction_type not in statistics:
        feature_count = len(FEATURE_COLUMNS)

        statistics[transaction_type] = {
            "count": 0,
            "sum": np.zeros(feature_count, dtype=np.float64),
            "cross_products": np.zeros(
                (feature_count, feature_count),
                dtype=np.float64,
            ),
        }

    group_statistics = statistics[transaction_type]

    group_statistics["count"] += len(values)
    group_statistics["sum"] += values.sum(axis=0)
    group_statistics["cross_products"] += values.T @ values


def create_correlation_matrix(
    group_statistics: dict[str, object],
) -> pd.DataFrame:
    """Calculate a Pearson correlation matrix from accumulated statistics."""

    count = group_statistics["count"]
    feature_sums = group_statistics["sum"]
    cross_products = group_statistics["cross_products"]

    centered_cross_products = (
        cross_products - np.outer(feature_sums, feature_sums) / count
    )

    variances = np.diag(centered_cross_products)

    standard_deviations = np.sqrt(np.maximum(variances, 0))

    denominators = np.outer(
        standard_deviations,
        standard_deviations,
    )

    correlations = np.divide(
        centered_cross_products,
        denominators,
        out=np.full_like(centered_cross_products, np.nan),
        where=denominators > 0,
    )

    return pd.DataFrame(
        correlations,
        index=FEATURE_COLUMNS,
        columns=FEATURE_COLUMNS,
    )


def print_correlation_results(
    transaction_type: str,
    statistics: dict[str, object],
) -> None:
    """Display feature relationships and correlations with fraud labels."""

    correlation_matrix = create_correlation_matrix(statistics)

    print("\n" + "=" * 100)
    print(f"{transaction_type} — FEATURE CORRELATION MATRIX")
    print("=" * 100)

    print(f"Transactions analysed: {statistics['count']:,}")

    print(correlation_matrix.to_string(float_format=lambda value: f"{value:.3f}"))

    fraud_correlations = (
        correlation_matrix["isFraud"]
        .drop("isFraud")
        .dropna()
        .sort_values(key=lambda values: values.abs(), ascending=False)
    )

    print("\nFEATURE CORRELATION WITH FRAUD LABEL")

    print(fraud_correlations.to_string(float_format=lambda value: f"{value:.4f}"))

    print("\nHIGHLY CORRELATED FEATURE PAIRS (ABSOLUTE CORRELATION >= 0.90)")

    upper_triangle = correlation_matrix.where(
        np.triu(
            np.ones(correlation_matrix.shape, dtype=bool),
            k=1,
        )
    )

    correlated_pairs = []

    for row_name, row in upper_triangle.iterrows():
        for column_name, correlation in row.items():
            if pd.notna(correlation) and abs(correlation) >= 0.90:
                correlated_pairs.append(
                    {
                        "feature_1": row_name,
                        "feature_2": column_name,
                        "correlation": correlation,
                    }
                )

    if correlated_pairs:
        print(
            pd.DataFrame(correlated_pairs).to_string(
                index=False,
                formatters={
                    "correlation": "{:.4f}".format,
                },
            )
        )
    else:
        print("No feature pairs met the correlation threshold.")


def analyze_balance_feature_relationships() -> None:
    """Analyse balance feature relationships by transaction type."""

    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"PaySim dataset not found: {DATASET_PATH}")

    statistics: dict[str, dict[str, object]] = {}
    total_rows = 0

    logger.info("Starting balance feature relationship analysis.")
    logger.info("Dataset: %s", DATASET_PATH)

    for chunk_number, transactions in enumerate(
        pd.read_csv(
            DATASET_PATH,
            chunksize=CHUNK_SIZE,
            usecols=lambda column: column in REQUIRED_COLUMNS,
        ),
        start=1,
    ):
        if chunk_number == 1:
            missing_columns = REQUIRED_COLUMNS - set(transactions.columns)

            if missing_columns:
                raise ValueError(
                    f"Dataset is missing required columns: {sorted(missing_columns)}"
                )

        if transactions[list(REQUIRED_COLUMNS)].isna().any().any():
            raise ValueError(f"Missing required values in chunk {chunk_number}.")

        if not transactions["isFraud"].isin({0, 1}).all():
            raise ValueError(f"Invalid fraud labels in chunk {chunk_number}.")

        if (
            not transactions["type"]
            .isin(TRANSACTION_TYPES | {"CASH_IN", "DEBIT", "PAYMENT"})
            .all()
        ):
            raise ValueError(f"Unexpected transaction type in chunk {chunk_number}.")

        relevant_transactions = transactions.loc[
            transactions["type"].isin(TRANSACTION_TYPES)
        ]

        for transaction_type, group in relevant_transactions.groupby("type"):
            features = create_features(group)

            update_statistics(
                statistics,
                str(transaction_type),
                features,
            )

        total_rows += len(transactions)

        if chunk_number % 5 == 0:
            logger.info(
                "Processed %s dataset transactions.",
                f"{total_rows:,}",
            )

    if not statistics:
        raise ValueError("No TRANSFER or CASH_OUT transactions were found.")

    for transaction_type, group_statistics in sorted(statistics.items()):
        print_correlation_results(
            transaction_type,
            group_statistics,
        )

    logger.info(
        "Completed analysis of %s TRANSFER and CASH_OUT transactions.",
        f"{sum(group['count'] for group in statistics.values()):,}",
    )


if __name__ == "__main__":
    analyze_balance_feature_relationships()

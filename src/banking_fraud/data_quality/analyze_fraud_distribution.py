"""Analyze fraud distribution across transaction types, amounts and time."""

import logging
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "paysim.csv"

CHUNK_SIZE = 250_000
FRAUD_LABEL = 1
STEP_INTERVAL = 50

AMOUNT_BINS = [
    0,
    1_000,
    10_000,
    100_000,
    1_000_000,
    float("inf"),
]

AMOUNT_LABELS = [
    "0-1K",
    "1K-10K",
    "10K-100K",
    "100K-1M",
    "Above 1M",
]

REQUIRED_COLUMNS = {
    "step",
    "type",
    "amount",
    "isFraud",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)


def initialize_statistics() -> dict:
    """Initialize counters used during fraud analysis."""

    return {
        "total_transactions": 0,
        "fraud_transactions": 0,
        "transaction_types": {},
        "fraud_by_type": {},
        "fraud_by_amount": {},
        "fraud_by_step": {},
        "fraud_amount_sum": 0.0,
        "minimum_fraud_amount": float("inf"),
        "maximum_fraud_amount": float("-inf"),
    }


def update_category_counts(
    counts: dict[str, int],
    values: pd.Series,
) -> None:
    """Add category counts from a dataset chunk."""

    for category, count in values.value_counts().items():
        counts[str(category)] = counts.get(str(category), 0) + int(count)


def process_transactions(
    transactions: pd.DataFrame,
    statistics: dict,
) -> None:
    """Accumulate transaction and fraud statistics."""

    statistics["total_transactions"] += len(transactions)

    update_category_counts(
        statistics["transaction_types"],
        transactions["type"],
    )

    fraud_transactions = transactions.loc[transactions["isFraud"] == FRAUD_LABEL]

    if fraud_transactions.empty:
        return

    statistics["fraud_transactions"] += len(fraud_transactions)

    update_category_counts(
        statistics["fraud_by_type"],
        fraud_transactions["type"],
    )

    amount_ranges = pd.cut(
        fraud_transactions["amount"],
        bins=AMOUNT_BINS,
        labels=AMOUNT_LABELS,
        include_lowest=True,
    )

    update_category_counts(
        statistics["fraud_by_amount"],
        amount_ranges,
    )

    step_periods = (
        (fraud_transactions["step"] - 1) // STEP_INTERVAL
    ) * STEP_INTERVAL + 1

    step_labels = (
        step_periods.astype(str) + "-" + (step_periods + STEP_INTERVAL - 1).astype(str)
    )

    update_category_counts(
        statistics["fraud_by_step"],
        step_labels,
    )

    statistics["fraud_amount_sum"] += float(fraud_transactions["amount"].sum())

    statistics["minimum_fraud_amount"] = min(
        statistics["minimum_fraud_amount"],
        float(fraud_transactions["amount"].min()),
    )

    statistics["maximum_fraud_amount"] = max(
        statistics["maximum_fraud_amount"],
        float(fraud_transactions["amount"].max()),
    )


def create_transaction_type_report(
    statistics: dict,
) -> pd.DataFrame:
    """Calculate fraud rates for individual transaction types."""

    report_rows = []

    for transaction_type, total in sorted(statistics["transaction_types"].items()):
        fraud_count = statistics["fraud_by_type"].get(
            transaction_type,
            0,
        )

        report_rows.append(
            {
                "transaction_type": transaction_type,
                "total_transactions": total,
                "fraud_transactions": fraud_count,
                "fraud_rate_pct": fraud_count / total * 100,
            }
        )

    return pd.DataFrame(report_rows)


def create_distribution_report(
    counts: dict[str, int],
    total_fraud: int,
    category_name: str,
) -> pd.DataFrame:
    """Calculate each category's share of all fraud transactions."""

    report = pd.DataFrame(
        [
            {
                category_name: category,
                "fraud_transactions": count,
                "fraud_share_pct": count / total_fraud * 100,
            }
            for category, count in sorted(counts.items())
        ]
    )

    return report


def print_report(statistics: dict) -> None:
    """Display the fraud distribution findings."""

    total_transactions = statistics["total_transactions"]
    total_fraud = statistics["fraud_transactions"]

    fraud_rate = total_fraud / total_transactions * 100

    print("\n" + "=" * 85)
    print("PAYSIM FRAUD DISTRIBUTION REPORT")
    print("=" * 85)

    print("\nDATASET OVERVIEW")
    print(f"Total transactions: {total_transactions:,}")
    print(f"Fraud transactions: {total_fraud:,}")
    print(f"Overall fraud rate: {fraud_rate:.4f}%")

    print("\nFRAUD DISTRIBUTION BY TRANSACTION TYPE")

    type_report = create_transaction_type_report(statistics)

    print(
        type_report.to_string(
            index=False,
            formatters={
                "fraud_rate_pct": "{:.4f}%".format,
            },
        )
    )

    average_amount = statistics["fraud_amount_sum"] / total_fraud

    print("\nFRAUD TRANSACTION AMOUNTS")
    print(f"Minimum amount: {statistics['minimum_fraud_amount']:,.2f}")
    print(f"Maximum amount: {statistics['maximum_fraud_amount']:,.2f}")
    print(f"Average amount: {average_amount:,.2f}")

    amount_report = create_distribution_report(
        statistics["fraud_by_amount"],
        total_fraud,
        "amount_range",
    )

    print("\nFRAUD DISTRIBUTION BY AMOUNT RANGE")
    print(
        amount_report.to_string(
            index=False,
            formatters={
                "fraud_share_pct": "{:.2f}%".format,
            },
        )
    )

    step_report = create_distribution_report(
        statistics["fraud_by_step"],
        total_fraud,
        "simulation_period",
    )

    print("\nFRAUD DISTRIBUTION BY SIMULATION PERIOD")
    print(
        step_report.to_string(
            index=False,
            formatters={
                "fraud_share_pct": "{:.2f}%".format,
            },
        )
    )

    print("\n" + "=" * 85)
    print("FRAUD DISTRIBUTION ANALYSIS COMPLETED")
    print("=" * 85)


def analyze_fraud_distribution() -> None:
    """Analyze fraud distribution using chunked dataset processing."""

    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"PaySim dataset not found: {DATASET_PATH}")

    statistics = initialize_statistics()

    logger.info("Starting fraud distribution analysis.")
    logger.info("Dataset: %s", DATASET_PATH)

    for chunk_number, transactions in enumerate(
        pd.read_csv(
            DATASET_PATH,
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):
        if chunk_number == 1:
            missing_columns = REQUIRED_COLUMNS - set(transactions.columns)

            if missing_columns:
                raise ValueError(
                    f"Dataset is missing required columns: {sorted(missing_columns)}"
                )

        process_transactions(
            transactions,
            statistics,
        )

    if statistics["total_transactions"] == 0:
        raise ValueError("The PaySim dataset contains no records.")

    if statistics["fraud_transactions"] == 0:
        raise ValueError("No fraudulent transactions were found.")

    print_report(statistics)

    logger.info(
        "Fraud distribution analysis completed. "
        "Processed %s transactions and found %s fraud records.",
        f"{statistics['total_transactions']:,}",
        f"{statistics['fraud_transactions']:,}",
    )


if __name__ == "__main__":
    analyze_fraud_distribution()

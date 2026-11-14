"""Analyze account balance patterns in the PaySim dataset."""

import logging
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "paysim.csv"

CHUNK_SIZE = 250_000
BALANCE_TOLERANCE = 0.01

TRANSACTION_TYPES = (
    "PAYMENT",
    "TRANSFER",
    "CASH_OUT",
    "CASH_IN",
    "DEBIT",
)

FRAUD_LABELS = (0, 1)

BALANCE_COLUMNS = [
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)


def initialize_statistics() -> dict[tuple[str, int], dict[str, int]]:
    """Initialize counters for each transaction type and fraud label."""

    statistics = {}

    for transaction_type in TRANSACTION_TYPES:
        for fraud_label in FRAUD_LABELS:
            statistics[(transaction_type, fraud_label)] = {
                "total_transactions": 0,
                "sender_unchanged": 0,
                "recipient_unchanged": 0,
                "sender_zero_before_after": 0,
                "recipient_zero_before_after": 0,
            }

    return statistics


def validate_dataset_columns(columns: list[str]) -> None:
    """Check that all required balance and label columns exist."""

    required_columns = {
        "type",
        "isFraud",
        *BALANCE_COLUMNS,
    }

    missing_columns = required_columns - set(columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )


def analyze_chunk(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Identify unchanged and zero-balance patterns in a chunk."""

    results = pd.DataFrame(index=transactions.index)

    results["type"] = transactions["type"]
    results["isFraud"] = transactions["isFraud"]

    # Check whether the sender balance remains unchanged.
    results["sender_unchanged"] = (
        transactions["oldbalanceOrg"] - transactions["newbalanceOrig"]
    ).abs() <= BALANCE_TOLERANCE

    # Check whether the recipient balance remains unchanged.
    results["recipient_unchanged"] = (
        transactions["oldbalanceDest"] - transactions["newbalanceDest"]
    ).abs() <= BALANCE_TOLERANCE

    # Check whether sender balances are zero before and after.
    results["sender_zero_before_after"] = (transactions["oldbalanceOrg"] == 0) & (
        transactions["newbalanceOrig"] == 0
    )

    # Check whether recipient balances are zero before and after.
    results["recipient_zero_before_after"] = (transactions["oldbalanceDest"] == 0) & (
        transactions["newbalanceDest"] == 0
    )

    return results


def update_statistics(
    statistics: dict[tuple[str, int], dict[str, int]],
    results: pd.DataFrame,
) -> None:
    """Accumulate balance pattern counts by transaction group."""

    grouped_results = results.groupby(
        ["type", "isFraud"],
        observed=True,
    ).agg(
        total_transactions=("isFraud", "size"),
        sender_unchanged=("sender_unchanged", "sum"),
        recipient_unchanged=("recipient_unchanged", "sum"),
        sender_zero_before_after=("sender_zero_before_after", "sum"),
        recipient_zero_before_after=("recipient_zero_before_after", "sum"),
    )

    for (transaction_type, fraud_label), row in grouped_results.iterrows():
        key = (transaction_type, int(fraud_label))

        if key not in statistics:
            continue

        for metric in statistics[key]:
            statistics[key][metric] += int(row[metric])


def create_report(
    statistics: dict[tuple[str, int], dict[str, int]],
) -> pd.DataFrame:
    """Build a report containing counts and percentages."""

    report_rows = []

    percentage_metrics = (
        "sender_unchanged",
        "recipient_unchanged",
        "sender_zero_before_after",
        "recipient_zero_before_after",
    )

    for (transaction_type, fraud_label), values in statistics.items():
        total = values["total_transactions"]

        if total == 0:
            continue

        row = {
            "transaction_type": transaction_type,
            "fraud_label": fraud_label,
            "total_transactions": total,
        }

        for metric in percentage_metrics:
            count = values[metric]

            row[f"{metric}_count"] = count
            row[f"{metric}_pct"] = count / total * 100

        report_rows.append(row)

    return pd.DataFrame(report_rows)


def print_report(report: pd.DataFrame, total_rows: int) -> None:
    """Display the account balance pattern report."""

    print("\n" + "=" * 110)
    print("PAYSIM ACCOUNT BALANCE PATTERN REPORT")
    print("=" * 110)

    print(f"\nTotal transactions: {total_rows:,}")
    print(f"Balance tolerance: {BALANCE_TOLERANCE:.2f}")

    percentage_columns = [
        "sender_unchanged_pct",
        "recipient_unchanged_pct",
        "sender_zero_before_after_pct",
        "recipient_zero_before_after_pct",
    ]

    formatted_report = report.copy()

    for column in percentage_columns:
        formatted_report[column] = formatted_report[column].map(
            lambda value: f"{value:.2f}%"
        )

    print("\nBALANCE PATTERNS BY TRANSACTION TYPE AND FRAUD LABEL")
    print(formatted_report.to_string(index=False))

    print("\n" + "=" * 110)
    print("ACCOUNT BALANCE PATTERN ANALYSIS COMPLETED")
    print("=" * 110)


def analyze_balance_patterns() -> None:
    """Process PaySim transactions and summarize account balance patterns."""

    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"PaySim dataset not found: {DATASET_PATH}")

    statistics = initialize_statistics()
    total_rows = 0

    logger.info("Starting account balance pattern analysis.")
    logger.info("Dataset: %s", DATASET_PATH)

    for chunk_number, transactions in enumerate(
        pd.read_csv(
            DATASET_PATH,
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):
        if chunk_number == 1:
            validate_dataset_columns(transactions.columns.tolist())

        total_rows += len(transactions)

        results = analyze_chunk(transactions)

        update_statistics(
            statistics,
            results,
        )

    if total_rows == 0:
        raise ValueError("The PaySim dataset contains no records.")

    report = create_report(statistics)

    if report.empty:
        raise ValueError("No balance pattern statistics were generated.")

    print_report(report, total_rows)

    logger.info(
        "Processed %s transactions.",
        f"{total_rows:,}",
    )


if __name__ == "__main__":
    analyze_balance_patterns()

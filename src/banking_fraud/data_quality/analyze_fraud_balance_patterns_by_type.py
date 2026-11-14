"""Compare balance patterns between fraud and non-fraud transactions."""

import logging
from pathlib import Path

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

BALANCE_COLUMNS = [
    "amount",
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


def initialize_statistics() -> dict[tuple[str, int], dict[str, float]]:
    """Initialize counters for each transaction type and fraud label."""

    return {}


def update_statistics(
    statistics: dict[tuple[str, int], dict[str, float]],
    transactions: pd.DataFrame,
) -> None:
    """Accumulate balance behaviour statistics for each group."""

    for (transaction_type, fraud_label), group in transactions.groupby(
        ["type", "isFraud"]
    ):
        key = (str(transaction_type), int(fraud_label))

        if key not in statistics:
            statistics[key] = {
                "total_transactions": 0,
                "sender_unchanged": 0,
                "recipient_unchanged": 0,
                "sender_zero_before": 0,
                "sender_zero_after": 0,
                "recipient_zero_before": 0,
                "recipient_zero_after": 0,
                "sender_balance_mismatch": 0,
                "recipient_balance_mismatch": 0,
            }

        sender_change = group["oldbalanceOrg"] - group["newbalanceOrig"]

        recipient_change = group["newbalanceDest"] - group["oldbalanceDest"]

        sender_mismatch = (sender_change - group["amount"]).abs() > BALANCE_TOLERANCE

        recipient_mismatch = (
            recipient_change - group["amount"]
        ).abs() > BALANCE_TOLERANCE

        statistics[key]["total_transactions"] += len(group)

        statistics[key]["sender_unchanged"] += int(
            (sender_change.abs() <= BALANCE_TOLERANCE).sum()
        )

        statistics[key]["recipient_unchanged"] += int(
            (recipient_change.abs() <= BALANCE_TOLERANCE).sum()
        )

        statistics[key]["sender_zero_before"] += int(
            (group["oldbalanceOrg"] == 0).sum()
        )

        statistics[key]["sender_zero_after"] += int(
            (group["newbalanceOrig"] == 0).sum()
        )

        statistics[key]["recipient_zero_before"] += int(
            (group["oldbalanceDest"] == 0).sum()
        )

        statistics[key]["recipient_zero_after"] += int(
            (group["newbalanceDest"] == 0).sum()
        )

        statistics[key]["sender_balance_mismatch"] += int(sender_mismatch.sum())

        statistics[key]["recipient_balance_mismatch"] += int(recipient_mismatch.sum())


def create_report(
    statistics: dict[tuple[str, int], dict[str, float]],
) -> pd.DataFrame:
    """Create percentage-based balance behaviour comparisons."""

    report_rows = []

    for (transaction_type, fraud_label), values in sorted(statistics.items()):
        total = int(values["total_transactions"])

        report_row = {
            "transaction_type": transaction_type,
            "fraud_label": fraud_label,
            "total_transactions": total,
        }

        percentage_columns = [
            "sender_unchanged",
            "recipient_unchanged",
            "sender_zero_before",
            "sender_zero_after",
            "recipient_zero_before",
            "recipient_zero_after",
            "sender_balance_mismatch",
            "recipient_balance_mismatch",
        ]

        for column in percentage_columns:
            count = values[column]
            report_row[f"{column}_pct"] = count / total * 100

        report_rows.append(report_row)

    return pd.DataFrame(report_rows)


def print_report(report: pd.DataFrame) -> None:
    """Display balance behaviour comparisons."""

    print("\n" + "=" * 110)
    print("FRAUD VS NON-FRAUD BALANCE PATTERNS")
    print("=" * 110)

    percentage_columns = [
        "sender_unchanged_pct",
        "recipient_unchanged_pct",
        "sender_zero_before_pct",
        "sender_zero_after_pct",
        "recipient_zero_before_pct",
        "recipient_zero_after_pct",
        "sender_balance_mismatch_pct",
        "recipient_balance_mismatch_pct",
    ]

    formatters = {column: "{:.2f}%".format for column in percentage_columns}

    print(
        report.to_string(
            index=False,
            formatters=formatters,
        )
    )

    print("\n" + "=" * 110)
    print("BALANCE PATTERN ANALYSIS COMPLETED")
    print("=" * 110)


def analyze_fraud_balance_patterns() -> None:
    """Process PaySim data and compare fraud-related balance patterns."""

    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"PaySim dataset not found: {DATASET_PATH}")

    statistics = initialize_statistics()
    total_rows = 0

    logger.info("Starting fraud balance pattern analysis.")
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

        if not transactions["type"].isin(TRANSACTION_TYPES).all():
            transactions = transactions.loc[
                transactions["type"].isin(TRANSACTION_TYPES)
            ]

        if not transactions["isFraud"].isin({0, 1}).all():
            raise ValueError(f"Invalid fraud labels in chunk {chunk_number}.")

        if (transactions[BALANCE_COLUMNS] < 0).any().any():
            raise ValueError(f"Negative monetary values found in chunk {chunk_number}.")

        total_rows += len(transactions)

        update_statistics(statistics, transactions)

        if chunk_number % 5 == 0:
            logger.info(
                "Processed %s relevant transactions.",
                f"{total_rows:,}",
            )

    if not statistics:
        raise ValueError("No TRANSFER or CASH_OUT transactions were found.")

    report = create_report(statistics)

    print_report(report)

    logger.info(
        "Analysed %s TRANSFER and CASH_OUT transactions.",
        f"{total_rows:,}",
    )


if __name__ == "__main__":
    analyze_fraud_balance_patterns()

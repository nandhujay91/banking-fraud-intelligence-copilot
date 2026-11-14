"""Analyze balance mismatches by transaction type and fraud label."""

import logging
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "paysim.csv"

CHUNK_SIZE = 250_000
BALANCE_TOLERANCE = 0.01
FRAUD_LABELS = (0, 1)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)


def initialize_statistics() -> dict[tuple[str, int], dict[str, int]]:
    """Create empty statistics for each transaction type and fraud label."""

    statistics = {}

    for transaction_type in (
        "PAYMENT",
        "TRANSFER",
        "CASH_OUT",
        "CASH_IN",
        "DEBIT",
    ):
        for fraud_label in FRAUD_LABELS:
            statistics[(transaction_type, fraud_label)] = {
                "total": 0,
                "sender_mismatches": 0,
                "recipient_mismatches": 0,
            }

    return statistics


def calculate_balance_mismatches(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Identify sender and recipient balance mismatches."""

    sender_balance_change = (
        transactions["oldbalanceOrg"] - transactions["newbalanceOrig"]
    )

    recipient_balance_change = (
        transactions["newbalanceDest"] - transactions["oldbalanceDest"]
    )

    return pd.DataFrame(
        {
            "type": transactions["type"],
            "isFraud": transactions["isFraud"],
            "sender_mismatch": (
                (sender_balance_change - transactions["amount"]).abs()
                > BALANCE_TOLERANCE
            ),
            "recipient_mismatch": (
                (recipient_balance_change - transactions["amount"]).abs()
                > BALANCE_TOLERANCE
            ),
        },
        index=transactions.index,
    )


def update_statistics(
    statistics: dict[tuple[str, int], dict[str, int]],
    results: pd.DataFrame,
) -> None:
    """Accumulate mismatch counts by transaction type and fraud label."""

    grouped_results = results.groupby(
        ["type", "isFraud"],
        observed=True,
    ).agg(
        total=("isFraud", "size"),
        sender_mismatches=("sender_mismatch", "sum"),
        recipient_mismatches=("recipient_mismatch", "sum"),
    )

    for (transaction_type, fraud_label), row in grouped_results.iterrows():
        key = (transaction_type, int(fraud_label))

        if key not in statistics:
            continue

        statistics[key]["total"] += int(row["total"])
        statistics[key]["sender_mismatches"] += int(row["sender_mismatches"])
        statistics[key]["recipient_mismatches"] += int(row["recipient_mismatches"])


def create_report(
    statistics: dict[tuple[str, int], dict[str, int]],
) -> pd.DataFrame:
    """Convert accumulated statistics into a readable report."""

    report_rows = []

    for (transaction_type, fraud_label), values in statistics.items():
        total = values["total"]

        if total == 0:
            continue

        sender_count = values["sender_mismatches"]
        recipient_count = values["recipient_mismatches"]

        report_rows.append(
            {
                "transaction_type": transaction_type,
                "fraud_label": fraud_label,
                "total_transactions": total,
                "sender_mismatches": sender_count,
                "sender_mismatch_pct": (sender_count / total * 100),
                "recipient_mismatches": recipient_count,
                "recipient_mismatch_pct": (recipient_count / total * 100),
            }
        )

    return pd.DataFrame(report_rows)


def analyze_balance_by_transaction_type() -> None:
    """Analyze balance mismatches across transaction types."""

    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"PaySim dataset not found: {DATASET_PATH}")

    statistics = initialize_statistics()
    total_rows = 0

    logger.info("Starting balance analysis by transaction type.")
    logger.info("Dataset: %s", DATASET_PATH)

    for transactions in pd.read_csv(
        DATASET_PATH,
        chunksize=CHUNK_SIZE,
    ):
        total_rows += len(transactions)

        results = calculate_balance_mismatches(transactions)

        update_statistics(
            statistics,
            results,
        )

    if total_rows == 0:
        raise ValueError("The PaySim dataset contains no records.")

    report = create_report(statistics)

    if report.empty:
        raise ValueError("No transaction statistics were generated.")

    print("\n" + "=" * 90)
    print("PAYSIM BALANCE ANALYSIS BY TRANSACTION TYPE")
    print("=" * 90)

    print(f"\nTotal transactions: {total_rows:,}")
    print(f"Balance tolerance: {BALANCE_TOLERANCE:.2f}")

    print("\nBALANCE MISMATCH SUMMARY")
    print(
        report.to_string(
            index=False,
            formatters={
                "sender_mismatch_pct": "{:.2f}%".format,
                "recipient_mismatch_pct": "{:.2f}%".format,
            },
        )
    )

    print("\n" + "=" * 90)
    print("TRANSACTION TYPE ANALYSIS COMPLETED")
    print("=" * 90)

    logger.info(
        "Processed %s transactions.",
        f"{total_rows:,}",
    )


if __name__ == "__main__":
    analyze_balance_by_transaction_type()

"""Investigate balance consistency in the PaySim transaction dataset."""

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


def calculate_balance_mismatches(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate sender and recipient balance differences."""

    results = pd.DataFrame(index=transactions.index)

    results["fraud_label"] = transactions["isFraud"]

    sender_change = transactions["oldbalanceOrg"] - transactions["newbalanceOrig"]

    recipient_change = transactions["newbalanceDest"] - transactions["oldbalanceDest"]

    results["sender_difference"] = (sender_change - transactions["amount"]).abs()

    results["recipient_difference"] = (recipient_change - transactions["amount"]).abs()

    results["sender_mismatch"] = results["sender_difference"] > BALANCE_TOLERANCE

    results["recipient_mismatch"] = results["recipient_difference"] > BALANCE_TOLERANCE

    return results


def update_mismatch_statistics(
    statistics: dict[int, dict[str, float]],
    results: pd.DataFrame,
) -> None:
    """Accumulate balance mismatch statistics by fraud label."""

    for fraud_label in FRAUD_LABELS:
        fraud_transactions = results[results["fraud_label"] == fraud_label]

        statistics[fraud_label]["total"] += len(fraud_transactions)

        sender_mismatches = fraud_transactions[fraud_transactions["sender_mismatch"]]

        recipient_mismatches = fraud_transactions[
            fraud_transactions["recipient_mismatch"]
        ]

        statistics[fraud_label]["sender_count"] += len(sender_mismatches)

        statistics[fraud_label]["recipient_count"] += len(recipient_mismatches)

        statistics[fraud_label]["sender_difference_sum"] += sender_mismatches[
            "sender_difference"
        ].sum()

        statistics[fraud_label]["recipient_difference_sum"] += recipient_mismatches[
            "recipient_difference"
        ].sum()


def initialize_statistics() -> dict[int, dict[str, float]]:
    """Create empty statistics for both fraud labels."""

    return {
        label: {
            "total": 0,
            "sender_count": 0,
            "recipient_count": 0,
            "sender_difference_sum": 0.0,
            "recipient_difference_sum": 0.0,
        }
        for label in FRAUD_LABELS
    }


def collect_transaction_type_counts(
    counts: dict[str, int],
    transactions: pd.DataFrame,
) -> None:
    """Accumulate transaction counts by transaction type."""

    chunk_counts = transactions["type"].value_counts()

    for transaction_type, count in chunk_counts.items():
        counts[transaction_type] = counts.get(transaction_type, 0) + int(count)


def print_balance_report(
    total_rows: int,
    statistics: dict[int, dict[str, float]],
    transaction_type_counts: dict[str, int],
) -> None:
    """Display the balance consistency investigation results."""

    print("\n" + "=" * 65)
    print("PAYSIM BALANCE CONSISTENCY REPORT")
    print("=" * 65)

    print(f"\nTotal transactions: {total_rows:,}")
    print(f"Balance tolerance: {BALANCE_TOLERANCE:.2f}")

    print("\nSENDER BALANCE MISMATCHES")

    for label in FRAUD_LABELS:
        result = statistics[label]
        total = result["total"]
        count = result["sender_count"]
        percentage = count / total * 100 if total else 0.0

        print(f"Fraud={label}: {count:,.0f} ({percentage:.2f}%)")

    print("\nRECIPIENT BALANCE MISMATCHES")

    for label in FRAUD_LABELS:
        result = statistics[label]
        total = result["total"]
        count = result["recipient_count"]
        percentage = count / total * 100 if total else 0.0

        print(f"Fraud={label}: {count:,.0f} ({percentage:.2f}%)")

    print("\nMISMATCH DIFFERENCE TOTALS")

    for label in FRAUD_LABELS:
        result = statistics[label]

        print(
            f"Fraud={label}: "
            f"Sender={result['sender_difference_sum']:,.2f}, "
            f"Recipient={result['recipient_difference_sum']:,.2f}"
        )

    print("\nTRANSACTION TYPE COUNTS")

    for transaction_type, count in sorted(transaction_type_counts.items()):
        print(f"{transaction_type}: {count:,}")

    print("\n" + "=" * 65)
    print("BALANCE CONSISTENCY INVESTIGATION COMPLETED")
    print("=" * 65)


def check_balance_consistency() -> None:
    """Process the dataset and report balance consistency findings."""

    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"PaySim dataset not found: {DATASET_PATH}")

    total_rows = 0
    statistics = initialize_statistics()
    transaction_type_counts: dict[str, int] = {}

    logger.info("Starting balance consistency investigation.")
    logger.info("Dataset: %s", DATASET_PATH)

    for transactions in pd.read_csv(
        DATASET_PATH,
        chunksize=CHUNK_SIZE,
    ):
        total_rows += len(transactions)

        results = calculate_balance_mismatches(transactions)

        update_mismatch_statistics(
            statistics,
            results,
        )

        collect_transaction_type_counts(
            transaction_type_counts,
            transactions,
        )

    if total_rows == 0:
        raise ValueError("The PaySim dataset contains no records.")

    print_balance_report(
        total_rows,
        statistics,
        transaction_type_counts,
    )

    logger.info(
        "Balance consistency investigation completed. Processed %s transactions.",
        f"{total_rows:,}",
    )


if __name__ == "__main__":
    check_balance_consistency()

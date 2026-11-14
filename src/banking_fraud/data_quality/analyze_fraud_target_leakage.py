"""Investigate potential target leakage in the PaySim dataset."""

import logging
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "paysim.csv"

CHUNK_SIZE = 250_000
BALANCE_TOLERANCE = 0.01

REQUIRED_COLUMNS = [
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]

MONEY_COLUMNS = [
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


def calculate_percentage(part: int, total: int) -> float:
    """Calculate a percentage safely."""
    if total == 0:
        return 0.0

    return part / total * 100


def validate_dataset() -> None:
    """Check that the required columns exist."""
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    columns = pd.read_csv(DATASET_PATH, nrows=0).columns
    missing_columns = set(REQUIRED_COLUMNS) - set(columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")


def analyze_chunks() -> tuple[pd.DataFrame, dict[str, int]]:
    """Analyse balance patterns and fraud flags in chunks."""

    results = []
    totals = {
        "transactions": 0,
        "fraud": 0,
        "flagged": 0,
        "fraud_and_flagged": 0,
    }

    for chunk in pd.read_csv(
        DATASET_PATH,
        usecols=REQUIRED_COLUMNS,
        chunksize=CHUNK_SIZE,
    ):
        if chunk[REQUIRED_COLUMNS].isna().any().any():
            raise ValueError("Missing values found in the dataset.")

        if not chunk["isFraud"].isin([0, 1]).all():
            raise ValueError("Invalid fraud labels found.")

        if not chunk["isFlaggedFraud"].isin([0, 1]).all():
            raise ValueError("Invalid simulator flags found.")

        if (chunk[MONEY_COLUMNS] < 0).any().any():
            raise ValueError("Negative monetary values found.")

        # Identify post-transaction balance patterns.
        chunk["recipient_unchanged"] = (
            chunk["newbalanceDest"] - chunk["oldbalanceDest"]
        ).abs() <= BALANCE_TOLERANCE

        chunk["sender_unchanged"] = (
            chunk["oldbalanceOrg"] - chunk["newbalanceOrig"]
        ).abs() <= BALANCE_TOLERANCE

        chunk["recipient_zero_after"] = chunk["newbalanceDest"] == 0

        # Summarise patterns by transaction type and fraud label.
        summary = (
            chunk.groupby(["type", "isFraud"])
            .agg(
                total=("isFraud", "size"),
                recipient_unchanged=("recipient_unchanged", "sum"),
                recipient_zero_after=("recipient_zero_after", "sum"),
                sender_unchanged=("sender_unchanged", "sum"),
            )
            .reset_index()
        )

        results.append(summary)

        totals["transactions"] += len(chunk)
        totals["fraud"] += int(chunk["isFraud"].sum())
        totals["flagged"] += int(chunk["isFlaggedFraud"].sum())
        totals["fraud_and_flagged"] += int(
            ((chunk["isFraud"] == 1) & (chunk["isFlaggedFraud"] == 1)).sum()
        )

    if not results:
        raise ValueError("The dataset contains no transactions.")

    # Combine summaries from all chunks.
    balance_summary = (
        pd.concat(results).groupby(["type", "isFraud"]).sum().reset_index()
    )

    return balance_summary, totals


def print_balance_analysis(summary: pd.DataFrame) -> None:
    """Display balance patterns for fraud and non-fraud transactions."""

    print("\nPOST-TRANSACTION BALANCE PATTERNS")
    print("=" * 90)

    rows = []

    for transaction_type in sorted(summary["type"].unique()):
        group = summary[summary["type"] == transaction_type]

        fraud = group[group["isFraud"] == 1]
        non_fraud = group[group["isFraud"] == 0]

        fraud_count = int(fraud["total"].sum())
        non_fraud_count = int(non_fraud["total"].sum())

        fraud = fraud.iloc[0] if not fraud.empty else None
        non_fraud = non_fraud.iloc[0] if not non_fraud.empty else None

        rows.append(
            {
                "type": transaction_type,
                "fraud_count": fraud_count,
                "recipient_unchanged_fraud_pct": calculate_percentage(
                    int(fraud["recipient_unchanged"]) if fraud is not None else 0,
                    fraud_count,
                ),
                "recipient_unchanged_non_fraud_pct": calculate_percentage(
                    int(non_fraud["recipient_unchanged"])
                    if non_fraud is not None
                    else 0,
                    non_fraud_count,
                ),
                "recipient_zero_after_fraud_pct": calculate_percentage(
                    int(fraud["recipient_zero_after"]) if fraud is not None else 0,
                    fraud_count,
                ),
                "recipient_zero_after_non_fraud_pct": calculate_percentage(
                    int(non_fraud["recipient_zero_after"])
                    if non_fraud is not None
                    else 0,
                    non_fraud_count,
                ),
                "sender_unchanged_fraud_pct": calculate_percentage(
                    int(fraud["sender_unchanged"]) if fraud is not None else 0,
                    fraud_count,
                ),
                "sender_unchanged_non_fraud_pct": calculate_percentage(
                    int(non_fraud["sender_unchanged"]) if non_fraud is not None else 0,
                    non_fraud_count,
                ),
            }
        )

    report = pd.DataFrame(rows)

    percentage_columns = [
        column for column in report.columns if column.endswith("_pct")
    ]

    print(
        report.to_string(
            index=False,
            formatters={column: "{:.2f}%".format for column in percentage_columns},
        )
    )


def print_flagged_fraud_analysis(totals: dict[str, int]) -> None:
    """Compare simulator flags against fraud labels."""

    total = totals["transactions"]
    fraud = totals["fraud"]
    flagged = totals["flagged"]
    true_positives = totals["fraud_and_flagged"]

    false_positives = flagged - true_positives
    false_negatives = fraud - true_positives
    true_negatives = total - fraud - false_positives

    precision = calculate_percentage(true_positives, flagged)
    recall = calculate_percentage(true_positives, fraud)

    print("\nSIMULATOR FRAUD FLAG ANALYSIS")
    print("=" * 90)

    print(f"Total transactions: {total:,}")
    print(f"Fraud transactions: {fraud:,}")
    print(f"Flagged transactions: {flagged:,}")

    print("\nConfusion matrix:")
    print(f"True positives: {true_positives:,}")
    print(f"False positives: {false_positives:,}")
    print(f"False negatives: {false_negatives:,}")
    print(f"True negatives: {true_negatives:,}")

    print(f"\nPrecision: {precision:.2f}%")
    print(f"Recall: {recall:.2f}%")


def print_feature_availability() -> None:
    """Display proposed pre-transaction feature availability."""

    availability = pd.DataFrame(
        [
            ("type", "Include", "Transaction type"),
            ("amount", "Include", "Proposed transaction amount"),
            ("oldbalanceOrg", "Include", "Previous sender balance"),
            ("oldbalanceDest", "Conditional", "Previous recipient balance"),
            ("newbalanceOrig", "Exclude", "Post-transaction sender balance"),
            ("newbalanceDest", "Exclude", "Post-transaction recipient balance"),
            ("isFlaggedFraud", "Exclude", "Simulator-generated flag"),
            ("nameOrig", "Exclude", "Sender identifier"),
            ("nameDest", "Exclude", "Recipient identifier"),
            ("isFraud", "Target only", "Fraud label"),
            ("step", "Pending", "Simulation time"),
        ],
        columns=["column", "decision", "reason"],
    )

    print("\nPRE-TRANSACTION FEATURE AVAILABILITY")
    print("=" * 90)
    print(availability.to_string(index=False))


def main() -> None:
    """Run the target leakage investigation."""

    logger.info("Starting target leakage investigation.")

    validate_dataset()

    summary, totals = analyze_chunks()

    print_balance_analysis(summary)
    print_flagged_fraud_analysis(totals)
    print_feature_availability()

    logger.info(
        "Investigation completed. Processed %s transactions.",
        f"{totals['transactions']:,}",
    )


if __name__ == "__main__":
    main()

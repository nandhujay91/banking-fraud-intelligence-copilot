"""Analyze fraud patterns by transaction type and simulation period."""

import logging
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "paysim.csv"

CHUNK_SIZE = 250_000
PERIOD_SIZE = 50

REQUIRED_COLUMNS = {"step", "type", "isFraud"}

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)


def update_statistics(
    statistics: dict[tuple[int, str], dict[str, int]],
    transactions: pd.DataFrame,
) -> None:
    """Accumulate transaction and fraud counts by period and type."""

    period_starts = ((transactions["step"] - 1) // PERIOD_SIZE) * PERIOD_SIZE + 1

    period_data = transactions.assign(period_start=period_starts)

    grouped_data = period_data.groupby(["period_start", "type"]).agg(
        total_transactions=("isFraud", "size"),
        fraud_transactions=("isFraud", "sum"),
    )

    for (period_start, transaction_type), row in grouped_data.iterrows():
        key = (int(period_start), str(transaction_type))

        if key not in statistics:
            statistics[key] = {
                "total_transactions": 0,
                "fraud_transactions": 0,
            }

        statistics[key]["total_transactions"] += int(row["total_transactions"])
        statistics[key]["fraud_transactions"] += int(row["fraud_transactions"])


def create_report(
    statistics: dict[tuple[int, str], dict[str, int]],
) -> pd.DataFrame:
    """Create fraud statistics for each period and transaction type."""

    report_rows = []

    for (period_start, transaction_type), values in sorted(statistics.items()):
        total = values["total_transactions"]
        fraud_count = values["fraud_transactions"]

        report_rows.append(
            {
                "simulation_period": (
                    f"{period_start}-{period_start + PERIOD_SIZE - 1}"
                ),
                "transaction_type": transaction_type,
                "total_transactions": total,
                "fraud_transactions": fraud_count,
                "non_fraud_transactions": total - fraud_count,
                "fraud_rate_pct": fraud_count / total * 100,
            }
        )

    return pd.DataFrame(report_rows)


def print_report(report: pd.DataFrame) -> None:
    """Display fraud statistics by period and transaction type."""

    print("\n" + "=" * 110)
    print("PAYSIM FRAUD PATTERNS BY TYPE AND SIMULATION PERIOD")
    print("=" * 110)

    print("\nFRAUD RATE BY TRANSACTION TYPE AND PERIOD")

    print(
        report.to_string(
            index=False,
            formatters={
                "fraud_rate_pct": "{:.4f}%".format,
            },
        )
    )

    type_summary = (
        report.groupby("transaction_type")
        .agg(
            total_transactions=("total_transactions", "sum"),
            fraud_transactions=("fraud_transactions", "sum"),
        )
        .reset_index()
    )

    type_summary["fraud_rate_pct"] = (
        type_summary["fraud_transactions"] / type_summary["total_transactions"] * 100
    )

    print("\nOVERALL FRAUD RATE BY TRANSACTION TYPE")

    print(
        type_summary.to_string(
            index=False,
            formatters={
                "fraud_rate_pct": "{:.4f}%".format,
            },
        )
    )

    print("\n" + "=" * 110)
    print("FRAUD PATTERN ANALYSIS COMPLETED")
    print("=" * 110)


def analyze_fraud_patterns() -> None:
    """Process PaySim data and analyse fraud by type and period."""

    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"PaySim dataset not found: {DATASET_PATH}")

    statistics: dict[tuple[int, str], dict[str, int]] = {}
    total_rows = 0

    logger.info("Starting fraud pattern analysis.")
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

        if transactions[["step", "type", "isFraud"]].isna().any().any():
            raise ValueError(f"Missing required values found in chunk {chunk_number}.")

        if (transactions["step"] < 1).any():
            raise ValueError(f"Invalid simulation step found in chunk {chunk_number}.")

        if not transactions["isFraud"].isin({0, 1}).all():
            raise ValueError(f"Invalid fraud labels found in chunk {chunk_number}.")

        total_rows += len(transactions)

        update_statistics(statistics, transactions)

        if chunk_number % 5 == 0:
            logger.info(
                "Processed %s transactions.",
                f"{total_rows:,}",
            )

    if total_rows == 0:
        raise ValueError("The PaySim dataset contains no records.")

    report = create_report(statistics)

    if report.empty:
        raise ValueError("No fraud statistics were generated.")

    print_report(report)

    logger.info(
        "Processed %s transactions across %s period/type combinations.",
        f"{total_rows:,}",
        len(report),
    )


if __name__ == "__main__":
    analyze_fraud_patterns()

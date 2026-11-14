"""Analyze fraud rates across simulation periods in the PaySim dataset."""

import logging
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "paysim.csv"

CHUNK_SIZE = 250_000
PERIOD_SIZE = 50

REQUIRED_COLUMNS = {"step", "isFraud"}

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)


def initialize_statistics() -> dict[int, dict[str, int]]:
    """Initialize transaction counters for simulation periods."""

    return {}


def update_period_statistics(
    statistics: dict[int, dict[str, int]],
    transactions: pd.DataFrame,
) -> None:
    """Accumulate transaction and fraud counts by simulation period."""

    period_starts = ((transactions["step"] - 1) // PERIOD_SIZE) * PERIOD_SIZE + 1

    period_data = transactions.assign(period_start=period_starts)

    grouped_data = period_data.groupby("period_start").agg(
        total_transactions=("isFraud", "size"),
        fraud_transactions=("isFraud", "sum"),
    )

    for period_start, row in grouped_data.iterrows():
        period = int(period_start)

        if period not in statistics:
            statistics[period] = {
                "total_transactions": 0,
                "fraud_transactions": 0,
            }

        statistics[period]["total_transactions"] += int(row["total_transactions"])

        statistics[period]["fraud_transactions"] += int(row["fraud_transactions"])


def create_period_report(
    statistics: dict[int, dict[str, int]],
) -> pd.DataFrame:
    """Calculate fraud rates for each simulation period."""

    report_rows = []

    for period_start, values in sorted(statistics.items()):
        total = values["total_transactions"]
        fraud_count = values["fraud_transactions"]

        fraud_rate = fraud_count / total * 100 if total else 0.0

        report_rows.append(
            {
                "simulation_period": (
                    f"{period_start}-{period_start + PERIOD_SIZE - 1}"
                ),
                "total_transactions": total,
                "fraud_transactions": fraud_count,
                "non_fraud_transactions": total - fraud_count,
                "fraud_rate_pct": fraud_rate,
            }
        )

    return pd.DataFrame(report_rows)


def print_report(report: pd.DataFrame) -> None:
    """Display fraud rates across simulation periods."""

    total_transactions = report["total_transactions"].sum()
    total_fraud = report["fraud_transactions"].sum()

    overall_fraud_rate = (
        total_fraud / total_transactions * 100 if total_transactions else 0.0
    )

    print("\n" + "=" * 90)
    print("PAYSIM FRAUD RATE BY SIMULATION PERIOD")
    print("=" * 90)

    print(f"\nTotal transactions: {total_transactions:,}")
    print(f"Total fraud transactions: {total_fraud:,}")
    print(f"Overall fraud rate: {overall_fraud_rate:.4f}%")

    print("\nPERIOD-LEVEL FRAUD DISTRIBUTION")

    print(
        report.to_string(
            index=False,
            formatters={
                "fraud_rate_pct": "{:.4f}%".format,
            },
        )
    )

    highest_rate_period = report.loc[report["fraud_rate_pct"].idxmax()]

    lowest_rate_period = report.loc[report["fraud_rate_pct"].idxmin()]

    print("\nFRAUD RATE COMPARISON")

    print(
        "Highest fraud rate: "
        f"{highest_rate_period['simulation_period']} "
        f"({highest_rate_period['fraud_rate_pct']:.4f}%)"
    )

    print(
        "Lowest fraud rate: "
        f"{lowest_rate_period['simulation_period']} "
        f"({lowest_rate_period['fraud_rate_pct']:.4f}%)"
    )

    print("\n" + "=" * 90)
    print("FRAUD RATE ANALYSIS COMPLETED")
    print("=" * 90)


def analyze_fraud_rate_by_period() -> None:
    """Process PaySim data and calculate period-level fraud rates."""

    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"PaySim dataset not found: {DATASET_PATH}")

    statistics = initialize_statistics()
    total_rows = 0

    logger.info("Starting fraud rate analysis by simulation period.")
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

            if not transactions["isFraud"].isin({0, 1}).all():
                raise ValueError("Fraud labels must contain only 0 and 1.")

        total_rows += len(transactions)

        update_period_statistics(
            statistics,
            transactions,
        )

    if total_rows == 0:
        raise ValueError("The PaySim dataset contains no records.")

    report = create_period_report(statistics)

    if report.empty:
        raise ValueError("No simulation period statistics were generated.")

    print_report(report)

    logger.info(
        "Processed %s transactions across %s simulation periods.",
        f"{total_rows:,}",
        len(report),
    )


if __name__ == "__main__":
    analyze_fraud_rate_by_period()

"""Analyze transaction amounts in the PaySim dataset."""

import logging
import tempfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "paysim.csv"

CHUNK_SIZE = 250_000
FRAUD_LABELS = (0, 1)

REQUIRED_COLUMNS = {
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
    """Initialize transaction amount statistics."""

    return {
        label: {
            "count": 0,
            "amount_sum": 0.0,
            "minimum": float("inf"),
            "maximum": float("-inf"),
            "zero_count": 0,
            "negative_count": 0,
        }
        for label in FRAUD_LABELS
    }


def update_statistics(
    statistics: dict,
    amounts: pd.Series,
) -> None:
    """Update descriptive statistics for a group of amounts."""

    if amounts.empty:
        return

    statistics["count"] += len(amounts)
    statistics["amount_sum"] += float(amounts.sum())
    statistics["minimum"] = min(
        statistics["minimum"],
        float(amounts.min()),
    )
    statistics["maximum"] = max(
        statistics["maximum"],
        float(amounts.max()),
    )
    statistics["zero_count"] += int((amounts == 0).sum())
    statistics["negative_count"] += int((amounts < 0).sum())


def create_amount_report(
    statistics: dict,
    percentile_file: Path,
) -> pd.DataFrame:
    """Calculate exact amount percentiles using disk-backed data."""

    report_rows = []

    for fraud_label in FRAUD_LABELS:
        result = statistics[fraud_label]

        if result["count"] == 0:
            continue

        amount_table = pq.read_table(
            percentile_file,
            filters=[("isFraud", "=", fraud_label)],
            columns=["amount"],
        )

        amounts = amount_table.column("amount").to_pandas()

        report_rows.append(
            {
                "fraud_label": fraud_label,
                "transaction_count": result["count"],
                "minimum": result["minimum"],
                "q1": amounts.quantile(0.25),
                "median": amounts.median(),
                "mean": result["amount_sum"] / result["count"],
                "q3": amounts.quantile(0.75),
                "maximum": result["maximum"],
                "zero_amounts": result["zero_count"],
                "negative_amounts": result["negative_count"],
            }
        )

    return pd.DataFrame(report_rows)


def analyze_transaction_amounts() -> None:
    """Process transaction amounts and generate descriptive reports."""

    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"PaySim dataset not found: {DATASET_PATH}")

    statistics = initialize_statistics()
    type_statistics: dict[tuple[str, int], dict] = {}
    total_rows = 0

    logger.info("Starting transaction amount analysis.")
    logger.info("Dataset: %s", DATASET_PATH)

    with tempfile.TemporaryDirectory(
        prefix="paysim_amount_analysis_"
    ) as temporary_directory:
        percentile_file = Path(temporary_directory) / "amounts.parquet"

        parquet_writer = None

        try:
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
                            "Dataset is missing required columns: "
                            f"{sorted(missing_columns)}"
                        )

                total_rows += len(transactions)

                for fraud_label in FRAUD_LABELS:
                    amounts = transactions.loc[
                        transactions["isFraud"] == fraud_label,
                        "amount",
                    ]

                    update_statistics(
                        statistics[fraud_label],
                        amounts,
                    )

                grouped_transactions = transactions.groupby(
                    ["type", "isFraud"],
                    observed=True,
                )

                for (transaction_type, fraud_label), group in grouped_transactions:
                    key = (transaction_type, int(fraud_label))

                    if key not in type_statistics:
                        type_statistics[key] = {
                            "count": 0,
                            "amount_sum": 0.0,
                            "minimum": float("inf"),
                            "maximum": float("-inf"),
                        }

                    result = type_statistics[key]
                    amounts = group["amount"]

                    result["count"] += len(amounts)
                    result["amount_sum"] += float(amounts.sum())
                    result["minimum"] = min(
                        result["minimum"],
                        float(amounts.min()),
                    )
                    result["maximum"] = max(
                        result["maximum"],
                        float(amounts.max()),
                    )

                amount_table = pa.Table.from_pandas(
                    transactions[["isFraud", "amount"]],
                    preserve_index=False,
                )

                if parquet_writer is None:
                    parquet_writer = pq.ParquetWriter(
                        percentile_file,
                        amount_table.schema,
                        compression="snappy",
                    )

                parquet_writer.write_table(amount_table)

        finally:
            if parquet_writer is not None:
                parquet_writer.close()

        if total_rows == 0:
            raise ValueError("The PaySim dataset contains no records.")

        amount_report = create_amount_report(
            statistics,
            percentile_file,
        )

    type_report_rows = []

    for (transaction_type, fraud_label), result in sorted(type_statistics.items()):
        type_report_rows.append(
            {
                "transaction_type": transaction_type,
                "fraud_label": fraud_label,
                "transaction_count": result["count"],
                "average_amount": (result["amount_sum"] / result["count"]),
                "minimum_amount": result["minimum"],
                "maximum_amount": result["maximum"],
            }
        )

    type_report = pd.DataFrame(type_report_rows)

    print("\n" + "=" * 100)
    print("PAYSIM TRANSACTION AMOUNT ANALYSIS")
    print("=" * 100)

    print(f"\nTotal transactions: {total_rows:,}")

    print("\nAMOUNT DISTRIBUTION BY FRAUD LABEL")

    print(
        amount_report.to_string(
            index=False,
            formatters={
                column: "{:,.2f}".format
                for column in (
                    "minimum",
                    "q1",
                    "median",
                    "mean",
                    "q3",
                    "maximum",
                )
            },
        )
    )

    print("\nAMOUNT DISTRIBUTION BY TRANSACTION TYPE")

    print(
        type_report.to_string(
            index=False,
            formatters={
                column: "{:,.2f}".format
                for column in (
                    "average_amount",
                    "minimum_amount",
                    "maximum_amount",
                )
            },
        )
    )

    print("\n" + "=" * 100)
    print("TRANSACTION AMOUNT ANALYSIS COMPLETED")
    print("=" * 100)

    logger.info("Processed %s transactions.", f"{total_rows:,}")


if __name__ == "__main__":
    analyze_transaction_amounts()

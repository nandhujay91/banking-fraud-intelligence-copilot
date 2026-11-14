"""Load PaySim transactions into PostgreSQL."""

import logging
import os
from pathlib import Path

import pandas as pd
import psycopg

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CSV_PATH = PROJECT_ROOT / "data" / "raw" / "paysim.csv"

DB_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "dbname": "banking_fraud_db",
    "user": "postgres",
    "password": os.environ["DB_PASSWORD"],
}

EXPECTED_ROWS = 6_362_620
CHUNK_SIZE = 100_000

COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]

COPY_SQL = """
    COPY transactions (
        step,
        type,
        amount,
        name_orig,
        old_balance_org,
        new_balance_orig,
        name_dest,
        old_balance_dest,
        new_balance_dest,
        is_fraud,
        is_flagged_fraud
    )
    FROM STDIN
"""

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)


def load_transactions() -> None:
    """Load the complete CSV dataset in a single database transaction."""

    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {CSV_PATH}")

    loaded_rows = 0

    logger.info("Starting PaySim transaction ingestion.")

    with (
        psycopg.connect(**DB_CONFIG) as connection,
        connection.cursor() as cursor,
    ):
        cursor.execute("SELECT COUNT(*) FROM transactions")
        existing_rows = cursor.fetchone()[0]

        if existing_rows != 0:
            raise ValueError(f"Target table is not empty: {existing_rows:,} rows.")

        for chunk in pd.read_csv(
            CSV_PATH,
            usecols=COLUMNS,
            chunksize=CHUNK_SIZE,
        ):
            chunk = chunk[COLUMNS]

            with cursor.copy(COPY_SQL) as copy:
                for row in chunk.itertuples(index=False, name=None):
                    copy.write_row(row)

            loaded_rows += len(chunk)

            logger.info("Loaded rows: %s", f"{loaded_rows:,}")

        cursor.execute("SELECT COUNT(*) FROM transactions")
        database_rows = cursor.fetchone()[0]

        if loaded_rows != EXPECTED_ROWS:
            raise ValueError(f"Unexpected CSV row count: {loaded_rows:,}")

        if database_rows != EXPECTED_ROWS:
            raise ValueError(f"Unexpected database row count: {database_rows:,}")

    logger.info("Transaction ingestion completed successfully.")
    logger.info("Total database rows: %s", f"{loaded_rows:,}")


if __name__ == "__main__":
    load_transactions()

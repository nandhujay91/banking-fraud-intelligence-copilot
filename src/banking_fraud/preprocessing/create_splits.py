"""Create chronological training and testing datasets."""

import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "fraud_features.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "splits"

TRAIN_PATH = OUTPUT_DIR / "train.parquet"
TEST_PATH = OUTPUT_DIR / "test.parquet"

TRAIN_END_STEP = 520
BATCH_SIZE = 250_000

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)


def create_splits() -> None:
    """Split transactions chronologically using the step column."""

    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {INPUT_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parquet_file = pq.ParquetFile(INPUT_PATH)

    train_writer = None
    test_writer = None

    train_rows = 0
    test_rows = 0

    try:
        for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE):
            data = batch.to_pandas()

            train_data = data[data["step"] <= TRAIN_END_STEP]
            test_data = data[data["step"] > TRAIN_END_STEP]

            if not train_data.empty:
                table = pa.Table.from_pandas(
                    train_data,
                    preserve_index=False,
                )

                if train_writer is None:
                    train_writer = pq.ParquetWriter(
                        TRAIN_PATH,
                        table.schema,
                        compression="snappy",
                    )

                train_writer.write_table(table)
                train_rows += len(train_data)

            if not test_data.empty:
                table = pa.Table.from_pandas(
                    test_data,
                    preserve_index=False,
                )

                if test_writer is None:
                    test_writer = pq.ParquetWriter(
                        TEST_PATH,
                        table.schema,
                        compression="snappy",
                    )

                test_writer.write_table(table)
                test_rows += len(test_data)

    except Exception:
        for writer in (train_writer, test_writer):
            if writer is not None:
                writer.close()

        for path in (TRAIN_PATH, TEST_PATH):
            if path.exists():
                path.unlink()

        raise

    else:
        for writer in (train_writer, test_writer):
            if writer is not None:
                writer.close()

    if train_rows + test_rows != parquet_file.metadata.num_rows:
        raise ValueError("Split row counts do not match the source dataset.")

    logger.info("Chronological split completed.")
    logger.info("Training rows: %s", f"{train_rows:,}")
    logger.info("Testing rows: %s", f"{test_rows:,}")
    logger.info("Training steps: 1–%s", TRAIN_END_STEP)
    logger.info("Testing steps: %s–743", TRAIN_END_STEP + 1)
    logger.info("Training file: %s", TRAIN_PATH)
    logger.info("Testing file: %s", TEST_PATH)


if __name__ == "__main__":
    create_splits()

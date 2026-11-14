
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]

SPLITS = {
    "train": ROOT_DIR / "data" / "validation" / "train.parquet",
    "validation": ROOT_DIR / "data" / "validation" / "validation.parquet",
    "test": ROOT_DIR / "data" / "splits" / "test.parquet",
}

FEATURE_COLUMNS = ["type", "amount", "oldbalanceOrg", "log_amount"]
TARGET_COLUMN = "isFraud"


def load_fingerprints(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(
        path,
        columns=FEATURE_COLUMNS + [TARGET_COLUMN],
    )

    fingerprints = pd.util.hash_pandas_object(
        df[FEATURE_COLUMNS],
        index=False,
    )

    return pd.DataFrame(
        {
            "fingerprint": fingerprints.to_numpy(),
            "target": df[TARGET_COLUMN].to_numpy(),
        }
    )


def main() -> None:
    split_data = {}

    for name, path in SPLITS.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")

        split_data[name] = load_fingerprints(path)
        print(f"{name}: {len(split_data[name]):,} rows loaded")

    names = list(split_data)

    print("\nCROSS-SPLIT DUPLICATE CHECK")
    print("=" * 55)

    for i, first_name in enumerate(names):
        for second_name in names[i + 1:]:
            first = split_data[first_name]
            second = split_data[second_name]

            first_hashes = first["fingerprint"].to_numpy()
            second_hashes = second["fingerprint"].to_numpy()

            common_hashes = np.intersect1d(
                first_hashes,
                second_hashes,
            )

            first_matches = np.isin(first_hashes, common_hashes)
            second_matches = np.isin(second_hashes, common_hashes)

            print(f"\n{first_name.upper()} vs {second_name.upper()}")
            print(
                f"Shared unique feature combinations: "
                f"{len(common_hashes):,}"
            )

            print(f"{first_name} matching rows: {first_matches.sum():,}")
            print(
                f"{first_name} matching fraud rows: "
                f"{first.loc[first_matches, 'target'].sum():,}"
            )

            print(f"{second_name} matching rows: {second_matches.sum():,}")
            print(
                f"{second_name} matching fraud rows: "
                f"{second.loc[second_matches, 'target'].sum():,}"
            )

            if len(common_hashes) > 0:
                first_labels = first.loc[
                    first_matches,
                    ["fingerprint", "target"],
                ].drop_duplicates()

                second_labels = second.loc[
                    second_matches,
                    ["fingerprint", "target"],
                ].drop_duplicates()

                combined = pd.concat(
                    [first_labels, second_labels],
                    ignore_index=True,
                )

                conflicting = (
                    combined.groupby("fingerprint")["target"]
                    .nunique()
                    .gt(1)
                    .sum()
                )

                print(
                    "Shared combinations with conflicting labels: "
                    f"{conflicting:,}"
                )

    print("\nDuplicate check completed.")
    print("No source data was modified.")


if __name__ == "__main__":
    main()

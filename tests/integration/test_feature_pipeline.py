"""Integration tests for the feature engineering pipeline."""

import pandas as pd
import pyarrow.parquet as pq
import pytest

from banking_fraud.features import build_features as pipeline


def test_feature_pipeline(tmp_path, monkeypatch):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.parquet"

    data = pd.DataFrame(
        {
            "step": [1, 2, 3],
            "type": ["TRANSFER", "CASH_OUT", "PAYMENT"],
            "amount": [1000.0, 500.0, 0.0],
            "oldbalanceOrg": [2000.0, 1000.0, 1500.0],
            "isFraud": [1, 0, 0],
        }
    )

    data.to_csv(input_path, index=False)

    monkeypatch.setattr(pipeline, "INPUT_PATH", input_path)
    monkeypatch.setattr(pipeline, "OUTPUT_PATH", output_path)
    monkeypatch.setattr(pipeline, "CHUNK_SIZE", 2)

    pipeline.main()

    assert output_path.exists()

    result = pq.read_table(output_path).to_pandas()

    assert len(result) == 3
    assert list(result.columns) == [
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "log_amount",
        "isFraud",
    ]
    assert result["step"].tolist() == [1, 2, 3]
    assert result["isFraud"].sum() == 1


def test_pipeline_removes_partial_output(tmp_path, monkeypatch):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.parquet"

    data = pd.DataFrame(
        {
            "step": [1, 2, 3],
            "type": ["TRANSFER", "CASH_OUT", "PAYMENT"],
            "amount": [1000.0, 500.0, 200.0],
            "oldbalanceOrg": [2000.0, 1000.0, 1500.0],
            "isFraud": [1, 0, 0],
        }
    )

    data.to_csv(input_path, index=False)

    monkeypatch.setattr(pipeline, "INPUT_PATH", input_path)
    monkeypatch.setattr(pipeline, "OUTPUT_PATH", output_path)
    monkeypatch.setattr(pipeline, "CHUNK_SIZE", 2)

    def fail_on_processing(data):
        raise RuntimeError("Simulated processing failure")

    monkeypatch.setattr(pipeline, "build_features", fail_on_processing)

    with pytest.raises(RuntimeError, match="Simulated processing failure"):
        pipeline.main()

    assert not output_path.exists()

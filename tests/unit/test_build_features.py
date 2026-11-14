"""Tests for the PaySim feature engineering pipeline."""

import numpy as np
import pandas as pd
import pytest

from banking_fraud.features.build_features import (
    build_features,
    validate_data,
)


def sample_data():
    return pd.DataFrame(
        {
            "step": [1, 2, 3],
            "type": ["TRANSFER", "CASH_OUT", "PAYMENT"],
            "amount": [1000.0, 500.0, 0.0],
            "oldbalanceOrg": [2000.0, 0.0, 1500.0],
            "isFraud": [1, 0, 0],
        }
    )


def test_build_features():
    data = sample_data()

    features = build_features(data)

    assert len(features) == 3
    assert list(features.columns) == [
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "log_amount",
        "isFraud",
    ]
    assert np.allclose(
        features["log_amount"],
        np.log1p(data["amount"]),
    )
    assert features["step"].tolist() == [1, 2, 3]


def test_zero_amount():
    data = sample_data()

    features = build_features(data)

    assert features.loc[2, "log_amount"] == 0


def test_missing_values():
    data = sample_data()
    data.loc[0, "amount"] = np.nan

    with pytest.raises(ValueError):
        validate_data(data)


def test_negative_amount():
    data = sample_data()
    data.loc[0, "amount"] = -100

    with pytest.raises(ValueError):
        validate_data(data)


def test_invalid_fraud_label():
    data = sample_data()
    data.loc[0, "isFraud"] = 2

    with pytest.raises(ValueError):
        validate_data(data)


def test_invalid_transaction_type():
    data = sample_data()
    data.loc[0, "type"] = "UNKNOWN"

    with pytest.raises(ValueError):
        validate_data(data)


def test_infinite_amount():
    data = sample_data()
    data.loc[0, "amount"] = np.inf

    with pytest.raises(ValueError):
        build_features(data)


def test_invalid_step():
    data = sample_data()
    data.loc[0, "step"] = 0

    with pytest.raises(ValueError):
        validate_data(data)

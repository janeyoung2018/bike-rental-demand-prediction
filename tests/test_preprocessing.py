# test_preprocessing.py

import pandas as pd
import pytest
from src.preprocessing import fill_missing_values, preprocess_data
from src.utils.config import Params as params


@pytest.fixture
def sample_df():
    data = {
        "dteday": [
            "2023-01-01",
            "2023-01-01",
            "2023-01-01",
            "2023-01-02",
            "2023-01-02",
            "2023-01-02",
        ],
        "hr": [0, 1, 3, 0, 2, 3],
        "cnt": [10, 20, 30, 40, 50, 60],
        "temp": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "weathersit": [1, 2, 1, 3, 2, 3],
        "atemp": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    }
    return pd.DataFrame(data)


def test_fill_missing_values(sample_df):
    processed_df = fill_missing_values(sample_df)

    # Check if the datetime index is complete
    expected_index = pd.date_range(
        start="2023-01-01 00:00:00", end="2023-01-02 03:00:00", freq="h"
    )
    assert all(processed_df["datetime"] == expected_index)

    # Check if missing cnt values are filled with 0
    assert all(processed_df["cnt"].fillna(-1).astype(int) >= 0)

    # Check if day static columns are filled correctly
    assert all(processed_df["temp"].notna())

    # Check if weather columns are filled correctly
    assert all(processed_df["weathersit"].notna())

    # Check if weathersit is integer
    assert all(isinstance(x, int) for x in processed_df["weathersit"].tolist())

    # Check if base columns are in the correct order
    base_cols = params.base_columns
    assert list(processed_df.columns[: len(base_cols)]) == base_cols


def test_preprocess_data(sample_df):
    processed_df = preprocess_data(sample_df)

    # Check if the datetime index is complete
    expected_index = pd.date_range(
        start="2023-01-01 00:00:00", end="2023-01-02 03:00:00", freq="h"
    )
    assert all(processed_df["datetime"] == expected_index)

    # Check if missing cnt values are filled with 0
    assert all(processed_df["cnt"].fillna(-1).astype(int) >= 0)

    # Check if day static columns are filled correctly
    assert all(processed_df["temp"].notna())

    # Check if weather columns are filled correctly
    assert all(processed_df["weathersit"].notna())

    # Check if weathersit is integer
    assert all(isinstance(x, int) for x in processed_df["weathersit"].tolist())

    # Check if base columns are in the correct order
    base_cols = params.base_columns
    assert list(processed_df.columns[: len(base_cols)]) == base_cols

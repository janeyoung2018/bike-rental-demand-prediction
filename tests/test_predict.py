# test_predict.py

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from src.predict import Predicter


@pytest.fixture
def sample_data1():
    """Sample data with many rows to allow for large shifts."""
    date_range = pd.date_range(
        start="2012-09-01 00:00", end="2012-10-10 23:00", freq="h"
    )
    data = {
        "dteday": date_range.date,
        "hr": date_range.hour,
        "cnt": np.random.randint(10, 100, len(date_range)),
        "temp": np.linspace(0.5, 1.0, len(date_range)),
        "weathersit": np.random.randint(1, 4, len(date_range)),
        "atemp": np.linspace(0.5, 1.0, len(date_range)),
        "hum": np.linspace(0.5, 1.0, len(date_range)),
        "windspeed": np.linspace(0, 1.0, len(date_range)),
        "season": [1] * len(date_range),
        "yr": [0] * len(date_range),
        "mnth": [1] * len(date_range),
        "day": date_range.day,
        "weekday": date_range.weekday,
        "workingday": [0 if d in [5, 6] else 1 for d in date_range.weekday],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_data2():
    data = {
        "dteday": pd.to_datetime(
            ["2012-10-02", "2012-10-02", "2012-10-03", "2012-10-03"]
        ),
        "hr": [23, 0, 23, 0],
        "cnt": [10, 20, 30, 40],
        "temp": [0.5, 0.6, 0.7, 0.8],
        "hum": [0.81, 0.76, 0.8, 0.75],
        "weathersit": [1, 2, 1, 3],
        "windspeed": [0.2985, 0, 0.194, 0],
        "atemp": [0.5, 0.6, 0.7, 0.8],
        "season": [3, 3, 4, 4],
        "yr": [1, 1, 1, 1],
        "mnth": [10, 10, 10, 10],
        "day": [2, 2, 3, 3],
        "weekday": [2, 2, 3, 3],
        "workingday": [1, 1, 1, 1],
        "lag_1": [10, 20, 30, 40],
        "rolling_24": [10, 20, 30, 40],
        "rolling_168": [10, 20, 30, 40],
        "day_mean": [10, 20, 30, 40],
        "day_max": [10, 20, 30, 40],
        "day_min": [10, 20, 30, 40],
        "day_std": [10, 20, 30, 40],
        "day_sum": [10, 20, 30, 40],
    }

    return pd.DataFrame(data)


@pytest.fixture
def mock_model():
    feature_list = [
        0,
        1,
        2,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
    ]
    model = xgb.XGBRegressor()
    model.fit(np.array([feature_list, feature_list]), np.array([1, 2]))
    return model


@pytest.fixture
def mock_predicter(tmp_path, sample_data1, mock_model):
    data_path = tmp_path / "hour.csv"
    sample_data1.to_csv(data_path, index=False)
    model_dir = tmp_path / "models" / "latest_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    output_path = tmp_path / "prediction"
    daily_infer = False
    daily_infer_dt = "2012-12-31"
    test_cutoff_dt = "2012-10-01"
    for i in range(1, 4):
        mock_model.save_model(str(model_dir / f"model_t+{i}.json"))
    return Predicter(
        str(data_path),
        str(tmp_path / "models"),
        str(output_path),
        daily_infer,
        daily_infer_dt,
        test_cutoff_dt,
    )


def test_preprocess_inference_data(mock_predicter):
    processed_df = mock_predicter.preprocess_inference_data()
    assert isinstance(processed_df, pd.DataFrame)
    assert "lag_1" in processed_df.columns


def test_shift_weather_features(mock_predicter, sample_data2):
    df_shifted = mock_predicter.shift_weather_features(sample_data2.copy(), 1)
    assert df_shifted["weathersit"].iloc[0] == 2
    assert np.isnan(df_shifted["weathersit"].iloc[3])


def test_create_target_cols(mock_predicter, sample_data2):
    df_target = mock_predicter.create_target_cols(sample_data2.copy(), 1)
    assert "t+1" in df_target.columns
    assert df_target["t+1"].iloc[0] == 20.0
    assert np.isnan(df_target["t+1"].iloc[3])


def test_load_horizon_model(mock_predicter, mock_model):
    loaded_model = mock_predicter.load_horizon_model(1)
    assert isinstance(loaded_model, xgb.XGBRegressor)


def test_predict_for_horizon(mock_predicter, mock_model, sample_data2):
    mock_predicter.df_processed = sample_data2.copy()
    mock_predicter.predict_for_horizon(1, mock_model)
    assert "t+1" in mock_predicter.maes


def test_predict_all_horizons(mock_predicter, mock_model, sample_data1):
    mock_predicter.predict_all_horizons(start=1, end=3)
    assert len(mock_predicter.maes) == 3


def test_evaluation(mock_predicter, mock_model, sample_data1):
    mock_predicter.predict_all_horizons(start=1, end=2)
    mock_predicter.evaluation()
    assert isinstance(mock_predicter.maes, dict)

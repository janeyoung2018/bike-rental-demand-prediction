# test_train.py

import pandas as pd
import numpy as np
import pytest
import xgboost as xgb
from train import HorizonModelTrainer


@pytest.fixture
def sample_data1():
    date_range = pd.date_range(
        start="2012-09-01 00:00", end="2012-10-10 23:00", freq="h"
    )
    data = {
        "dteday": date_range.date,
        "hr": date_range.hour,
        "cnt": np.random.randint(10, 100, len(date_range)),
        "temp": np.linspace(0.5, 1.5, len(date_range)),
        "weathersit": np.random.randint(1, 4, len(date_range)),
        "atemp": np.linspace(0.5, 1.5, len(date_range)),
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
def mock_trainer(tmp_path, sample_data1):
    data_path = tmp_path / "hour.csv"
    sample_data1.to_csv(data_path, index=False)
    model_dir = tmp_path / "models"
    train_cutoff_dt = "2012-07-01"
    val_cutoff_dt = "2012-10-01"
    return HorizonModelTrainer(
        str(data_path), str(model_dir), train_cutoff_dt, val_cutoff_dt
    )


def test_preprocess_train_data(mock_trainer, sample_data1):
    processed_df = mock_trainer.preprocess_train_data()
    assert isinstance(processed_df, pd.DataFrame)
    assert "lag_1" in processed_df.columns


def test_shift_weather_features(mock_trainer, sample_data2):
    df_shifted = mock_trainer.shift_weather_features(sample_data2.copy(), 1)
    assert df_shifted["weathersit"].iloc[0] == 2.0
    assert np.isnan(df_shifted["weathersit"].iloc[-1])


def test_create_target_cols(mock_trainer, sample_data2):
    df_target = mock_trainer.create_target_cols(sample_data2.copy(), 1)
    assert "t+1" in df_target.columns
    assert df_target["t+1"].iloc[0] == sample_data2["cnt"].iloc[1]
    assert np.isnan(df_target["t+1"].iloc[-1])


def test_split_data(mock_trainer, sample_data1):
    mock_trainer.df_processed = mock_trainer.preprocess_train_data()
    mock_trainer.df_horizon = mock_trainer.shift_weather_features(
        mock_trainer.df_processed, 1
    )
    mock_trainer.df_horizon = mock_trainer.create_target_cols(
        mock_trainer.df_horizon, 1
    )
    X_train, y_train, X_val, y_val = mock_trainer.split_data(
        mock_trainer.df_horizon, "t+1"
    )
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(X_val, pd.DataFrame)
    assert isinstance(y_val, pd.Series)


def test_train_for_horizon(mock_trainer, sample_data1):
    mock_trainer.df_processed = mock_trainer.preprocess_train_data()
    mock_trainer.train_for_horizon(1)
    assert "t+1" in mock_trainer.models
    assert isinstance(mock_trainer.models["t+1"], xgb.XGBRegressor)
    assert "t+1" in mock_trainer.best_params_list
    assert isinstance(mock_trainer.best_params_list["t+1"], dict)


def test_train_all_horizons(mock_trainer, sample_data2):
    mock_trainer.train_all_horizons(start=1, end=2)
    assert len(mock_trainer.models) == 2
    assert "t+1" in mock_trainer.models
    assert "t+2" in mock_trainer.models


def test_save_models(mock_trainer, sample_data1, tmp_path):
    mock_trainer.df_processed = mock_trainer.preprocess_train_data()
    mock_trainer.train_all_horizons(start=1, end=2)
    mock_trainer.save_models()
    today_str = pd.Timestamp("today").strftime("%Y%m%d")
    model_dir = tmp_path / f"models/models_{today_str}"
    assert (model_dir / "model_t+1.json").exists()
    assert (model_dir / "model_t+2.json").exists()

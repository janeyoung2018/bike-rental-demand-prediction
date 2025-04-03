from typing import List


class Params:
    feature_cols: List[str] = [
        "season",
        "yr",
        "mnth",
        "day",
        "hr",
        "weekday",
        "workingday",
        "weathersit",
        "temp",
        "atemp",
        "hum",
        "windspeed",
        "lag_1",
        "rolling_24",
        "rolling_168",
        "day_mean",
        "day_max",
        "day_min",
        "day_std",
        "day_sum",
    ]
    categorical_columns: List[str] = [
        "season",
        "yr",
        "mnth",
        "day",
        "hr",
        "weekday",
        "workingday",
        "weathersit",
    ]
    numerical_columns: List[str] = ["temp", "atemp", "hum", "windspeed"]
    drop_columns: List[str] = ["casual", "registered", "instant"]
    day_static_columns: List[str] = [
        "season",
        "yr",
        "mnth",
        "holiday",
        "weekday",
        "workingday",
    ]
    weather_columns: List[str] = [
        "weathersit",
        "temp",
        "atemp",
        "hum",
        "windspeed",
    ]
    base_columns: List[str] = ["datetime", "dteday", "hr", "cnt"]
    date_col: str = "dteday"
    label_col: str = "cnt"

# preprocessing.py
# flake8: noqa

import pandas as pd
from utils.config import Params as params
import logging

logging.basicConfig(
    format="%(asctime)s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills missing records by reindexing full datetime range,
    setting missing cnt to 0, and restoring original int dtypes.

    Parameters:
        df (pd.DataFrame): Raw DataFrame

    Returns:
        pd.DataFrame: Cleaned DataFrame with complete hourly coverage,
                      'filled' flag, and original int types restored.
    """
    df = df.copy()

    # Ensure date col is datetime
    date_col = params.date_col
    df[date_col] = pd.to_datetime(df[date_col])

    # Create full datetime index
    df["datetime"] = df[date_col] + pd.to_timedelta(df["hr"], unit="h")
    df.set_index("datetime", inplace=True)

    # Create full hourly range
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="h")
    df_full = df.reindex(full_range)

    # Fill label col with 0 where missing
    label_col = params.label_col
    df_full[label_col] = df_full[label_col].fillna(0).astype(int)

    # Restore datetime fields
    df_full = df_full.reset_index().rename(columns={"index": "datetime"})

    # Extract date and hour
    df_full["dteday"] = pd.to_datetime(df_full["datetime"].dt.date)
    df_full["hr"] = df_full["datetime"].dt.hour

    # Fill static daily columns using values from the same day
    day_static_cols = params.day_static_columns

    for col in day_static_cols:
        if col in df_full.columns:
            # Use transform to align the index after groupby and filling
            df_full[col] = (
                df_full.groupby(date_col)[col]
                .transform(lambda x: x.ffill().bfill())
                .astype(int)
            )

    # Weather-related columns: fill forward(not by day)
    weather_cols = params.weather_columns
    for col in weather_cols:
        if col in df_full.columns:
            df_full[col] = df_full[col].ffill()

    # Final cast to int for weathersit
    df_full["weathersit"] = df_full["weathersit"].astype(int)

    # Reorder common columns for readability
    base_cols = params.base_columns
    remaining_cols = [col for col in df_full.columns if col not in base_cols]
    df_full = df_full[base_cols + remaining_cols]

    return df_full


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Loads, handles missing values, distinguishing between numerical and categorical features.

    Args:
        filepath (str): Path to the CSV file.
        target_column (str): Name of the target variable column.

    Returns:
        df (pd.DataFrame): DataFrame
    """
    df = df.copy()

    # Apply preprocessing
    df_processed = fill_missing_values(df)
    logging.info("Preprocessing completed.")

    return df_processed

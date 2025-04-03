# flake8: noqa
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import plotly.graph_objects as go
import numpy as np


def fill_missing_hours(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills missing hourly records by reindexing full datetime range,
    setting missing cnt to 0, and restoring original int dtypes.

    Parameters:
        df (pd.DataFrame): Raw DataFrame

    Returns:
        pd.DataFrame: Cleaned DataFrame with complete hourly coverage,
                      'filled' flag, and original int types restored.
    """
    df = df.copy()

    # Ensure dteday is datetime
    df["dteday"] = pd.to_datetime(df["dteday"])

    # Create full datetime index
    df["datetime"] = df["dteday"] + pd.to_timedelta(df["hr"], unit="h")
    df.set_index("datetime", inplace=True)

    # Create full hourly range
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="h")
    df_full = df.reindex(full_range)

    # Fill cnt, casual, registered with 0 where missing, and add filled flag
    df_full["cnt"] = df_full["cnt"].fillna(0).astype(int)
    df_full["casual"] = df_full["casual"].fillna(0).astype(int)
    df_full["registered"] = df_full["registered"].fillna(0).astype(int)
    df_full["filled"] = df_full["cnt"] == 0

    # Restore datetime fields
    df_full = df_full.reset_index().rename(columns={"index": "datetime"})

    # Reorder record index
    df_full = (
        df_full.drop(columns="instant")
        .reset_index()
        .rename(columns={"index": "instant"})
    )
    df_full["instant"] += 1

    # Extract date and hour
    df_full["dteday"] = df_full["datetime"].dt.date
    df_full["hr"] = df_full["datetime"].dt.hour

    # Fill static daily columns using values from the same day
    day_static_cols = ["season", "yr", "mnth", "holiday", "weekday", "workingday"]

    for col in day_static_cols:
        if col in df_full.columns:
            # Use transform to align the index after groupby and filling
            df_full[col] = (
                df_full.groupby("dteday")[col]
                .transform(lambda x: x.ffill().bfill())
                .astype(int)
            )

    # Weather-related columns: fill forward/backward (not by day)
    weather_cols = ["weathersit", "temp", "atemp", "hum", "windspeed"]
    for col in weather_cols:
        if col in df_full.columns:
            df_full[col] = df_full[col].ffill().bfill()

    # Final cast to int for weathersit
    df_full["weathersit"] = df_full["weathersit"].astype(int)

    # Reorder common columns for readability
    base_cols = ["datetime", "dteday", "hr", "cnt", "filled"]
    remaining_cols = [col for col in df_full.columns if col not in base_cols]
    df_full = df_full[base_cols + remaining_cols]

    return df_full


def label_test_period(results_df, test_weeks):
    labels = []
    for _, row in results_df.iterrows():
        date = pd.to_datetime(row["date"])
        period_label = None
        for i, (start, end) in enumerate(test_weeks):
            if pd.to_datetime(start) <= date <= pd.to_datetime(end):
                period_label = f"Test Week {i+1} ({start} to {end})"
                break
        labels.append(period_label)
    results_df["test_period"] = labels
    return results_df


def evaluate_baseline(df, test_weeks):
    baseline_results = []

    for start, end in test_weeks:
        for day in pd.date_range(start, end, freq="D"):
            test_start = day
            test_end = day + pd.Timedelta(hours=23)

            predictions = []
            actuals = []

            for timestamp in pd.date_range(test_start, test_end, freq="h"):
                dow = timestamp.dayofweek
                hour = timestamp.hour
                past_values = []

                # Look back 1 to 7 weeks (same day and hour)
                for weeks_back in range(1, 8):
                    past_time = timestamp - pd.Timedelta(weeks=weeks_back)
                    if (
                        past_time in df.index
                        and past_time.hour == hour
                        and past_time.dayofweek == dow
                    ):
                        past_values.append(df.loc[past_time])

                if past_values:
                    pred = np.mean(past_values)
                    predictions.append(pred)
                    actuals.append(df.loc[timestamp])

            if len(predictions) == 24:
                mae = mean_absolute_error(actuals, predictions)
                baseline_results.append(
                    {"model": "Baseline", "date": day.date(), "mae": mae}
                )

    return pd.DataFrame(baseline_results)


def evaluate_arima(df, test_weeks):
    df = df.asfreq("h")  # Sets explicit hourly frequency
    arima_results = []
    for start, end in test_weeks:
        for day in pd.date_range(start, end, freq="D"):
            train_end = day - pd.Timedelta(hours=1)
            test_start = day
            test_end = day + pd.Timedelta(hours=23)

            train_data = df.loc[:train_end]
            test_data = df.loc[test_start:test_end]

            # Apply Seasonal Differencing
            seasonal_lag = 24
            train_diff = train_data.diff(seasonal_lag).dropna()
            # check_stationarity(train_diff)

            # Fit ARIMA Model
            model = ARIMA(train_diff, order=(2, 0, 3), freq="h")
            results = model.fit()

            # Forecast Test
            test_forecast_diff = results.forecast(steps=24)

            # Step 1: Inverse first seasonal_lag values using train data
            initial_base = train_data.iloc[-seasonal_lag:]
            initial_base.index = test_data.index[:seasonal_lag]
            test_forecast_original = pd.Series(index=test_data.index, dtype="float64")
            test_forecast_original.iloc[:seasonal_lag] = (
                test_forecast_diff[:seasonal_lag] + initial_base
            )

            # Step 2: Inverse transform remaining values recursively
            for i in range(seasonal_lag, len(test_forecast_diff)):
                base = test_forecast_original.iloc[i - seasonal_lag]
                test_forecast_original.iloc[i] = base + test_forecast_diff[i]

            mae = mean_absolute_error(test_data, test_forecast_original)
            arima_results.append({"model": "ARIMA", "date": day.date(), "mae": mae})
    return pd.DataFrame(arima_results)


def create_xgb_features(data, lags=[24, 48, 168]):
    df_feat = data.copy()
    for lag in lags:
        df_feat[f"lag_{lag}"] = df_feat["cnt"].shift(lag)
    df_feat.dropna(inplace=True)
    return df_feat


def evaluate_xgboost_weekly(df, test_weeks):
    df["day"] = df["datetime"].dt.day
    df = df.set_index("datetime")
    df_feat = create_xgb_features(df)
    drop_columns = ["dteday", "filled", "instant", "casual", "registered"]
    df_feat = df_feat.drop(columns=drop_columns)
    xgb_results = []

    for start, end in test_weeks:
        test_days = pd.date_range(start, end, freq="D")

        # Group test days by week (starting from Monday)
        weeks = test_days.to_series().groupby(
            test_days.to_series().dt.isocalendar().week
        )

        for _, week_days in weeks:
            week_days = week_days.tolist()

            # Pick first day of the week to train
            first_day = week_days[0]
            train_end = first_day - pd.Timedelta(hours=1)

            train_data = df_feat.loc[:train_end]
            if len(train_data) < 100:
                continue

            # Prepare X/y
            X_train = train_data.drop(columns=["cnt"])
            y_train = train_data["cnt"]

            # Train only once per week
            model = xgb.XGBRegressor(n_estimators=100, max_depth=3)
            model.fit(X_train, y_train)

            # Predict for each day in that week
            for day in week_days:
                test_start = day
                test_end = day + pd.Timedelta(hours=23)
                test_data = df_feat.loc[test_start:test_end]

                if len(test_data) < 24:
                    continue

                X_test = test_data.drop(columns=["cnt"])
                y_test = test_data["cnt"]

                preds = model.predict(X_test)
                mae = mean_absolute_error(y_test, preds)
                xgb_results.append({"model": "XGBoost", "date": day.date(), "mae": mae})

    return pd.DataFrame(xgb_results)


def check_stationarity(ts):
    dftest = adfuller(ts)
    adf = dftest[0]
    pvalue = dftest[1]
    critical_value = dftest[4]["5%"]
    if (pvalue < 0.05) and (adf < critical_value):
        print("The series is stationary")
    else:
        print("The series is NOT stationary")


def plot_acf_pacf_plotly(data, lags=72):
    """Plots ACF and PACF using Plotly."""

    acf_values, _ = sm.tsa.acf(data, nlags=lags, alpha=0.05)
    pacf_values, _ = sm.tsa.pacf(data, nlags=lags, alpha=0.05)

    # ACF Plot
    fig_acf = go.Figure()
    fig_acf.add_trace(go.Bar(x=list(range(lags + 1)), y=acf_values, name="ACF"))
    fig_acf.update_layout(
        title="Autocorrelation Function (ACF)",
        xaxis_title="Lags",
        yaxis_title="ACF Value",
    )
    fig_acf.show()

    # PACF Plot
    fig_pacf = go.Figure()
    fig_pacf.add_trace(go.Bar(x=list(range(lags + 1)), y=pacf_values, name="PACF"))
    fig_pacf.update_layout(
        title="Partial Autocorrelation Function (PACF)",
        xaxis_title="Lags",
        yaxis_title="PACF Value",
    )
    fig_pacf.show()

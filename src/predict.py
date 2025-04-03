# predict.py

import logging
import os
from argparse import ArgumentParser
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

from preprocessing import preprocess_data
from utils.config import Params as params
from utils.utils import create_xgb_features

logging.basicConfig(level=logging.INFO)


class Predicter:
    def __init__(
        self,
        input_path,
        model_path,
        output_path,
        daily_infer,
        daily_infer_dt,
        test_cutoff_dt,
    ):
        self.input_path = input_path
        self.model_path = model_path
        self.output_path = output_path
        self.daily_infer = daily_infer
        self.daily_infer_dt = daily_infer_dt
        self.test_cutoff_dt = test_cutoff_dt
        self.maes = {}

    def preprocess_inference_data(self):
        df = pd.read_csv(self.input_path)
        df_processed = preprocess_data(df)
        df_processed = create_xgb_features(df_processed)
        # Convert categorical columns to 'category' type
        for col in params.categorical_columns:
            df_processed[col] = df_processed[col].astype("category")
        return df_processed

    def shift_weather_features(self, df_horizon, horizon):
        for col in params.weather_columns:
            df_horizon[col] = df_horizon[col].shift(-horizon)
        return df_horizon

    def create_target_cols(self, df_horizon, horizon):
        df_horizon[f"t+{horizon}"] = df_horizon[params.label_col].shift(-horizon)
        return df_horizon

    def predict_for_horizon(self, horizon, model):
        target_col = f"t+{horizon}"
        logging.info(f"\n=== Predicting for {target_col} ===")

        df_horizon = self.df_processed.copy()
        df_horizon = df_horizon.dropna().reset_index(drop=True)
        logging.info(df_horizon.shape)
        if not self.daily_infer:
            df_horizon = self.shift_weather_features(df_horizon, horizon)
            df_horizon = self.create_target_cols(df_horizon, horizon)
            daily_rows = (
                df_horizon[df_horizon["hr"] == 23].dropna().reset_index(drop=True)
            )
            daily_rows = daily_rows[daily_rows["dteday"] > self.test_cutoff_dt]
            self.X_infer = daily_rows[params.feature_cols]
            self.y_infer = daily_rows[target_col]
            self.y_pred = model.predict(self.X_infer)
            # make sure the prediction is non-negative and is integer
            self.y_pred = np.clip(
                np.round(self.y_pred).astype(int), a_min=0, a_max=None
            )
            mae = mean_absolute_error(self.y_infer, self.y_pred)
            self.maes[target_col] = mae
            logging.info(f"{target_col} MAE: {mae:.3f}")
        else:
            # assume we have the weather info of inference day, and the target values are unknown
            df_horizon = self.shift_weather_features(df_horizon, horizon)
            daily_rows = df_horizon[df_horizon["hr"] == 23].reset_index(drop=True)
            daily_rows = daily_rows[daily_rows["dteday"] == self.daily_infer_dt]
            self.X_infer = daily_rows[params.feature_cols]
            self.y_pred = model.predict(self.X_infer)
            self.y_pred = np.clip(
                np.round(self.y_pred).astype(int), a_min=0, a_max=None
            )
        if horizon == 1:
            self.results = daily_rows
        self.results[f"{target_col}_pred"] = self.y_pred

    def predict_all_horizons(self, start=1, end=24):
        self.df_processed = self.preprocess_inference_data()
        for horizon in range(start, end + 1):
            model = self.load_horizon_model(horizon)
            self.predict_for_horizon(horizon, model)

    def load_horizon_model(self, horizon):
        """Load model and predict on inference data"""
        model_dir = Path(self.model_path)
        available_models = sorted(os.listdir(str(model_dir)))
        newest_model_path = str(
            model_dir / available_models[-1] / f"model_t+{horizon}.json"
        )
        model = xgb.XGBRegressor()
        model.load_model(newest_model_path)
        return model

    def evaluation(self):
        """Evaluation."""
        # Overall MAE
        avg_mae = np.mean(list(self.maes.values()))
        logging.info(f"\nAverage MAE over 24 horizons: {avg_mae:.3f}")

    def save_results(self):
        """Save the inference results"""
        trw = (date.today() + timedelta(days=1)).strftime("%Y%m%d")
        # shift one day, since we predict for the next day
        self.results["dteday"] = pd.to_datetime(self.results["dteday"]) + pd.Timedelta(
            days=1
        )
        # Construct the output path
        if daily_infer:
            output_data_path = Path(self.output_path) / "infer" / f"pred_{trw}.csv"
        else:
            output_data_path = Path(self.output_path) / "test" / f"pred_{trw}.csv"
        self.results.to_csv(output_data_path)


if __name__ == "__main__":
    args = ArgumentParser()

    args.add_argument("--input_file_path", default="data/hour.csv", type=str)
    args.add_argument("--input_model_path", default="models", type=str)
    args.add_argument("--output_file_path", default="data/prediction", type=str)
    args.add_argument(
        "--daily_infer", help="daily inference or for test evaluation", default=False
    )
    args.add_argument(
        "--daily_infer_dt",
        help="If daily inference is true, provide the daily inference date, e.g., '2021-11-08'",
        default=date.today().strftime("%Y-%m-%d"),
        type=str,
    )
    args.add_argument(
        "--test_cutoff_dt",
        help="If daily inference is false, we are doing test evaluation, please provide the test cutoff date, e.g., '2021-11-08'",
        default="2012-10-01",
        type=str,
    )
    parsed_args, _ = args.parse_known_args()

    root_dir = os.getcwd()
    input_file_path = os.path.join(root_dir, parsed_args.input_file_path)
    model_path = os.path.join(root_dir, parsed_args.input_model_path)
    output_file_path = os.path.join(root_dir, parsed_args.output_file_path)
    daily_infer = parsed_args.daily_infer
    daily_infer_dt = parsed_args.daily_infer_dt
    test_cutoff_dt = parsed_args.test_cutoff_dt

    predicter = Predicter(
        input_file_path,
        model_path,
        output_file_path,
        daily_infer,
        daily_infer_dt,
        test_cutoff_dt,
    )
    predicter.predict_all_horizons()
    if not daily_infer:
        predicter.evaluation()
    predicter.save_results()
    logging.info("Prediction completed.")

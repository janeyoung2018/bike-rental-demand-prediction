# train.py

import logging
import os
from argparse import ArgumentParser
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.metrics import mean_absolute_error

from preprocessing import preprocess_data
from utils.config import Params as params
from utils.utils import create_xgb_features

logging.basicConfig(
    format="%(asctime)s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)


class HorizonModelTrainer:
    """
    Train a xgboost model for each hour
    """

    def __init__(self, file_path, model_path, train_cutoff_dt, val_cutoff_dt):
        self.file_path = file_path
        self.model_path = model_path
        self.train_cutoff_dt = train_cutoff_dt
        self.val_cutoff_dt = val_cutoff_dt
        self.models = {}
        self.best_params_list = {}

    def preprocess_train_data(self):
        df = pd.read_csv(self.file_path)
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

    def split_data(self, df_horizon, target_col):
        daily_rows = df_horizon[df_horizon["hr"] == 23].dropna().reset_index(drop=True)
        # Train date before 2012-07-01
        train = daily_rows[daily_rows["dteday"] < self.train_cutoff_dt]
        # Val date between 2012-07-01 and 2012-09-30
        val = daily_rows[
            (daily_rows["dteday"] >= self.train_cutoff_dt)
            & (daily_rows["dteday"] < self.val_cutoff_dt)
        ]

        X_train = train[params.feature_cols]
        y_train = train[target_col]
        X_val = val[params.feature_cols]
        y_val = val[target_col]

        return X_train, y_train, X_val, y_val

    def train_for_horizon(self, horizon):
        target_col = f"t+{horizon}"
        print(f"\n=== Training for {target_col} ===")

        df_horizon = self.df_processed.copy()
        df_horizon = self.shift_weather_features(df_horizon, horizon)
        df_horizon = self.create_target_cols(df_horizon, horizon)
        X_train, y_train, X_val, y_val = self.split_data(df_horizon, target_col)

        def objective(params):
            model = xgb.XGBRegressor(
                n_estimators=int(params["n_estimators"]),
                max_depth=int(params["max_depth"]),
                learning_rate=params["learning_rate"],
                subsample=params["subsample"],
                colsample_bytree=params["colsample_bytree"],
                enable_categorical=True,
                eval_metric="mae",
                early_stopping_rounds=10,
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            preds = model.predict(X_val)
            loss = mean_absolute_error(y_val, preds)
            return {"loss": loss, "status": STATUS_OK}

        search_space = {
            "n_estimators": hp.quniform("n_estimators", 50, 300, 10),
            "max_depth": hp.quniform("max_depth", 3, 10, 1),
            "learning_rate": hp.loguniform("learning_rate", -3, 0),
            "subsample": hp.uniform("subsample", 0.6, 1.0),
            "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 1.0),
        }

        trials = Trials()
        best_params = fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=50,
            trials=trials,
            rstate=np.random.default_rng(42),
        )

        best_params["n_estimators"] = int(best_params["n_estimators"])
        best_params["max_depth"] = int(best_params["max_depth"])

        final_model = xgb.XGBRegressor(
            **best_params, random_state=42, n_jobs=-1, enable_categorical=True
        )
        final_model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))

        self.models[target_col] = final_model
        self.best_params_list[target_col] = best_params

    def train_all_horizons(self, start=1, end=24):
        self.df_processed = self.preprocess_train_data()
        for horizon in range(start, end + 1):
            self.train_for_horizon(horizon)

    def save_models(self):
        # Format today's date as YYYYMMDD
        eval_date = date.today().strftime("%Y%m%d")
        # Construct the output path
        output_dir = Path(self.model_path) / f"models_{eval_date}"
        # Create the directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        for key, model in self.models.items():
            xgb_model_path = str(output_dir / f"model_{key}.json")
            model.save_model(xgb_model_path)


if __name__ == "__main__":
    args = ArgumentParser()

    args.add_argument("--input_file_path", default="data/hour.csv", type=str)
    args.add_argument("--output_model_path", default="models", type=str)
    args.add_argument(
        "--train_cutoff_dt",
        help="train cutoff date, e.g., '2021-11-08'",
        default="2012-07-01",
        type=str,
    )
    args.add_argument(
        "--val_cutoff_dt",
        help="validation cutoff date, e.g., '2021-11-08'",
        default="2012-10-01",
        type=str,
    )
    parsed_args, _ = args.parse_known_args()

    root_dir = os.getcwd()
    file_path = os.path.join(root_dir, parsed_args.input_file_path)
    model_path = os.path.join(root_dir, parsed_args.output_model_path)
    train_cutoff_dt = parsed_args.train_cutoff_dt
    val_cutoff_dt = parsed_args.val_cutoff_dt
    trainer = HorizonModelTrainer(file_path, model_path, train_cutoff_dt, val_cutoff_dt)
    trainer.train_all_horizons()
    trainer.save_models()
    logging.info("Training completed.")

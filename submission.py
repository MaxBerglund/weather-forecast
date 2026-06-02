from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import emflow as ef


class QuantileRegressionPredictor(ef.Predictor):
    def __init__(self, name="quantile-regression-ar"):
        self.name = name
        self.lags = [1, 24]
        self.quantile = 0.5
        self.models = {}

    def _prepare_features(self, df: pd.DataFrame, col: str):
        series = df[col]
        # Use timezone-normalized timestamps to derive time features
        # Note: input_df index is expected to be a DatetimeIndex
        hour = df.index.hour
        day_of_week = df.index.dayofweek
        month = df.index.month

        X = pd.DataFrame(
            {
                "hour": hour,
                "day_of_week": day_of_week,
                "month": month,
                "lag_1h": series.shift(1),
                "lag_24h": series.shift(24),
            },
            index=df.index,
        )
        return X

    def train(self, train_df: pd.DataFrame):
        if isinstance(train_df, pd.Series):
            train_df = train_df.to_frame()

        # Filter for Stockholm-Observatoriekullen A as per user's specific interest
        # though the contract implies handling all columns passed.
        # Given the "swedish-temperatures:ar" context, we handle all.
        for col in train_df.columns:
            X = self._prepare_features(train_df, col)
            y = train_df[col]

            # Combine and drop NaNs
            data = pd.concat([X, y], axis=1).dropna()
            if len(data) < 100:  # Heuristic for sufficient data
                self.models[col] = None
                continue

            X_train = data.drop(columns=[col])
            y_train = data[col]

            model = GradientBoostingRegressor(
                loss="quantile",
                alpha=self.quantile,
                n_estimators=100,
                max_depth=5,
                random_state=42,
            )
            model.fit(X_train, y_train)
            self.models[col] = model
        return self

    def predict(self, input_df: pd.DataFrame):
        if isinstance(input_df, pd.Series):
            input_df = input_df.to_frame()

        preds = {}
        for col in input_df.columns:
            model = self.models.get(col)
            if model is None:
                preds[col] = np.full(len(input_df), np.nan)
                continue

            X = self._prepare_features(input_df, col)
            # We can't dropna here because we need a value for the last row
            # even if it has NaNs in current values (but lags should be there)
            # The contract says last timestamp is the target to forecast.
            # Only the current value at last timestamp is NaN. Lags should be fine.

            # Fill NaNs in features with a neutral value or previous if necessary
            # but usually for the last row, shift(1) of NaN is the value at T-1.
            out = np.full(len(input_df), np.nan)

            # Identify rows where we have all features
            valid_mask = X.notna().all(axis=1)
            if valid_mask.any():
                out[valid_mask] = model.predict(X[valid_mask])

            preds[col] = out

        return pd.DataFrame(preds, index=input_df.index, columns=input_df.columns)


model = QuantileRegressionPredictor()

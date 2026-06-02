import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import joblib


def main():
    # 1. Load the data
    print("Loading data...")
    train_df = pd.read_parquet("/home/maxb/rebase/hack/data/train.parquet")
    test_df = pd.read_parquet("/home/maxb/rebase/hack/data/test.parquet")

    # 2. Feature Engineering
    # Combine to ensure consistent lagging
    train_df["set"] = "train"
    test_df["set"] = "test"
    full_df = (
        pd.concat([train_df, test_df], axis=0)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    full_df["timestamp"] = pd.to_datetime(full_df["timestamp"])

    # Handle missing values in temperature
    # Using interpolation for time series
    full_df["temperature"] = full_df["temperature"].interpolate(method="linear")

    # Create time-based features
    full_df["hour"] = full_df["timestamp"].dt.hour
    full_df["day_of_week"] = full_df["timestamp"].dt.dayofweek
    full_df["month"] = full_df["timestamp"].dt.month

    # Create lag features
    # Note: In a real scenario, you'd be careful not to leak future data.
    # Here, we lag the 'temperature' which is our target.
    full_df["lag_1h"] = full_df["temperature"].shift(1)
    full_df["lag_24h"] = full_df["temperature"].shift(24)

    # Drop rows where lags are NaN (the first 24 hours of training)
    full_df = full_df.dropna(subset=["lag_1h", "lag_24h"])

    # Split back
    train_processed = full_df[full_df["set"] == "train"].copy()
    test_processed = full_df[full_df["set"] == "test"].copy()

    features = ["hour", "day_of_week", "month", "lag_1h", "lag_24h"]
    target = "temperature"

    X_train = train_processed[features]
    y_train = train_processed[target]
    X_test = test_processed[features]
    y_test = test_processed[target]

    # 3. Train Quantile Regression models
    quantiles = [0.1, 0.5, 0.9]
    models = {}
    preds = {}

    print("Training models...")
    for q in quantiles:
        print(f"  Training quantile {q}...")
        model = GradientBoostingRegressor(
            loss="quantile", alpha=q, n_estimators=100, random_state=42
        )
        model.fit(X_train, y_train)
        models[q] = model

        # 4. Predict on test data
        preds[q] = model.predict(X_test)

    # 5. Calculate RMSE for 0.5 quantile
    rmse = np.sqrt(mean_squared_error(y_test, preds[0.5]))

    # 6. Summary
    summary = pd.DataFrame(
        {"actual": y_test, "q0.1": preds[0.1], "q0.5": preds[0.5], "q0.9": preds[0.9]}
    )

    print(f"\nRMSE (0.5 quantile): {rmse:.4f}")
    print("\nPrediction Summary (First 5 rows of test):")
    print(summary.head())
    print("\nQuantile Coverage Statistics:")
    print(f"  % actual < q0.1: {(summary['actual'] < summary['q0.1']).mean():.2%}")
    print(f"  % actual < q0.5: {(summary['actual'] < summary['q0.5']).mean():.2%}")
    print(f"  % actual < q0.9: {(summary['actual'] < summary['q0.9']).mean():.2%}")

    # 7. Save the models
    joblib.dump(models, "/home/maxb/rebase/hack/model.pkl")
    print(f"\nModels saved to /home/maxb/rebase/hack/model.pkl")


if __name__ == "__main__":
    main()

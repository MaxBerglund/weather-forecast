import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuration
MODEL_PATH = "/home/maxb/rebase/hack/model.pkl"
TEST_DATA_PATH = "/home/maxb/rebase/hack/data/test.parquet"
OUTPUT_PLOT = "forecast_visualization.png"


def load_data_and_model():
    print(f"Loading model from {MODEL_PATH}...")
    models = joblib.load(MODEL_PATH)

    print(f"Loading test data from {TEST_DATA_PATH}...")
    df_test = pd.read_parquet(TEST_DATA_PATH)

    # Preprocessing (must match training)
    df_test["timestamp"] = pd.to_datetime(df_test["timestamp"])
    df_test = df_test.sort_values("timestamp")

    # Feature engineering
    df_test["hour"] = df_test["timestamp"].dt.hour
    df_test["day_of_week"] = df_test["timestamp"].dt.dayofweek
    df_test["month"] = df_test["timestamp"].dt.month
    df_test["lag_1h"] = df_test["temperature"].shift(1)
    df_test["lag_24h"] = df_test["temperature"].shift(24)

    # Drop rows with NaN from lags
    df_test = df_test.dropna()

    return models, df_test


def visualize_results(models, df):
    print("Generating predictions...")
    X = df[["hour", "day_of_week", "month", "lag_1h", "lag_24h"]]
    y_actual = df["temperature"]

    # Get predictions for each quantile
    # The models dictionary was saved as {0.1: m1, 0.5: m2, 0.9: m3}
    preds = {}
    for q, model in models.items():
        preds[q] = model.predict(X)

    print("Plotting results...")
    plt.figure(figsize=(15, 7))

    # Plot a slice of the data for better visibility (first 500 hours)
    slice_idx = 500
    time = df["timestamp"].iloc[:slice_idx]

    plt.plot(
        time,
        y_actual.iloc[:slice_idx],
        color="black",
        label="Actual Temperature",
        linewidth=1,
        alpha=0.7,
    )
    plt.plot(
        time,
        preds[0.5][:slice_idx],
        color="blue",
        label="Median Prediction (q0.5)",
        linewidth=1.5,
    )

    # Fill between quantiles
    plt.fill_between(
        time,
        preds[0.1][:slice_idx],
        preds[0.9][:slice_idx],
        color="blue",
        alpha=0.2,
        label="80% Prediction Interval (q0.1 - q0.9)",
    )

    plt.title("Probabilistic Temperature Forecast (Stockholm-Observatoriekullen 2026)")
    plt.xlabel("Timestamp")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(OUTPUT_PLOT)
    print(f"Visualization saved to {OUTPUT_PLOT}")


if __name__ == "__main__":
    if os.path.exists(MODEL_PATH) and os.path.exists(TEST_DATA_PATH):
        m, d = load_data_and_model()
        visualize_results(m, d)
    else:
        print("Required files not found.")

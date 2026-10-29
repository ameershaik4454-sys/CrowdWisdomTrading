
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

INPUT = Path("data/processed/btc_vader_sentiment_dataset_2022_2023.csv")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

print("Loading dataset...", flush=True)
df = pd.read_csv(INPUT)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)

for col in [
    "close", "average_sentiment", "tweet_count",
    "positive_count", "neutral_count", "negative_count"
]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df["daily_return"] = df["close"].pct_change()

feature_cols = []

# Use previous day's sentiment data.
for col in [
    "average_sentiment", "tweet_count",
    "positive_count", "neutral_count", "negative_count"
]:
    if col in df.columns:
        name = col + "_lag1"
        df[name] = df[col].shift(1)
        feature_cols.append(name)

# Historical returns.
for lag in [1, 2, 3, 5, 7]:
    name = f"return_lag{lag}"
    df[name] = df["daily_return"].shift(lag)
    feature_cols.append(name)

# Rolling statistics use current and previous returns.
# They are available after the current day's close.
for window in [3, 7, 14]:
    for stat, series in [
        ("mean", df["daily_return"].rolling(window).mean()),
        ("std", df["daily_return"].rolling(window).std()),
    ]:
        name = f"return_{stat}_{window}d"
        df[name] = series
        feature_cols.append(name)

metrics = []
all_predictions = []

for horizon in range(1, 6):
    print(f"Training horizon +{horizon} day...", flush=True)

    work = df.copy()
    work["target_return"] = (
        work["close"].shift(-horizon) / work["close"] - 1
    )

    work = work.replace([np.inf, -np.inf], np.nan)
    work = work.dropna(
        subset=feature_cols + ["target_return", "close", "date"]
    ).reset_index(drop=True)

    split = int(len(work) * 0.80)
    if split < 100 or split >= len(work):
        raise ValueError("Not enough data for walk-forward evaluation.")

    actual_returns = []
    predicted_returns = []
    records = []

    # Expanding-window evaluation; refit every 7 test days.
    cached_model = None
    last_fit_index = -7

    for i in range(split, len(work)):
        if cached_model is None or i - last_fit_index >= 7:

            # A training target must be known by the forecast
            # origin. For a target at j + horizon, require j+horizon <= i.
            train_end = i - horizon + 1
            train = work.iloc[:train_end]

            if len(train) < 100:
                raise ValueError(
                    f"Not enough training rows for horizon {horizon} "
                    f"at index {i}."
                )

            cached_model = RandomForestRegressor(
                n_estimators=60,
                max_depth=5,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            )
            cached_model.fit(train[feature_cols], train["target_return"])
            last_fit_index = i

        row = work.iloc[[i]]
        pred = float(cached_model.predict(row[feature_cols])[0])
        actual = float(row["target_return"].iloc[0])
        close = float(row["close"].iloc[0])

        actual_returns.append(actual)
        predicted_returns.append(pred)

        records.append({
            "date": row["date"].iloc[0],
            "horizon_days": horizon,
            "current_close": close,
            "actual_future_return_pct": actual * 100,
            "predicted_future_return_pct": pred * 100,
            "actual_future_close": close * (1 + actual),
            "predicted_future_close": close * (1 + pred),
        })

    a = np.asarray(actual_returns)
    p = np.asarray(predicted_returns)

    test_closes = work.iloc[split:]["close"].to_numpy()
    price_actual = test_closes * (1 + a)
    price_pred = test_closes * (1 + p)

    metrics.append({
        "horizon_days": horizon,
        "test_predictions": len(a),
        "return_MAE_percentage_points":
            mean_absolute_error(a, p) * 100,
        "return_RMSE_percentage_points":
            np.sqrt(mean_squared_error(a, p)) * 100,
        "directional_accuracy_pct":
            np.mean(np.sign(a) == np.sign(p)) * 100,
        "price_MAE_USD":
            mean_absolute_error(price_actual, price_pred),
    })

    all_predictions.extend(records)

    # Save progress after each horizon.
    pd.DataFrame(metrics).to_csv(
        OUT / "walk_forward_model_metrics.csv", index=False
    )
    pd.DataFrame(all_predictions).to_csv(
        OUT / "walk_forward_predictions.csv", index=False
    )

print("\nWALK-FORWARD RESULTS")
print(pd.DataFrame(metrics).round(4).to_string(index=False))
print("\nSaved metrics and predictions in data/processed/")

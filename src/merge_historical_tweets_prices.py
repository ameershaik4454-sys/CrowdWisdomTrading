
import pandas as pd
from pathlib import Path

# Project directories
root = Path(__file__).resolve().parent.parent

tweets_file = root / "data" / "raw" / "bitcoin_tweets_dataset.csv"
prices_file = root / "data" / "raw" / "btc_prices_2022_2023.csv"
output_file = (
    root / "data" / "processed" /
    "btc_sentiment_dataset_2022_2023.csv"
)

print("Step 1: Loading tweet and Bitcoin price data...")

tweets = pd.read_csv(tweets_file)
prices = pd.read_csv(prices_file)

# Standardize dates
tweets["date"] = pd.to_datetime(
    tweets["date"], errors="coerce"
).dt.strftime("%Y-%m-%d")

prices["date"] = pd.to_datetime(
    prices["date"], errors="coerce"
).dt.strftime("%Y-%m-%d")

# Clean sentiment labels
tweets["sentiment_label"] = (
    tweets["sentiment_label"].astype(str).str.strip().str.title()
)

# Count each sentiment label for every day
daily_counts = (
    tweets.groupby(["date", "sentiment_label"])
    .size()
    .unstack(fill_value=0)
)

# Ensure all three columns exist
for label in ["Positive", "Neutral", "Negative"]:
    if label not in daily_counts.columns:
        daily_counts[label] = 0

daily_counts = daily_counts[
    ["Positive", "Neutral", "Negative"]
]

daily_counts = daily_counts.rename(columns={
    "Positive": "positive_count",
    "Neutral": "neutral_count",
    "Negative": "negative_count"
})

daily_counts["tweet_count"] = daily_counts[
    ["positive_count", "neutral_count", "negative_count"]
].sum(axis=1)

# A transparent daily sentiment measure based on the labels:
# -1 = all negative, 0 = balanced, +1 = all positive
daily_counts["average_sentiment"] = (
    daily_counts["positive_count"] -
    daily_counts["negative_count"]
) / daily_counts["tweet_count"].replace(0, pd.NA)

daily_counts = daily_counts.reset_index()

print("Step 2: Merging daily sentiment with Bitcoin prices...")

dataset = prices.merge(
    daily_counts,
    on="date",
    how="left",
    validate="one_to_one"
)

# Fill counts with zero if a date has no tweets
count_columns = [
    "positive_count", "neutral_count",
    "negative_count", "tweet_count"
]

dataset[count_columns] = dataset[count_columns].fillna(0).astype(int)

# Calculate daily percentage change in Bitcoin's closing price
dataset["daily_return_pct"] = dataset["close"].pct_change() * 100

# Save the new dataset
output_file.parent.mkdir(parents=True, exist_ok=True)
dataset.to_csv(output_file, index=False)

print("\nSUCCESS! Dataset created.")
print("Total daily rows:", len(dataset))
print("Total tweets:", dataset["tweet_count"].sum())
print("First date:", dataset["date"].min())
print("Last date:", dataset["date"].max())
print("Saved to:", output_file)

print("\nFirst five rows:")
print(dataset.head().to_string(index=False))

print("\nMissing values:")
print(dataset.isna().sum())

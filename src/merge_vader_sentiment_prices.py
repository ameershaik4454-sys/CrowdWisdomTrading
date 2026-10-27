
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent.parent

tweets_file = (
    root / "data" / "processed" /
    "historical_tweets_vader_sentiment.csv"
)
prices_file = root / "data" / "raw" / "btc_prices_2022_2023.csv"
output_file = (
    root / "data" / "processed" /
    "btc_vader_sentiment_dataset_2022_2023.csv"
)

# Load data
tweets = pd.read_csv(tweets_file)
prices = pd.read_csv(prices_file)

tweets["date"] = pd.to_datetime(
    tweets["date"], errors="coerce"
).dt.strftime("%Y-%m-%d")

prices["date"] = pd.to_datetime(
    prices["date"], errors="coerce"
).dt.strftime("%Y-%m-%d")

tweets["vader_compound"] = pd.to_numeric(
    tweets["vader_compound"], errors="coerce"
)

# Calculate daily VADER sentiment and tweet volume
daily = (
    tweets.dropna(subset=["date", "vader_compound"])
    .groupby("date")
    .agg(
        average_sentiment=("vader_compound", "mean"),
        tweet_count=("vader_compound", "size"),
        positive_count=("vader_label", lambda x: (x == "positive").sum()),
        neutral_count=("vader_label", lambda x: (x == "neutral").sum()),
        negative_count=("vader_label", lambda x: (x == "negative").sum()),
    )
    .reset_index()
)

# Merge daily sentiment with daily Bitcoin closing prices
dataset = prices.merge(
    daily, on="date", how="left", validate="one_to_one"
)

count_columns = [
    "tweet_count", "positive_count", "neutral_count", "negative_count"
]
dataset[count_columns] = dataset[count_columns].fillna(0).astype(int)

# Keep missing sentiment as NaN on days with no tweets.
dataset["daily_return_pct"] = dataset["close"].pct_change() * 100

output_file.parent.mkdir(parents=True, exist_ok=True)
dataset.to_csv(output_file, index=False)

print("VADER sentiment and Bitcoin prices merged successfully.")
print("Daily price rows:", len(dataset))
print("Total tweets:", dataset["tweet_count"].sum())
print("Days with sentiment:", dataset["average_sentiment"].notna().sum())
print("Saved to:", output_file)

print("\nSample:")
print(dataset.head().to_string(index=False))

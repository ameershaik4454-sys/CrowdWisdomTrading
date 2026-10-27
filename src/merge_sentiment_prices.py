
from pathlib import Path
import pandas as pd

prices_file = Path("data/processed/btc_prices_clean.csv")
tweets_file = Path("data/processed/tweets_sentiment.csv")
output_file = Path("data/processed/btc_sentiment_dataset.csv")

prices = pd.read_csv(prices_file)
tweets = pd.read_csv(tweets_file)

# Normalize price dates and tweet timestamps to UTC days
prices["Date"] = pd.to_datetime(prices["Date"], utc=True, errors="coerce")
prices["day"] = prices["Date"].dt.strftime("%Y-%m-%d")

tweets["created_at"] = pd.to_datetime(
    tweets["created_at"], utc=True, errors="coerce", format="mixed"
)
tweets = tweets.dropna(subset=["created_at"])
tweets["day"] = tweets["created_at"].dt.strftime("%Y-%m-%d")

# Summarize tweets for each day
daily_sentiment = (
    tweets.groupby("day")
    .agg(
        tweet_count=("sentiment_score", "size"),
        average_sentiment=("sentiment_score", "mean"),
        positive_tweets=("sentiment_label", lambda s: (s == "positive").sum()),
        negative_tweets=("sentiment_label", lambda s: (s == "negative").sum()),
        total_likes=("likes", "sum"),
        total_reposts=("reposts", "sum"),
    )
    .reset_index()
)

# Merge daily sentiment with price data
merged = prices.merge(daily_sentiment, on="day", how="left")

# Days without collected tweets are unknown, not neutral.
merged["tweet_count"] = merged["tweet_count"].fillna(0).astype(int)

output_file.parent.mkdir(parents=True, exist_ok=True)
merged.to_csv(output_file, index=False)

print("Merge completed!")
print("Price rows:", len(prices))
print("Days with collected tweets:", int((merged["tweet_count"] > 0).sum()))
print("Saved to:", output_file.resolve())
print("\nSample rows:")
print(merged[["day", "Close", "tweet_count", "average_sentiment"]].tail(10))

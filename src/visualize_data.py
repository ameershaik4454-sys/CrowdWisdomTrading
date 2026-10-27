
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------
# 1. File paths
# ----------------------------------
input_file = Path("data/processed/btc_sentiment_dataset.csv")

output_dir = Path("reports/figures")
output_dir.mkdir(parents=True, exist_ok=True)

# ----------------------------------
# 2. Load the merged dataset
# ----------------------------------
df = pd.read_csv(input_file)

# Convert dates and sort chronologically
df["day"] = pd.to_datetime(df["day"], errors="coerce")
df = df.dropna(subset=["day"]).sort_values("day")

# Make sure numeric columns contain numeric values
for column in ["Close", "average_sentiment", "tweet_count"]:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

print("Dataset loaded successfully!")
print("Total price records:", len(df))
print("Days with tweets:", int((df["tweet_count"].fillna(0) > 0).sum()))

# ----------------------------------
# 3. Graph 1: Bitcoin closing price
# ----------------------------------
price_df = df.dropna(subset=["Close"])

plt.figure(figsize=(12, 6))

plt.plot(
    price_df["day"],
    price_df["Close"],
    linewidth=2
)

plt.title("Bitcoin Closing Price Over Time")
plt.xlabel("Date")
plt.ylabel("Closing Price (USD)")
plt.xticks(rotation=45)
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()

plt.savefig(
    output_dir / "bitcoin_closing_price.png",
    dpi=300
)

plt.show()
plt.close()

# ----------------------------------
# 4. Graph 2: Daily average sentiment
# ----------------------------------
sentiment_df = df.dropna(subset=["average_sentiment"])

plt.figure(figsize=(12, 6))

if not sentiment_df.empty:
    plt.plot(
        sentiment_df["day"],
        sentiment_df["average_sentiment"],
        marker="o",
        linestyle="-",
        linewidth=2,
        markersize=8
    )

    plt.axhline(y=0, linestyle="--", linewidth=1)

    plt.title("Daily Average Bitcoin Tweet Sentiment")
    plt.xlabel("Date")
    plt.ylabel("Average Sentiment Score")

    # Display the sentiment value near each point
    for _, row in sentiment_df.iterrows():
        plt.annotate(
            f"{row['average_sentiment']:.4f}",
            (row["day"], row["average_sentiment"]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center"
        )
else:
    plt.text(
        0.5, 0.5,
        "No sentiment data available",
        ha="center",
        va="center",
        transform=plt.gca().transAxes
    )
    plt.title("Daily Average Bitcoin Tweet Sentiment")

plt.xlabel("Date")
plt.xticks(rotation=45)
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()

plt.savefig(
    output_dir / "daily_tweet_sentiment.png",
    dpi=300
)

plt.show()
plt.close()

# ----------------------------------
# 5. Graph 3: Daily tweet volume
# ----------------------------------
tweet_df = df[df["tweet_count"].fillna(0) > 0].copy()

plt.figure(figsize=(12, 6))

if not tweet_df.empty:
    bars = plt.bar(
        tweet_df["day"],
        tweet_df["tweet_count"],
        width=0.6
    )

    plt.title("Daily Bitcoin Tweet Volume")
    plt.xlabel("Date")
    plt.ylabel("Number of Tweets")

    # Display the tweet count above each bar
    plt.bar_label(bars, padding=3)

else:
    plt.text(
        0.5, 0.5,
        "No tweets available",
        ha="center",
        va="center",
        transform=plt.gca().transAxes
    )
    plt.title("Daily Bitcoin Tweet Volume")

plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()

plt.savefig(
    output_dir / "daily_tweet_volume.png",
    dpi=300
)

plt.show()
plt.close()

# ----------------------------------
# 6. Finish
# ----------------------------------
print("\nVisualization completed successfully!")
print("Charts saved in:", output_dir.resolve())
print("1. bitcoin_closing_price.png")
print("2. daily_tweet_sentiment.png")
print("3. daily_tweet_volume.png")


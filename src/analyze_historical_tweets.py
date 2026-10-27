
from pathlib import Path

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# 1. File paths
root = Path(__file__).resolve().parent.parent

input_file = root / "data" / "raw" / "bitcoin_tweets_dataset.csv"
output_dir = root / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "historical_tweets_vader_sentiment.csv"

# 2. Load historical tweets
if not input_file.exists():
    raise FileNotFoundError(f"Cannot find input file: {input_file}")

tweets = pd.read_csv(input_file)

required_columns = ["date", "text"]
missing = [col for col in required_columns if col not in tweets.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

tweets["date"] = pd.to_datetime(tweets["date"], errors="coerce")
tweets["text"] = tweets["text"].fillna("").astype(str).str.strip()

tweets = tweets[
    tweets["date"].notna() & tweets["text"].ne("")
].copy()

if tweets.empty:
    raise ValueError("No usable historical tweets were found.")

# 3. Calculate VADER sentiment
analyzer = SentimentIntensityAnalyzer()

tweets["vader_compound"] = tweets["text"].map(
    lambda text: analyzer.polarity_scores(text)["compound"]
)

def label_sentiment(score):
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    return "neutral"

tweets["vader_label"] = tweets["vader_compound"].map(label_sentiment)

# 4. Save results
tweets.to_csv(output_file, index=False, encoding="utf-8")

# 5. Print summary
print("Historical tweet sentiment analysis completed!")
print("Tweets analyzed:", len(tweets))
print("Date range:", tweets["date"].min(), "to", tweets["date"].max())

print("\nVADER sentiment counts:")
print(
    tweets["vader_label"]
    .value_counts()
    .reindex(["positive", "neutral", "negative"], fill_value=0)
)

print("\nSample results:")
print(
    tweets[["date", "text", "vader_compound", "vader_label"]]
    .head()
    .to_string(index=False)
)

print("\nSaved to:", output_file)

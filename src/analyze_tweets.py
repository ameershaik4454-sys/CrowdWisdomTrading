import json
from pathlib import Path

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ----------------------------------
# 1. Input and output file paths
# ----------------------------------
input_file = Path("data/raw/x_tweets_historical.json")
output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "tweets_sentiment.csv"

# ----------------------------------
# 2. Check and load the JSON file
# ----------------------------------
if not input_file.exists():
    raise FileNotFoundError(
        f"Cannot find {input_file}. "
        "Check that your Apify JSON file is saved in data/raw/."
    )

with input_file.open("r", encoding="utf-8") as file:
    raw_data = json.load(file)

# Handle common JSON structures
if isinstance(raw_data, dict):
    if isinstance(raw_data.get("items"), list):
        raw_data = raw_data["items"]
    elif isinstance(raw_data.get("data"), list):
        raw_data = raw_data["data"]
    else:
        raw_data = [raw_data]

if not isinstance(raw_data, list) or not raw_data:
    raise ValueError("The JSON file contains no tweet records.")

# ----------------------------------
# 3. Convert JSON records to DataFrame
# ----------------------------------
tweets = pd.json_normalize(raw_data, sep="_")

# Rename Apify fields
column_mapping = {
    "createdAt": "created_at",
    "author_userName": "username",
    "author_name": "author_name",
    "likeCount": "likes",
    "retweetCount": "reposts",
    "replyCount": "replies",
    "quoteCount": "quotes",
    "viewCount": "views",
    "twitterUrl": "url",
}

tweets = tweets.rename(columns=column_mapping)

# Check for the tweet text column
if "text" not in tweets.columns:
    if "fullText" in tweets.columns:
        tweets["text"] = tweets["fullText"]
    else:
        raise ValueError(
            "No 'text' or 'fullText' field found in the JSON."
        )

# Use fullText when text is empty
if "fullText" in tweets.columns:
    full_text = tweets["fullText"].fillna("").astype(str)
    short_text = tweets["text"].fillna("").astype(str)
    tweets["text"] = short_text.where(
        short_text.str.strip() != "", full_text
    )

# Keep useful columns that actually exist
wanted_columns = [
    "text",
    "created_at",
    "username",
    "author_name",
    "likes",
    "reposts",
    "replies",
    "quotes",
    "views",
    "url",
]

available_columns = [
    column for column in wanted_columns
    if column in tweets.columns
]
tweets = tweets[available_columns].copy()

# Remove empty tweet text
tweets["text"] = tweets["text"].fillna("").astype(str)
tweets = tweets[tweets["text"].str.strip() != ""].copy()

if tweets.empty:
    raise ValueError("No tweets with usable text were found.")

# ----------------------------------
# 4. Convert timestamps to UTC
# ----------------------------------
if "created_at" in tweets.columns:
    tweets["created_at"] = pd.to_datetime(
        tweets["created_at"],
        format="mixed",
        utc=True,
        errors="coerce",
    )

    invalid_dates = tweets["created_at"].isna().sum()
    if invalid_dates:
        print(
            f"Warning: {invalid_dates} tweets have invalid timestamps."
        )

# ----------------------------------
# 5. Calculate VADER sentiment
# ----------------------------------
analyzer = SentimentIntensityAnalyzer()

tweets["sentiment_score"] = tweets["text"].apply(
    lambda text: analyzer.polarity_scores(text)["compound"]
)

def label_sentiment(score):
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    return "neutral"

tweets["sentiment_label"] = tweets["sentiment_score"].apply(
    label_sentiment
)

# ----------------------------------
# 6. Save the processed data
# ----------------------------------
tweets.to_csv(output_file, index=False, encoding="utf-8")

# ----------------------------------
# 7. Display results
# ----------------------------------
print("Tweet sentiment analysis completed!")
print("Input file:", input_file.resolve())
print("Number of tweets:", len(tweets))

print("\nSentiment counts:")
print(
    tweets["sentiment_label"]
    .value_counts()
    .reindex(["positive", "neutral", "negative"], fill_value=0)
)

if "created_at" in tweets.columns:
    valid_dates = tweets["created_at"].dropna()

    if not valid_dates.empty:
        print("\nEarliest tweet:", valid_dates.min())
        print("Latest tweet:", valid_dates.max())

print("\nSample results:")
print(
    tweets[["text", "sentiment_score", "sentiment_label"]]
    .head()
    .to_string(index=False)
)

print("\nSaved to:", output_file.resolve())


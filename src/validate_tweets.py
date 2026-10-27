
import json
import pandas as pd
from pathlib import Path

# Load historical tweets
file_path = Path("data/raw/x_tweets_historical.json")

with open(file_path, "r", encoding="utf-8") as file:
    tweets = json.load(file)

# Convert JSON records into a DataFrame
df = pd.DataFrame(tweets)

print("Total tweets:", len(df))
print("Available columns:", df.columns.tolist())

# Check required columns
if "createdAt" not in df.columns:
    raise ValueError("The file has no 'createdAt' column.")

# Convert tweet dates
df["createdAt"] = pd.to_datetime(
    df["createdAt"], errors="coerce", utc=True
)

# Remove records with missing text or invalid dates
df = df.dropna(subset=["createdAt", "text"])

# Remove duplicates
df = df.drop_duplicates(subset=["id"])

print("\nEarliest tweet:", df["createdAt"].min())
print("Latest tweet:", df["createdAt"].max())

print("\nTweets per day:")
print(df.groupby(df["createdAt"].dt.date).size())

# Save validated data
output_path = Path(
    "data/processed/historical_tweets_validated.csv"
)
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

print("\nValidated data saved successfully!")

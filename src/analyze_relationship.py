
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------
# 1. File paths
# ----------------------------------
input_file = Path(
    "data/processed/btc_vader_sentiment_dataset_2022_2023.csv"
)
output_dir = Path("data/processed")
figures_dir = Path("reports/figures")

output_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)

results_file = output_dir / "price_sentiment_analysis.csv"

# ----------------------------------
# 2. Load the merged dataset
# ----------------------------------
df = pd.read_csv(input_file)

required_columns = [
    "date",
    "close",
    "average_sentiment",
    "tweet_count",
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["close"] = pd.to_numeric(df["close"], errors="coerce")
df["average_sentiment"] = pd.to_numeric(
    df["average_sentiment"], errors="coerce"
)
df["tweet_count"] = pd.to_numeric(
    df["tweet_count"], errors="coerce"
)

df = (
    df.dropna(subset=["date", "close"])
    .sort_values("date")
    .drop_duplicates(subset=["date"], keep="last")
    .reset_index(drop=True)
)

# ----------------------------------
# 3. Calculate daily Bitcoin returns
# ----------------------------------
# Percentage change in closing price from the previous date.
df["daily_return_pct"] = df["close"].pct_change() * 100

# ----------------------------------
# 4. Select dates with sentiment data
# ----------------------------------
analysis_df = df.dropna(
    subset=["average_sentiment", "daily_return_pct"]
).copy()

# Save the calculated data for inspection.
df.to_csv(results_file, index=False)

print("Price and sentiment analysis")
print("-" * 40)
print("Total price dates:", len(df))
print("dates with tweet sentiment:", int(
    df["average_sentiment"].notna().sum()
))
print("dates available for comparison:", len(analysis_df))

print("\nSample calculated data:")
print(
    df[
        [
            "date",
            "close",
            "daily_return_pct",
            "tweet_count",
            "average_sentiment",
        ]
    ].tail(10).to_string(index=False)
)

# ----------------------------------
# 5. Plot daily Bitcoin returns
# ----------------------------------
return_df = df.dropna(subset=["daily_return_pct"])

plt.figure(figsize=(12, 6))

plt.plot(
    return_df["date"],
    return_df["daily_return_pct"],
    marker="o",
    markersize=3,
    linewidth=1.5,
)

plt.axhline(y=0, linestyle="--", linewidth=1)
plt.title("Daily Bitcoin Price Change (%)")
plt.xlabel("Date")
plt.ylabel("Daily Return (%)")
plt.xticks(rotation=45)
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()

plt.savefig(
    figures_dir / "bitcoin_daily_returns.png",
    dpi=300,
)
plt.show()
plt.close()

# ----------------------------------
# 6. Plot sentiment versus price returns
# ----------------------------------
plt.figure(figsize=(9, 6))

if not analysis_df.empty:
    plt.scatter(
        analysis_df["average_sentiment"],
        analysis_df["daily_return_pct"],
        s=80,
    )

    plt.axhline(y=0, linestyle="--", linewidth=1)
    plt.axvline(x=0, linestyle="--", linewidth=1)

    plt.title("Tweet Sentiment vs Bitcoin Daily Return")
    plt.xlabel("Average Tweet Sentiment")
    plt.ylabel("Bitcoin Daily Return (%)")

    if len(analysis_df) == 1:
        plt.annotate(
            "Only one date available",
            (
                analysis_df["average_sentiment"].iloc[0],
                analysis_df["daily_return_pct"].iloc[0],
            ),
            xytext=(10, 10),
            textcoords="offset points",
        )
else:
    plt.text(
        0.5,
        0.5,
        "No matching sentiment and return data",
        ha="center",
        va="center",
        transform=plt.gca().transAxes,
    )
    plt.title("Tweet Sentiment vs Bitcoin Daily Return")

plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()

plt.savefig(
    figures_dir / "sentiment_vs_bitcoin_returns.png",
    dpi=300,
)
plt.show()
plt.close()

# ----------------------------------
# 7. Calculate correlation cautiously
# ----------------------------------
print("\nCorrelation analysis")
print("-" * 40)

if len(analysis_df) >= 2:
    correlation = analysis_df[
        ["average_sentiment", "daily_return_pct"]
    ].corr().iloc[0, 1]

    if pd.notna(correlation):
        print(
            "Pearson correlation between sentiment "
            "and daily Bitcoin returns:",
            round(correlation, 4),
        )
        print(
            "This measures linear association, not causation."
        )
    else:
        print(
            "Correlation is undefined because one variable "
            "has no variation."
        )
else:
    print(
        "Not enough matching dates to calculate a meaningful "
        "correlation. At least two dates are required, and "
        "more observations are recommended."
    )

# ----------------------------------
# 8. Finish
# ----------------------------------
print("\nAnalysis completed!")
print("Calculated dataset:", results_file.resolve())
print("Charts saved in:", figures_dir.resolve())
print("1. bitcoin_daily_returns.png")
print("2. sentiment_vs_bitcoin_returns.png")

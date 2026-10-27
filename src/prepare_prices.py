
import pandas as pd
from pathlib import Path

# File locations
input_file = Path("data/raw/btc_prices.csv")
output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

# Load the downloaded data
prices = pd.read_csv(input_file, header=[0, 1], index_col=0)

# Simplify the column names
prices.columns = prices.columns.get_level_values(0)

# Remove duplicate column names, if any
prices = prices.loc[:, ~prices.columns.duplicated()]

# Convert the index to dates
prices.index = pd.to_datetime(prices.index)
prices.index.name = "Date"

# Sort dates and remove duplicate dates
prices = prices.sort_index()
prices = prices[~prices.index.duplicated(keep="last")]

# Keep the columns needed for our project
prices = prices[["Open", "High", "Low", "Close", "Volume"]]

# Remove rows missing essential values
prices = prices.dropna(subset=["Close"])

# Add daily percentage return
prices["daily_return"] = prices["Close"].pct_change()

# Save the cleaned data
output_file = output_dir / "btc_prices_clean.csv"
prices.to_csv(output_file)

print("Price data prepared successfully!")
print("Number of rows:", len(prices))
print("\nColumns:")
print(prices.columns.tolist())
print("\nFirst five rows:")
print(prices.head())
print("\nSaved to:", output_file.resolve())

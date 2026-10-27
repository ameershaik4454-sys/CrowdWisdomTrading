
import yfinance as yf
from pathlib import Path

print("Step 1: Downloading historical Bitcoin prices...")

try:
    prices = yf.download(
        "BTC-USD",
        start="2022-01-01",
        end="2023-06-23",
        interval="1d",
        progress=False,
        auto_adjust=False,
        threads=False,
        timeout=30
    )

    if prices.empty:
        print("ERROR: No price data returned.")
        print("Check your internet connection or Yahoo Finance access.")

    else:
        # Handle possible multi-level column names from yfinance.
        if isinstance(prices.columns, __import__("pandas").MultiIndex):
            prices.columns = prices.columns.get_level_values(0)

        prices = prices.reset_index()

        # Keep only the date and closing price.
        prices = prices[["Date", "Close"]].rename(
            columns={"Date": "date", "Close": "close"}
        )

        prices["date"] = prices["date"].dt.strftime("%Y-%m-%d")

        output_dir = Path("data/raw")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "btc_prices_2022_2023.csv"
        prices.to_csv(output_file, index=False)

        print("SUCCESS! Bitcoin prices downloaded.")
        print("Number of rows:", len(prices))
        print("First date:", prices["date"].min())
        print("Last date:", prices["date"].max())
        print("File saved at:", output_file.resolve())
        print(prices.head())

except Exception as error:
    print("DOWNLOAD FAILED!")
    print("Error details:", repr(error))

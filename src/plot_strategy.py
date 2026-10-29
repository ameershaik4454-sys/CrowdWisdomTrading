
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

INPUT = Path("data/processed/strategy_daily_returns.csv")
OUTPUT = Path("data/processed/strategy_vs_buy_hold.png")

df = pd.read_csv(INPUT)
df["date"] = pd.to_datetime(df["date"])

plt.figure(figsize=(12, 6))
plt.plot(
    df["date"], df["strategy_equity_USD"],
    label="Long-or-cash model strategy"
)
plt.plot(
    df["date"], df["buy_hold_equity_USD"],
    label="Buy-and-hold benchmark"
)

plt.title("Bitcoin Strategy vs Buy-and-Hold")
plt.xlabel("Date")
plt.ylabel("Portfolio Value (USD)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT, dpi=150)
plt.close()

print(f"Chart saved to {OUTPUT}")

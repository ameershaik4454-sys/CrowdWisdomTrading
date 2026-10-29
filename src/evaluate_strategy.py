
from pathlib import Path

import numpy as np
import pandas as pd

INPUT = Path("data/processed/walk_forward_predictions.csv")
OUTPUT = Path("data/processed/strategy_evaluation.csv")
DETAIL_OUTPUT = Path("data/processed/strategy_daily_returns.csv")

INITIAL_CAPITAL = 10_000.0
ANNUALIZATION = 365
TRADING_COST_BPS = 10  # Assumed cost per unit of position change

df = pd.read_csv(INPUT)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["horizon_days"] = pd.to_numeric(df["horizon_days"], errors="coerce")

df = df[df["horizon_days"] == 1].copy()
df = df.sort_values("date").dropna(
    subset=["date", "actual_future_return_pct",
            "predicted_future_return_pct"]
)

if df.empty:
    raise ValueError("No valid one-day predictions found.")

if df["date"].duplicated().any():
    raise ValueError("Duplicate dates found in one-day predictions.")

# Forecast made at the close on date t; its target is the
# return from close t to close t+1.
df["market_return"] = df["actual_future_return_pct"] / 100.0

# Long if predicted return is positive; otherwise remain in cash.
df["position"] = (
    df["predicted_future_return_pct"] > 0
).astype(float)

# Charge costs when entering or exiting the long position.
df["position_change"] = df["position"].diff().abs()
df.loc[df.index[0], "position_change"] = abs(df["position"].iloc[0])

df["transaction_cost"] = (
    df["position_change"] * TRADING_COST_BPS / 10_000.0
)

df["strategy_return"] = (
    df["position"] * df["market_return"] - df["transaction_cost"]
)

# Buy-and-hold benchmark across the same future-return intervals.
df["buy_hold_return"] = df["market_return"]

def calculate_metrics(returns, label):
    returns = pd.Series(returns, dtype=float).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    if len(returns) < 2:
        raise ValueError(f"Not enough observations for {label}.")

    mean_return = returns.mean()
    volatility = returns.std(ddof=1)
    downside_deviation = np.sqrt(
        np.mean(np.minimum(returns.to_numpy(), 0.0) ** 2)
    )

    sharpe = (
        np.sqrt(ANNUALIZATION) * mean_return / volatility
        if volatility > 0 else np.nan
    )
    sortino = (
        np.sqrt(ANNUALIZATION) * mean_return / downside_deviation
        if downside_deviation > 0 else np.nan
    )

    equity = INITIAL_CAPITAL * (1 + returns).cumprod()
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1

    return {
        "strategy": label,
        "test_days": len(returns),
        "initial_capital_USD": INITIAL_CAPITAL,
        "final_capital_USD": equity.iloc[-1],
        "total_return_pct": (equity.iloc[-1] / INITIAL_CAPITAL - 1) * 100,
        "sharpe_ratio_annualized": sharpe,
        "sortino_ratio_annualized": sortino,
        "maximum_drawdown_pct": drawdown.min() * 100,
    }

strategy_metrics = calculate_metrics(
    df["strategy_return"], "Long-or-cash, 1-day forecast"
)
benchmark_metrics = calculate_metrics(
    df["buy_hold_return"], "Buy-and-hold benchmark"
)

results = pd.DataFrame([strategy_metrics, benchmark_metrics])

df["strategy_equity_USD"] = (
    INITIAL_CAPITAL * (1 + df["strategy_return"]).cumprod()
)
df["buy_hold_equity_USD"] = (
    INITIAL_CAPITAL * (1 + df["buy_hold_return"]).cumprod()
)

results["transaction_cost_bps"] = [TRADING_COST_BPS, 0]
results["exposure_pct"] = [
    df["position"].mean() * 100,
    100.0,
]
results["position_entries"] = [
    int(((df["position"] == 1) &
         (df["position"].shift(1).fillna(0) == 0)).sum()),
    1,
]

results.to_csv(OUTPUT, index=False)
df.to_csv(DETAIL_OUTPUT, index=False)

print("\nSTRATEGY AND BENCHMARK RESULTS")
print(results.round(4).to_string(index=False))
print(f"\nSaved summary: {OUTPUT}")
print(f"Saved daily details: {DETAIL_OUTPUT}")

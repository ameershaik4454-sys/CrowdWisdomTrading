# Bitcoin Sentiment Price Prediction

A Python research project exploring the relationship between Bitcoin
prices and sentiment from Bitcoin-related tweets. It combines historical
tweet sentiment and price features, generates 1--5 day return forecasts
with a Random Forest model, evaluates forecasts using walk-forward
validation, and compares a simple long-or-cash strategy with
buy-and-hold.

**Repository:** https://github.com/ameershaik4454-sys/CrowdWisdomTrading

## Project scope

-   Analyze historical Bitcoin tweet sentiment and daily price data.
-   Create daily sentiment and tweet-volume features.
-   Forecast cumulative Bitcoin returns at horizons of 1, 2, 3, 4, and 5
    days.
-   Evaluate predictions with MAE, RMSE, directional accuracy, and price
    MAE.
-   Evaluate a one-day long-or-cash strategy using transaction costs and
    compare it with buy-and-hold.
-   Generate charts for price, returns, tweet sentiment, tweet volume,
    and strategy equity curves.

## Technologies

-   Python
-   pandas and NumPy
-   scikit-learn (Random Forest regression)
-   VADER Sentiment
-   Matplotlib
-   yfinance (for price-data retrieval scripts)

## Installation

Use Python 3.12 or a compatible Python version. From the project root,
run:

``` powershell
python -m pip install -r requirements.txt
```

## Run the pipeline

Run these commands from the repository root, in order:

``` powershell
python src\predict_prices.py
python src\evaluate_strategy.py
python src\plot_strategy.py
```

To run the existing sentiment/price relationship analysis:

``` powershell
python src\analyze_relationship.py
```

The top-level `main.py` is also present for the earlier project
workflow.

## Key outputs

Prediction metrics and forecasts:

-   `data/processed/walk_forward_model_metrics.csv`
-   `data/processed/walk_forward_predictions.csv`

Strategy evaluation:

-   `data/processed/strategy_evaluation.csv`
-   `data/processed/strategy_daily_returns.csv`
-   `data/processed/strategy_vs_buy_hold.png`

Other existing visualizations include Bitcoin closing price, daily
returns, daily tweet sentiment, daily tweet volume, and sentiment versus
Bitcoin returns. Depending on the script, charts may be saved under
`data/processed/` or `reports/figures/`.

## Current historical backtest snapshot

The saved walk-forward evaluation contains 104--105 test predictions per
horizon. Directional accuracy was 49.52%, 53.33%, 54.29%, 50.00%, and
59.62% for horizons +1 through +5 days respectively.

In the one-day strategy test, the long-or-cash strategy returned 37.91%,
compared with 46.90% for buy-and-hold. The strategy's reported maximum
drawdown was −14.76%, compared with −17.59% for buy-and-hold. These
results are historical and depend on the selected data period and
assumptions; they are not evidence of guaranteed future performance.

For full methodology, metrics, limitations, and interpretation, see
[`EVALUATION_REPORT.md`](EVALUATION_REPORT.md).

## Data scope and limitations

The current modeling dataset contains 538 daily observations from 1
January 2022 through 22 June 2023. It does **not** represent the most
recent 180 days of X posts. Recent-data collection and external news/API
integrations are not included in the current evaluation. The results are
preliminary, and more recent out-of-sample validation is needed.

Correlation does not imply causation. This is an educational research
prototype, not financial advice or a production-ready trading system.

## Credentials and security

Do not commit API tokens or passwords. Store any credentials in
environment variables or a local `.env` file excluded by `.gitignore`.

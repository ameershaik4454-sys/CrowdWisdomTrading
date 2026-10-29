# Bitcoin Sentiment Price Prediction --- Evaluation Report

**Project:** CrowdWisdomTrading --- X Sentiment Price Prediction Agent\
**Evaluation date:** 29 October 2026\
**Repository:** https://github.com/ameershaik4454-sys/CrowdWisdomTrading

## 1. Executive summary

This project explores whether daily Bitcoin sentiment, historical
Bitcoin prices, and return features can help predict Bitcoin returns
over one to five days. A Random Forest regression model is evaluated
using expanding-window walk-forward validation. A simple one-day
long-or-cash strategy is compared with buy-and-hold.

The strategy produced a **37.91% total return**, versus **46.90%** for
buy-and-hold. It reported a lower maximum drawdown (**−14.76%** versus
**−17.59%**) and higher annualized Sharpe and Sortino ratios. These
historical results are sample-dependent and do not guarantee future
profitability.

## 2. Data and scope

The modeling dataset contains **538 daily observations from 1 January
2022 through 22 June 2023**, combining Bitcoin closing prices with daily
tweet sentiment and volume features. Fields include average sentiment,
tweet count, positive/neutral/negative counts, closing price, and daily
returns.

**Scope limitation:** this dataset does not represent the most recent
180 days of X posts. Recent-data collection and external news/API
integrations were not completed for this evaluation, so conclusions are
limited to the historical dataset.

## 3. Modeling methodology

-   **Model:** `RandomForestRegressor`.
-   **Forecast horizons:** 1, 2, 3, 4, and 5 days ahead.
-   **Target:** cumulative return from the forecast-origin close to the
    close at the selected future horizon.
-   **Features:** lagged sentiment and tweet-volume/count features;
    lagged daily returns; rolling return statistics over 3, 7, and 14
    days.
-   **Validation:** expanding-window walk-forward testing, with periodic
    refitting.
-   **Leakage safeguard:** training rows are restricted so each training
    target is already observable at the forecast origin.
-   **Metrics:** return MAE, return RMSE, directional accuracy, and
    price MAE.

## 4. Walk-forward prediction results

  -------------------------------------------------------------------------------
  Forecast            Test    Return MAE   Return RMSE   Directional    Price MAE
  horizon      predictions   (percentage   (percentage      accuracy        (USD)
                                 points)       points)               
  ---------- ------------- ------------- ------------- ------------- ------------
  +1 day               105        1.9292        2.6944        49.52%     \$518.80

  +2 days              105        2.6647        4.0161        53.33%     \$700.30

  +3 days              105        3.4090        5.0772        54.29%     \$887.01

  +4 days              104        4.1656        6.1074        50.00%   \$1,084.46

  +5 days              104        4.5042        6.7527        59.62%   \$1,164.44
  -------------------------------------------------------------------------------

Errors increase at longer horizons. Directional accuracy varies; the
+1-day result of 49.52% is close to chance. These results do not
establish robust predictive skill, and testing on more recent unseen
data is needed.

## 5. Trading strategy and benchmark

The strategy holds Bitcoin when the one-day predicted return is positive
and otherwise holds cash. Transaction costs are **10 basis points
(0.10%) per unit of position change**. The benchmark is buy-and-hold
over the same 105 test observations. Initial capital is \$10,000.

  Metric                       Long-or-cash strategy   Buy-and-hold benchmark
  -------------------------- ----------------------- ------------------------
  Test days                                      105                      105
  Initial capital                        \$10,000.00              \$10,000.00
  Final capital                          \$13,790.73              \$14,689.51
  Total return                                37.91%                   46.90%
  Annualized Sharpe ratio                       3.06                     2.92
  Annualized Sortino ratio                      6.39                     5.50
  Maximum drawdown                           −14.76%                  −17.59%
  Exposure                                    61.90%                  100.00%
  Position entries                                20                        1

The strategy underperformed buy-and-hold in total return by
approximately 8.99 percentage points, while showing a smaller maximum
drawdown and higher reported risk-adjusted ratios. Metrics depend on the
test period, annualization convention, transaction-cost assumptions, and
implementation details; verify them against the daily-return series
before making investment claims.

## 6. Visual outputs

The repository contains these charts:

-   `data/processed/strategy_vs_buy_hold.png` --- strategy versus
    buy-and-hold equity curves.
-   `data/processed/bitcoin_closing_price.png` --- Bitcoin closing
    price.
-   `data/processed/bitcoin_daily_returns.png` --- daily returns.
-   `data/processed/daily_tweet_sentiment.png` --- daily sentiment.
-   `data/processed/daily_tweet_volume.png` --- daily tweet volume.
-   `data/processed/sentiment_vs_bitcoin_returns.png` --- sentiment and
    Bitcoin-return relationship.

The previously calculated Pearson correlation between sentiment and
daily Bitcoin returns was **0.0947**, a weak positive linear
association. Correlation does not establish causation or predictive
usefulness.

## 7. Reproducibility

From the repository root, run in this order:

``` powershell
python src\predict_prices.py
python src\evaluate_strategy.py
python src\plot_strategy.py
```

Prediction outputs: - `data/processed/walk_forward_model_metrics.csv` -
`data/processed/walk_forward_predictions.csv`

Strategy outputs: - `data/processed/strategy_evaluation.csv` -
`data/processed/strategy_daily_returns.csv` -
`data/processed/strategy_vs_buy_hold.png`

Keep API credentials in environment variables or a local `.env` file
excluded from version control. Never include live tokens in the report
or repository.

## 8. Limitations and next steps

1.  The modeling period is historical (2022--2023), not the required
    recent 180-day X dataset.
2.  There are only 104--105 test predictions per horizon, so metrics may
    be unstable.
3.  Random Forest may not capture market regime changes or sudden
    shocks; these results do not establish causation from sentiment.
4.  The strategy does not fully model slippage, taxes, or all execution
    constraints.
5.  Total return was below buy-and-hold in this test.
6.  Recent social-media collection, news enrichment, and validation on a
    genuinely recent out-of-sample period remain future work.

## 9. Conclusion

The project has a historical sentiment/price modeling pipeline,
walk-forward forecasts for five horizons, a one-day strategy evaluation,
and visualization outputs. In this test, the strategy showed lower
drawdown and higher reported Sharpe and Sortino ratios than
buy-and-hold, but lower total return. Findings are preliminary and must
be presented alongside the data and validation limitations.

**Responsible-use note:** This is an educational research prototype, not
financial advice or a production-ready trading system.

# Stock Market Analysis — Next-Day Direction Prediction

Technical analytics, walk-forward model evaluation, and cost-aware backtesting
for equity price data, served through a Streamlit app.

## Contents

```
app.ipynb                  full analysis walkthrough (EDA → features → models → backtest)
app.py                     Streamlit app
train_model.py             trains and persists the model artifacts
stock_model.joblib         fitted classifier
scaler.joblib              fitted StandardScaler
feature_metadata.joblib    feature order, training window, metrics
requirements.txt
src/
  data/fetch.py            yfinance loader with Parquet cache + synthetic fallback
  features/build_features.py   24 lag-safe time-series features
  models/train.py          walk-forward evaluation, MLflow tracking
  backtest/engine.py       signal → returns, Sharpe, drawdown, costs
tests/test_pipeline.py     16 tests incl. leakage guard + negative control
```

## Quick start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python train_model.py --ticker AAPL     # produces the three .joblib artifacts
streamlit run app.py                    # launch the dashboard
jupyter notebook app.ipynb              # or read the full walkthrough
pytest tests/ -v
```

The committed `.joblib` files were trained on **synthetic data** as placeholders so the
app runs out of the box. Retrain on a real ticker before reading anything into the
predictions — the app shows a warning until you do.

## Why it's built this way

Most published stock-prediction projects report accuracy figures that don't survive
contact with reality. Two mistakes account for nearly all of it, and this repo is
structured to make both impossible.

**1. Shuffled cross-validation leaks the future.** Ordinary k-fold CV puts tomorrow in
the training set and yesterday in the test set. Every evaluation here uses
`TimeSeriesSplit` expanding-window walk-forward validation. `tests/test_pipeline.py`
asserts that truncating the price history leaves earlier feature values unchanged — if
any feature peeked forward, that test fails.

**2. Accuracy is not profit.** A model can be right 55% of the time and lose money if
it's wrong on the days that move most. The backtest converts the signal into a return
stream, charges a configurable cost on every position change, and compares against
buy-and-hold on the same window.

## Features

24 features computed from information available at or before the close of day *t*,
predicting the return realised on day *t+1*:

| Group | Features |
| --- | --- |
| Momentum | Lagged returns (1, 2, 3, 5, 10d), 5d and 21d cumulative |
| Trend | Close vs SMA (5, 10, 20, 50), EMA 12/26 ratio |
| Volatility | 10d and 21d realised vol, vol ratio, high–low range |
| Oscillators | RSI(14), MACD, MACD histogram, Bollinger position |
| Volume | Volume vs 20d SMA, volume change |
| Calendar | Day of week, month |

## Models

`baseline_majority` (the bar to clear), `logistic_regression`, `random_forest`,
`gradient_boosting`. The baseline is reported deliberately — daily equity direction is
close to a coin flip, and a model that doesn't beat it isn't worth deploying.

## Results

Accuracy lands marginally above the majority-class baseline, and the edge frequently
does **not** survive transaction costs. That's a real finding, not a failure of the
implementation: daily direction prediction from price history alone sits close to the
efficient-market limit. The value here is the evaluation discipline — leakage guard,
baseline comparison, negative control, cost sensitivity — not a profitable strategy.

On synthetic random-walk data, every model lands at the baseline, which is the correct
outcome and confirms the pipeline isn't leaking.

## Notes

Market data from Yahoo Finance via `yfinance`, for educational purposes.
**Not financial advice.** MIT licensed.

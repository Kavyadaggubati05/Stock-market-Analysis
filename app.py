"""Streamlit app for the stock market analysis and direction model.

Loads the persisted artifacts produced by `train_model.py` and serves
predictions plus the analytics dashboard. Run with:

    streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.backtest.engine import backtest, score_returns
from src.data.fetch import fetch_prices, synthetic_ohlcv
from src.features.build_features import build_features, feature_columns
from src.models.train import compare_models
from train_model import load_artifacts, predict_latest

st.set_page_config(page_title="Stock Market Analysis", page_icon="📈", layout="wide")


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_prices(ticker: str, start: str, offline: bool) -> pd.DataFrame:
    if offline:
        return synthetic_ohlcv()
    return fetch_prices(ticker, start=start)


@st.cache_resource(show_spinner=False)
def get_artifacts():
    return load_artifacts()


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.title("Configuration")
offline = st.sidebar.checkbox("Offline mode (synthetic data)", value=False)
ticker = st.sidebar.text_input("Ticker", value="AAPL", disabled=offline)
start = st.sidebar.date_input("History from", value=pd.Timestamp("2018-01-01"))
cost_bps = st.sidebar.slider("Transaction cost (bps)", 0.0, 50.0, 5.0, 1.0)
n_splits = st.sidebar.slider("Walk-forward folds", 3, 8, 5)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Not financial advice. Educational project — see the README for why the "
    "evaluation is built the way it is."
)

st.title("📈 Stock Market Analysis")
st.caption("Technical analytics, walk-forward model evaluation, and cost-aware backtesting.")

try:
    prices = load_prices(ticker.upper(), str(start), offline)
except Exception as exc:  # noqa: BLE001 - surface any fetch failure to the user
    st.error(f"Could not load price data: {exc}")
    st.stop()

features = build_features(prices)
label = "SYNTHETIC" if offline else ticker.upper()

tab_overview, tab_model, tab_backtest, tab_predict = st.tabs(
    ["Overview", "Model evaluation", "Backtest", "Prediction"]
)


# --------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------
with tab_overview:
    returns = prices["close"].pct_change().dropna()
    stats = score_returns(returns)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last close", f"${prices['close'].iloc[-1]:,.2f}")
    c2.metric("Annualised return", f"{stats.annualised_return:.1%}")
    c3.metric("Annualised volatility", f"{stats.annualised_volatility:.1%}")
    c4.metric("Max drawdown", f"{stats.max_drawdown:.1%}")

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.04
    )
    fig.add_trace(
        go.Candlestick(
            x=prices.index,
            open=prices["open"],
            high=prices["high"],
            low=prices["low"],
            close=prices["close"],
            name="Price",
        ),
        row=1,
        col=1,
    )
    for window in (20, 50):
        fig.add_trace(
            go.Scatter(
                x=prices.index,
                y=prices["close"].rolling(window).mean(),
                name=f"SMA {window}",
                line=dict(width=1),
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Bar(x=prices.index, y=prices["volume"], name="Volume", marker_color="lightslategray"),
        row=2,
        col=1,
    )
    fig.update_layout(height=600, xaxis_rangeslider_visible=False, margin=dict(t=30))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Engineered features"):
        st.write(f"{len(feature_columns(features))} features across {len(features)} rows.")
        st.dataframe(features[feature_columns(features)].tail(10), use_container_width=True)


# --------------------------------------------------------------------------
# Model evaluation
# --------------------------------------------------------------------------
with tab_model:
    st.subheader("Walk-forward leaderboard")
    st.markdown(
        "Models are evaluated with expanding-window walk-forward validation, so every "
        "prediction comes from a model that only saw earlier data. **The majority-class "
        "baseline is the bar to clear** — daily direction is close to a coin flip, and a "
        "model that doesn't beat the baseline isn't worth deploying."
    )

    with st.spinner("Running walk-forward evaluation..."):
        board = compare_models(features, n_splits=n_splits)

    baseline = board.loc[board["model"] == "baseline_majority", "mean_accuracy"].iloc[0]
    st.dataframe(
        board.style.format(
            {
                "mean_accuracy": "{:.4f}",
                "std_accuracy": "{:.4f}",
                "mean_f1": "{:.4f}",
                "mean_roc_auc": "{:.4f}",
            }
        ).background_gradient(subset=["mean_accuracy"], cmap="Blues"),
        use_container_width=True,
    )

    best = board.iloc[0]
    edge = best["mean_accuracy"] - baseline
    if edge < 0.01:
        st.warning(
            f"Best model beats the baseline by only {edge:+.2%}. On daily equity direction "
            "this is the expected result, not a bug — price history alone carries very "
            "little next-day signal."
        )
    else:
        st.info(f"Best model beats the baseline by {edge:+.2%}. Check it survives costs in the backtest.")


# --------------------------------------------------------------------------
# Backtest
# --------------------------------------------------------------------------
with tab_backtest:
    st.subheader("Signal backtest vs buy-and-hold")
    model_choice = st.selectbox(
        "Model", ["gradient_boosting", "random_forest", "logistic_regression"]
    )

    with st.spinner("Backtesting..."):
        bt = backtest(features, model_name=model_choice, n_splits=n_splits, cost_bps=cost_bps)

    strategy, benchmark = bt["strategy"], bt["buy_and_hold"]
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Strategy return",
        f"{strategy.total_return:.1%}",
        delta=f"{bt['excess_return']:+.1%} vs hold",
    )
    c2.metric("Strategy Sharpe", f"{strategy.sharpe_ratio:.2f}", delta=f"{benchmark.sharpe_ratio:.2f} hold")
    c3.metric("Time in market", f"{bt['days_in_market']:.0%}")

    curve = pd.DataFrame(
        {"Strategy": strategy.equity_curve, "Buy and hold": benchmark.equity_curve}
    )
    st.plotly_chart(
        go.Figure(
            [go.Scatter(x=curve.index, y=curve[c], name=c) for c in curve.columns]
        ).update_layout(height=420, yaxis_title="Growth of $1", margin=dict(t=30)),
        use_container_width=True,
    )

    st.dataframe(
        pd.DataFrame({"Strategy": strategy.summary(), "Buy and hold": benchmark.summary()}),
        use_container_width=True,
    )
    st.caption(
        f"Costs of {cost_bps:.0f} bps are charged on every position change. Raise the slider "
        "to see how quickly a thin edge disappears."
    )


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
with tab_predict:
    st.subheader("Next-day direction")

    try:
        model, scaler, metadata = get_artifacts()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    if metadata.get("is_synthetic"):
        st.warning(
            "The loaded model was trained on **synthetic data** as a placeholder. "
            "Retrain on a real ticker before reading anything into the prediction:\n\n"
            "```\npython train_model.py --ticker AAPL\n```"
        )

    try:
        prediction = predict_latest(features, model, scaler, metadata)
    except ValueError as exc:
        st.error(f"Feature contract violated — retrain the model.\n\n{exc}")
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Signal", prediction["direction"])
    c2.metric(
        "Model confidence",
        f"{prediction['confidence']:.1%}" if prediction["confidence"] is not None else "n/a",
    )
    c3.metric("As of", prediction["as_of"])

    st.markdown("**Model card**")
    wf, bt_meta = metadata["walk_forward"], metadata["backtest"]
    st.json(
        {
            "trained_on": metadata["ticker"],
            "model": metadata["model_name"],
            "trained_at": metadata["trained_at"],
            "training_window": f"{metadata['train_start']} to {metadata['train_end']}",
            "rows": metadata["n_rows"],
            "walk_forward_accuracy": round(wf["mean_accuracy"], 4),
            "walk_forward_roc_auc": round(wf["mean_roc_auc"], 4),
            "backtest_sharpe": round(bt_meta["strategy_sharpe"], 3),
            "benchmark_sharpe": round(bt_meta["benchmark_sharpe"], 3),
        }
    )
    st.caption(
        "Confidence is the model's predicted probability, not a calibrated likelihood, and "
        "an accuracy near 50% means this signal should not be traded."
    )

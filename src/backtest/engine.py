"""Turn model predictions into a trading signal and score it honestly.

Accuracy is a poor proxy for whether a signal is useful. A model can be
right 55% of the time and still lose money if it is wrong on the days that
move most, or if trading costs eat the edge. This module converts
walk-forward predictions into a return stream and compares it against
buy-and-hold on the same window.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.features.build_features import feature_columns
from src.models.train import MODELS

TRADING_DAYS = 252


@dataclass
class BacktestResult:
    total_return: float
    annualised_return: float
    annualised_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    hit_rate: float
    n_trades: int
    equity_curve: pd.Series

    def summary(self) -> dict:
        return {
            "total_return": self.total_return,
            "annualised_return": self.annualised_return,
            "annualised_volatility": self.annualised_volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "hit_rate": self.hit_rate,
            "n_trades": self.n_trades,
        }


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough decline of an equity curve, as a negative number."""
    running_peak = equity.cummax()
    return float((equity / running_peak - 1.0).min())


def score_returns(returns: pd.Series, risk_free_rate: float = 0.0) -> BacktestResult:
    """Compute standard performance statistics from a return stream."""
    returns = returns.dropna()
    equity = (1.0 + returns).cumprod()
    n_days = len(returns)

    total = float(equity.iloc[-1] - 1.0) if n_days else 0.0
    ann_return = float(equity.iloc[-1] ** (TRADING_DAYS / n_days) - 1.0) if n_days else 0.0
    ann_vol = float(returns.std() * np.sqrt(TRADING_DAYS))
    sharpe = float((ann_return - risk_free_rate) / ann_vol) if ann_vol > 0 else 0.0

    return BacktestResult(
        total_return=total,
        annualised_return=ann_return,
        annualised_volatility=ann_vol,
        sharpe_ratio=sharpe,
        max_drawdown=max_drawdown(equity),
        hit_rate=float((returns > 0).mean()) if n_days else 0.0,
        n_trades=n_days,
        equity_curve=equity,
    )


def walk_forward_signal(
    frame: pd.DataFrame,
    model_name: str = "gradient_boosting",
    n_splits: int = 5,
) -> pd.Series:
    """Generate out-of-sample predictions across the walk-forward folds.

    Each prediction comes from a model trained only on data preceding it,
    so the resulting signal is tradeable in principle rather than a
    hindsight artefact.
    """
    features = feature_columns(frame)
    X = frame[features].to_numpy()
    y = frame["target_direction"].to_numpy()

    signal = pd.Series(np.nan, index=frame.index, dtype=float)
    splitter = TimeSeriesSplit(n_splits=n_splits)

    for train_idx, test_idx in splitter.split(X):
        model = MODELS[model_name]()
        model.fit(X[train_idx], y[train_idx])
        signal.iloc[test_idx] = model.predict(X[test_idx])

    return signal


def backtest(
    frame: pd.DataFrame,
    model_name: str = "gradient_boosting",
    n_splits: int = 5,
    cost_bps: float = 5.0,
) -> dict:
    """Backtest the model signal against buy-and-hold.

    Args:
        cost_bps: Round-trip transaction cost in basis points, charged
            whenever the position changes. Ignoring costs is the most
            common way a backtest flatters itself.
    """
    signal = walk_forward_signal(frame, model_name=model_name, n_splits=n_splits)
    evaluated = frame.loc[signal.notna()].copy()
    position = signal.loc[signal.notna()]

    market_returns = evaluated["target_return"]
    gross = position * market_returns

    turnover = position.diff().abs().fillna(position.iloc[0])
    costs = turnover * (cost_bps / 10_000.0)
    net = gross - costs

    strategy = score_returns(net)
    benchmark = score_returns(market_returns)

    return {
        "model": model_name,
        "strategy": strategy,
        "buy_and_hold": benchmark,
        "excess_return": strategy.total_return - benchmark.total_return,
        "days_in_market": float((position > 0).mean()),
        "cost_bps": cost_bps,
    }

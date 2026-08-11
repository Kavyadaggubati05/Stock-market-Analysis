"""Feature engineering for next-day return prediction.

Every feature here is built only from information available at or before
the close of day t. The prediction target is the return realised on day
t+1. Any feature that peeks forward is a bug, and `tests/test_features.py`
asserts against it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Columns that must exist on the input frame.
REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def _bollinger_position(close: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.Series:
    """Where the close sits inside its Bollinger band, scaled to roughly [0, 1]."""
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = sma + n_std * std
    lower = sma - n_std * std
    width = (upper - lower).replace(0.0, np.nan)
    return (close - lower) / width


def build_features(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Turn an OHLCV frame into a modelling matrix plus targets.

    Args:
        df: DatetimeIndex frame with the columns in REQUIRED_COLUMNS.
        horizon: How many days ahead to predict. 1 = next trading day.

    Returns:
        A frame with engineered features, a continuous target
        (`target_return`) and a binary target (`target_direction`).
        Rows with any NaN from warm-up windows are dropped.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"input frame is missing required columns: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("input frame must be indexed by date")

    out = pd.DataFrame(index=df.index.copy())
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)

    daily_return = close.pct_change()

    # --- momentum: what has already happened ---
    for lag in (1, 2, 3, 5, 10):
        out[f"return_lag_{lag}"] = daily_return.shift(lag - 1) if lag == 1 else daily_return.shift(lag - 1)
    out["return_5d"] = close.pct_change(5)
    out["return_21d"] = close.pct_change(21)

    # --- trend: price relative to its own moving averages ---
    for window in (5, 10, 20, 50):
        sma = close.rolling(window).mean()
        out[f"close_over_sma_{window}"] = close / sma - 1.0
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    out["ema_12_over_26"] = ema_12 / ema_26 - 1.0

    # --- volatility ---
    out["volatility_10d"] = daily_return.rolling(10).std()
    out["volatility_21d"] = daily_return.rolling(21).std()
    out["volatility_ratio"] = out["volatility_10d"] / out["volatility_21d"].replace(0.0, np.nan)
    out["high_low_range"] = (df["high"] - df["low"]) / close

    # --- oscillators ---
    out["rsi_14"] = _rsi(close, 14)
    macd_line, signal_line, hist = _macd(close)
    out["macd"] = macd_line / close
    out["macd_hist"] = hist / close
    out["bollinger_pos"] = _bollinger_position(close)

    # --- volume ---
    vol_sma = volume.rolling(20).mean()
    out["volume_over_sma_20"] = volume / vol_sma.replace(0.0, np.nan) - 1.0
    out["volume_change"] = volume.pct_change()

    # --- calendar ---
    out["day_of_week"] = df.index.dayofweek
    out["month"] = df.index.month

    # --- targets: strictly forward-looking, never fed back as features ---
    out["target_return"] = close.pct_change(horizon).shift(-horizon)
    out["target_direction"] = (out["target_return"] > 0).astype(int)

    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Feature names only — targets excluded."""
    return [c for c in frame.columns if not c.startswith("target_")]

"""Historical price loading with a local cache and an offline fallback.

The cache keeps the dashboard responsive and avoids hammering the API on
every rerun. `synthetic_ohlcv` exists so the test suite and CI can run
without network access — it is never used as a data source for reported
results.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path("data/cache")
COLUMNS = ["open", "high", "low", "close", "volume"]


def _cache_path(ticker: str, start: str, end: str) -> Path:
    return CACHE_DIR / f"{ticker.upper()}_{start}_{end}.parquet"


def fetch_prices(
    ticker: str,
    start: str = "2015-01-01",
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load daily OHLCV for one ticker, preferring the local Parquet cache."""
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    path = _cache_path(ticker, start, end)

    if use_cache and path.exists():
        return pd.read_parquet(path)

    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance is required to fetch live data. Install it with "
            "`pip install yfinance`, or use synthetic_ohlcv() for offline work."
        ) from exc

    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if raw.empty:
        raise ValueError(f"no data returned for {ticker!r} between {start} and {end}")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    frame = raw.rename(columns=str.lower)[COLUMNS]
    frame.index = pd.to_datetime(frame.index)
    frame.index.name = "date"

    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path)

    return frame


def synthetic_ohlcv(n_days: int = 1500, seed: int = 42, start: str = "2018-01-01") -> pd.DataFrame:
    """Generate a geometric-random-walk price series for offline testing.

    Deliberately has no predictable structure, which makes it a useful
    negative control: a model that appears to beat the baseline here is
    almost certainly leaking information.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days, name="date")

    returns = rng.normal(loc=0.0003, scale=0.015, size=n_days)
    close = 100.0 * np.exp(np.cumsum(returns))

    intraday = np.abs(rng.normal(0.0, 0.008, size=n_days))
    frame = pd.DataFrame(
        {
            "open": close * (1.0 + rng.normal(0.0, 0.004, size=n_days)),
            "high": close * (1.0 + intraday),
            "low": close * (1.0 - intraday),
            "close": close,
            "volume": rng.integers(1_000_000, 20_000_000, size=n_days).astype(float),
        },
        index=dates,
    )
    frame["high"] = frame[["open", "high", "close"]].max(axis=1)
    frame["low"] = frame[["open", "low", "close"]].min(axis=1)
    return frame

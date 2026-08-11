"""Train the next-day direction model and persist it for the Streamlit app.

Produces three artifacts, mirroring a standard deployable-model layout:

    stock_model.joblib      fitted classifier
    scaler.joblib           fitted StandardScaler for the feature matrix
    feature_metadata.joblib feature order, ticker, training window, metrics

The feature metadata file plays the role a label encoder plays in a
classification app: it is the contract between training and serving. If
the app builds features in a different order than the model was trained
on, predictions are silently wrong, so the order is persisted and
verified at load time.

Usage:
    python train_model.py --ticker AAPL
    python train_model.py --offline          # synthetic data, no network
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.backtest.engine import backtest
from src.data.fetch import fetch_prices, synthetic_ohlcv
from src.features.build_features import build_features, feature_columns
from src.models.train import MODELS, walk_forward_evaluate

ARTIFACT_DIR = Path(".")
MODEL_PATH = ARTIFACT_DIR / "stock_model.joblib"
SCALER_PATH = ARTIFACT_DIR / "scaler.joblib"
METADATA_PATH = ARTIFACT_DIR / "feature_metadata.joblib"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and persist the direction model")
    p.add_argument("--ticker", default="AAPL")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--model", default="gradient_boosting", choices=sorted(MODELS))
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--offline", action="store_true", help="train on synthetic data")
    return p.parse_args()


def train(ticker: str, start: str, model_name: str, n_splits: int, offline: bool) -> dict:
    if offline:
        print("Training on synthetic data (offline placeholder artifacts).")
        prices = synthetic_ohlcv()
        ticker = "SYNTHETIC"
    else:
        print(f"Fetching {ticker} from {start}...")
        prices = fetch_prices(ticker, start=start)

    frame = build_features(prices)
    features = feature_columns(frame)
    X = frame[features].to_numpy()
    y = frame["target_direction"].to_numpy()
    print(f"{len(frame)} rows, {len(features)} features")

    # Honest evaluation first, on unscaled data through the walk-forward split.
    print("\nWalk-forward evaluation...")
    result = walk_forward_evaluate(frame, model_name=model_name, n_splits=n_splits)
    print(f"  accuracy {result.mean_accuracy:.4f} (+/- {result.std_accuracy:.4f})")
    print(f"  roc auc  {result.mean_roc_auc:.4f}")

    print("\nBacktest (5 bps costs)...")
    bt = backtest(frame, model_name=model_name, n_splits=n_splits, cost_bps=5.0)
    print(f"  strategy sharpe    {bt['strategy'].sharpe_ratio:+.3f}")
    print(f"  buy-and-hold sharpe {bt['buy_and_hold'].sharpe_ratio:+.3f}")
    print(f"  excess return       {bt['excess_return']:+.2%}")

    # Refit on the full history for serving.
    scaler = StandardScaler().fit(X)
    model = MODELS[model_name]()
    model.fit(scaler.transform(X), y)

    metadata = {
        "feature_names": features,
        "ticker": ticker,
        "model_name": model_name,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_start": str(frame.index.min().date()),
        "train_end": str(frame.index.max().date()),
        "n_rows": int(len(frame)),
        "is_synthetic": offline,
        "walk_forward": {
            "mean_accuracy": result.mean_accuracy,
            "std_accuracy": result.std_accuracy,
            "mean_f1": result.mean_f1,
            "mean_roc_auc": result.mean_roc_auc,
            "n_splits": n_splits,
        },
        "backtest": {
            "strategy_sharpe": bt["strategy"].sharpe_ratio,
            "benchmark_sharpe": bt["buy_and_hold"].sharpe_ratio,
            "strategy_max_drawdown": bt["strategy"].max_drawdown,
            "excess_return": bt["excess_return"],
            "cost_bps": bt["cost_bps"],
        },
    }

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(metadata, METADATA_PATH)
    print(f"\nSaved {MODEL_PATH.name}, {SCALER_PATH.name}, {METADATA_PATH.name}")
    return metadata


def load_artifacts():
    """Load the three artifacts, raising a clear error if any is missing."""
    for path in (MODEL_PATH, SCALER_PATH, METADATA_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"{path.name} not found — run `python train_model.py --ticker AAPL` first."
            )
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH), joblib.load(METADATA_PATH)


def predict_latest(frame, model, scaler, metadata) -> dict:
    """Score the most recent row, verifying the feature contract holds."""
    expected = metadata["feature_names"]
    actual = feature_columns(frame)
    if actual != expected:
        raise ValueError(
            "feature mismatch between training and serving.\n"
            f"  expected: {expected}\n  got:      {actual}"
        )

    X = scaler.transform(frame[expected].tail(1).to_numpy())
    proba = float(model.predict_proba(X)[0][1]) if hasattr(model, "predict_proba") else float("nan")
    return {
        "as_of": str(frame.index[-1].date()),
        "direction": "UP" if model.predict(X)[0] == 1 else "DOWN",
        "confidence": proba if not np.isnan(proba) else None,
    }


if __name__ == "__main__":
    args = parse_args()
    train(args.ticker, args.start, args.model, args.n_splits, args.offline)

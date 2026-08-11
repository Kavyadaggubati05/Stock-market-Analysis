"""Walk-forward training and evaluation for next-day direction prediction.

Financial time series break the assumptions behind ordinary k-fold cross
validation: shuffling rows leaks future information into the training set
and produces accuracy numbers that evaporate in live use. This module uses
expanding-window walk-forward validation instead, so every prediction is
made by a model that only ever saw earlier data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.build_features import feature_columns

MODELS = {
    "baseline_majority": lambda: DummyClassifier(strategy="most_frequent"),
    "logistic_regression": lambda: Pipeline(
        [("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, C=0.1))]
    ),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=300, max_depth=5, min_samples_leaf=25, random_state=42, n_jobs=-1
    ),
    "gradient_boosting": lambda: GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
    ),
}


@dataclass
class FoldResult:
    fold: int
    train_size: int
    test_size: int
    accuracy: float
    f1: float
    roc_auc: float


@dataclass
class ModelResult:
    model_name: str
    n_splits: int
    mean_accuracy: float
    std_accuracy: float
    mean_f1: float
    mean_roc_auc: float
    folds: list

    def to_dict(self) -> dict:
        d = asdict(self)
        d["folds"] = [asdict(f) if not isinstance(f, dict) else f for f in self.folds]
        return d


def walk_forward_evaluate(
    frame: pd.DataFrame,
    model_name: str = "gradient_boosting",
    n_splits: int = 5,
    target: str = "target_direction",
) -> ModelResult:
    """Evaluate one model with expanding-window walk-forward validation."""
    if model_name not in MODELS:
        raise KeyError(f"unknown model {model_name!r}; choose from {sorted(MODELS)}")

    features = feature_columns(frame)
    X = frame[features].to_numpy()
    y = frame[target].to_numpy()

    splitter = TimeSeriesSplit(n_splits=n_splits)
    folds: list[FoldResult] = []

    for i, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        model = MODELS[model_name]()
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])

        try:
            proba = model.predict_proba(X[test_idx])[:, 1]
            auc = float(roc_auc_score(y[test_idx], proba))
        except (AttributeError, ValueError):
            auc = float("nan")

        folds.append(
            FoldResult(
                fold=i,
                train_size=len(train_idx),
                test_size=len(test_idx),
                accuracy=float(accuracy_score(y[test_idx], preds)),
                f1=float(f1_score(y[test_idx], preds, zero_division=0)),
                roc_auc=auc,
            )
        )

    accuracies = [f.accuracy for f in folds]
    return ModelResult(
        model_name=model_name,
        n_splits=n_splits,
        mean_accuracy=float(np.mean(accuracies)),
        std_accuracy=float(np.std(accuracies)),
        mean_f1=float(np.mean([f.f1 for f in folds])),
        mean_roc_auc=float(np.nanmean([f.roc_auc for f in folds])),
        folds=folds,
    )


def compare_models(frame: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """Run every registered model and return a leaderboard.

    The majority-class baseline is included deliberately. On daily equity
    direction it is a genuinely hard benchmark to beat, and a model that
    does not beat it is not worth deploying.
    """
    rows = []
    for name in MODELS:
        result = walk_forward_evaluate(frame, model_name=name, n_splits=n_splits)
        rows.append(
            {
                "model": name,
                "mean_accuracy": result.mean_accuracy,
                "std_accuracy": result.std_accuracy,
                "mean_f1": result.mean_f1,
                "mean_roc_auc": result.mean_roc_auc,
            }
        )
    return pd.DataFrame(rows).sort_values("mean_accuracy", ascending=False).reset_index(drop=True)


def fit_final_model(frame: pd.DataFrame, model_name: str = "gradient_boosting"):
    """Refit the chosen model on the full history for live prediction."""
    features = feature_columns(frame)
    model = MODELS[model_name]()
    model.fit(frame[features].to_numpy(), frame["target_direction"].to_numpy())
    return model, features


def log_to_mlflow(result: ModelResult, ticker: str, experiment: str = "stock-direction") -> bool:
    """Log a run to MLflow if it is installed. Returns False if unavailable."""
    try:
        import mlflow
    except ImportError:
        return False

    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=f"{ticker}-{result.model_name}"):
        mlflow.log_params({"ticker": ticker, "model": result.model_name, "n_splits": result.n_splits})
        mlflow.log_metrics(
            {
                "mean_accuracy": result.mean_accuracy,
                "std_accuracy": result.std_accuracy,
                "mean_f1": result.mean_f1,
                "mean_roc_auc": result.mean_roc_auc,
            }
        )
    return True


def save_results(result: ModelResult, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2))

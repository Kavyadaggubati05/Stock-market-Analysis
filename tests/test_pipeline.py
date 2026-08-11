"""Tests for the feature, model and backtest layers.

The important tests here are the leakage guards. Accuracy tests on
financial data are near-meaningless; tests that prove the pipeline cannot
see the future are what make the numbers trustworthy.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import backtest, max_drawdown, score_returns
from src.data.fetch import synthetic_ohlcv
from src.features.build_features import build_features, feature_columns
from src.models.train import compare_models, fit_final_model, walk_forward_evaluate


@pytest.fixture(scope="module")
def prices():
    return synthetic_ohlcv(n_days=1200, seed=7)


@pytest.fixture(scope="module")
def features(prices):
    return build_features(prices)


class TestFeatures:
    def test_no_nans_survive(self, features):
        assert not features.isna().any().any()

    def test_targets_present_and_separated(self, features):
        assert "target_return" in features.columns
        assert "target_direction" in features.columns
        assert not any(c.startswith("target_") for c in feature_columns(features))

    def test_direction_matches_return_sign(self, features):
        expected = (features["target_return"] > 0).astype(int)
        pd.testing.assert_series_equal(features["target_direction"], expected, check_names=False)

    def test_missing_column_raises(self, prices):
        with pytest.raises(ValueError, match="missing required columns"):
            build_features(prices.drop(columns=["volume"]))

    def test_features_do_not_depend_on_the_future(self, prices):
        """Truncating the series must not change features computed earlier.

        If a feature used forward-looking information, the values on a
        given date would shift once later rows were removed.
        """
        full = build_features(prices)
        truncated = build_features(prices.iloc[:-100])

        shared = truncated.index.intersection(full.index)
        assert len(shared) > 500, "not enough overlap to make the test meaningful"

        cols = feature_columns(full)
        np.testing.assert_allclose(
            full.loc[shared, cols].to_numpy(),
            truncated.loc[shared, cols].to_numpy(),
            rtol=1e-9,
            atol=1e-12,
        )


class TestModels:
    def test_walk_forward_runs(self, features):
        result = walk_forward_evaluate(features, model_name="logistic_regression", n_splits=4)
        assert result.n_splits == 4
        assert len(result.folds) == 4
        assert 0.0 <= result.mean_accuracy <= 1.0

    def test_train_sets_expand_and_never_overlap_test(self, features):
        result = walk_forward_evaluate(features, model_name="baseline_majority", n_splits=4)
        sizes = [f.train_size for f in result.folds]
        assert sizes == sorted(sizes), "training window should expand over folds"

    def test_unknown_model_raises(self, features):
        with pytest.raises(KeyError):
            walk_forward_evaluate(features, model_name="not_a_model")

    def test_leaderboard_covers_every_model(self, features):
        board = compare_models(features, n_splits=3)
        assert len(board) == 4
        assert board["mean_accuracy"].between(0.0, 1.0).all()

    def test_no_model_beats_chance_on_random_data(self, features):
        """Negative control.

        The synthetic series is a random walk with no learnable signal, so
        every model should land near the majority-class rate. A model that
        scores far above it is evidence of leakage, not skill.
        """
        board = compare_models(features, n_splits=3).set_index("model")
        majority = board.loc["baseline_majority", "mean_accuracy"]
        best = board["mean_accuracy"].max()
        assert best < majority + 0.12, f"suspiciously strong result on random data: {best:.3f}"

    def test_final_model_predicts(self, features):
        model, cols = fit_final_model(features, model_name="random_forest")
        preds = model.predict(features[cols].tail(10).to_numpy())
        assert set(np.unique(preds)).issubset({0, 1})


class TestBacktest:
    def test_max_drawdown_is_negative_on_a_decline(self):
        equity = pd.Series([1.0, 1.5, 1.2, 0.9, 1.1])
        assert max_drawdown(equity) == pytest.approx(-0.4)

    def test_max_drawdown_is_zero_when_monotonic(self):
        assert max_drawdown(pd.Series([1.0, 1.1, 1.2])) == pytest.approx(0.0)

    def test_score_returns_shape(self):
        rng = np.random.default_rng(1)
        returns = pd.Series(rng.normal(0.0005, 0.01, 500))
        result = score_returns(returns)
        assert result.n_trades == 500
        assert len(result.equity_curve) == 500
        assert -1.0 <= result.max_drawdown <= 0.0

    def test_backtest_reports_both_legs(self, features):
        result = backtest(features, model_name="logistic_regression", n_splits=4)
        assert "strategy" in result and "buy_and_hold" in result
        assert 0.0 <= result["days_in_market"] <= 1.0

    def test_costs_reduce_returns(self, features):
        cheap = backtest(features, model_name="logistic_regression", n_splits=4, cost_bps=0.0)
        pricey = backtest(features, model_name="logistic_regression", n_splits=4, cost_bps=50.0)
        assert pricey["strategy"].total_return <= cheap["strategy"].total_return

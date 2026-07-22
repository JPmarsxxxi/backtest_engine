import time

import numpy as np
import pandas as pd
import pytest

from backtest.engine import Engine
from backtest.risk import RiskConfig
from backtest.simulation import panel_from_returns
from backtest.simulation.batch import BatchDataView, BatchResult, run_batch
from backtest.strategy import Strategy


# --- shared fixtures ----------------------------------------------------

@pytest.fixture
def assets():
    return ["A", "B", "C", "D", "E"]


@pytest.fixture
def dates():
    return pd.date_range("2020-01-01", periods=520, freq="B")


@pytest.fixture
def paths(assets, dates):
    rng = np.random.default_rng(0)
    return rng.normal(0.0003, 0.015, (10, len(dates), len(assets)))


# --- strategies for tests -----------------------------------------------

class _MomBoth(Strategy):
    """Cross-sectional momentum with both per-path and batch interfaces.
    apply_risk overridden to identity for batch-vs-per-path equivalence."""

    rebalance_frequency = "monthly"
    risk = RiskConfig(max_position=1.0, max_gross=2.0, max_net=2.0)

    def __init__(self, lookback: int = 60):
        self.lookback = lookback

    def __repr__(self):
        return f"_MomBoth(lookback={self.lookback})"

    def apply_risk(self, proposed, state, data):
        return proposed.fillna(0.0).astype(float)

    def generate_weights(self, data, t):
        if len(data.prices) < self.lookback + 1:
            return pd.Series(0.0, index=data.assets)
        rets = data.prices.iloc[-1] / data.prices.iloc[-self.lookback - 1] - 1
        rets = rets.reindex(data.assets).dropna()
        if len(rets) < 2:
            return pd.Series(0.0, index=data.assets)
        med = rets.median()
        n = len(rets)
        w = pd.Series(0.0, index=rets.index)
        w[rets > med] = 1.0 / n
        w[rets < med] = -1.0 / n
        return w

    def generate_weights_batch(self, view, t):
        prices = view.prices  # (N, t+1, K)
        N, T_so_far, K = prices.shape
        if T_so_far < self.lookback + 1:
            return np.zeros((N, K))
        rets = prices[:, -1, :] / prices[:, -self.lookback - 1, :] - 1
        med = np.median(rets, axis=1, keepdims=True)
        w = np.zeros((N, K), dtype=np.float64)
        w[rets > med] = 1.0 / K
        w[rets < med] = -1.0 / K
        return w


class _MomNoBatch(Strategy):
    rebalance_frequency = "monthly"

    def __repr__(self):
        return "_MomNoBatch()"

    def generate_weights(self, data, t):
        return pd.Series(0.0, index=data.assets)


# --- tests --------------------------------------------------------------

def test_batch_data_view_shape():
    rng = np.random.default_rng(0)
    prices = rng.normal(100, 1, (10, 50, 5))
    view = BatchDataView(prices, t=10, assets=["A", "B", "C", "D", "E"])
    assert view.prices.shape == (10, 11, 5)


def test_batch_data_view_t_zero():
    rng = np.random.default_rng(0)
    prices = rng.normal(100, 1, (10, 50, 5))
    view = BatchDataView(prices, t=0, assets=["A", "B", "C", "D", "E"])
    assert view.prices.shape == (10, 1, 5)


def test_run_batch_produces_correct_shapes(paths, assets, dates):
    result = run_batch(_MomBoth(60), paths, assets, dates, warmup_bars=120)
    N = paths.shape[0]
    expected_T = paths.shape[1] - 120
    assert isinstance(result, BatchResult)
    assert result.path_returns.shape == (N, expected_T)
    assert result.path_equity.shape == (N, expected_T)
    assert result.weights.shape == (N, expected_T, len(assets))
    assert len(result.dates) == expected_T
    assert result.n_paths == N


def test_run_batch_rejects_strategy_without_batch_method(paths, assets, dates):
    with pytest.raises(TypeError, match="generate_weights_batch"):
        run_batch(_MomNoBatch(), paths, assets, dates, warmup_bars=120)


def test_run_batch_rejects_bad_paths_shape(assets, dates):
    with pytest.raises(ValueError, match="3-D"):
        run_batch(_MomBoth(60), np.zeros((10, 5)), assets, dates)


def test_run_batch_rejects_warmup_out_of_range(paths, assets, dates):
    with pytest.raises(ValueError, match="warmup_bars"):
        run_batch(_MomBoth(60), paths, assets, dates, warmup_bars=-1)
    with pytest.raises(ValueError, match="warmup_bars"):
        run_batch(_MomBoth(60), paths, assets, dates, warmup_bars=10000)


def test_run_batch_rejects_bad_initial_capital(paths, assets, dates):
    with pytest.raises(ValueError, match="initial_capital"):
        run_batch(
            _MomBoth(60), paths, assets, dates,
            warmup_bars=120, initial_capital=-1,
        )


def test_run_batch_reproducibility(paths, assets, dates):
    a = run_batch(_MomBoth(60), paths, assets, dates, warmup_bars=120)
    b = run_batch(_MomBoth(60), paths, assets, dates, warmup_bars=120)
    np.testing.assert_array_equal(a.path_returns, b.path_returns)
    np.testing.assert_array_equal(a.path_equity, b.path_equity)
    np.testing.assert_array_equal(a.weights, b.weights)


def test_run_batch_matches_per_path_engine(paths, assets, dates):
    """Critical equivalence test: batch matches per-path Engine bit-for-bit
    (no costs, no liquidity, identity risk)."""
    per_path = []
    for p in range(paths.shape[0]):
        panel = panel_from_returns(paths[p], assets, dates)
        result = Engine(panel, _MomBoth(60)).run(start=panel.dates[120])
        per_path.append(result.returns.values)
    per_path = np.array(per_path)

    batch_result = run_batch(_MomBoth(60), paths, assets, dates, warmup_bars=120)
    np.testing.assert_allclose(
        batch_result.path_returns, per_path, atol=1e-10
    )


def test_batch_result_to_monte_carlo_result(paths, assets, dates):
    result = run_batch(_MomBoth(60), paths, assets, dates, warmup_bars=120)
    mcr = result.to_monte_carlo_result(gen_name="batch")
    assert "batch" in mcr.path_returns
    assert mcr.path_returns["batch"].shape == (
        paths.shape[1] - 120, paths.shape[0],
    )
    assert mcr.n_paths == paths.shape[0]
    table = mcr.sensitivity_table()
    assert "batch" in table.index


def test_run_batch_speedup_vs_per_path(paths, assets, dates):
    """Batch should be at least 2x faster than per-path Engine on N=10 paths."""
    t0 = time.perf_counter()
    for p in range(paths.shape[0]):
        panel = panel_from_returns(paths[p], assets, dates)
        Engine(panel, _MomBoth(60)).run(start=panel.dates[120])
    per_path_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    run_batch(_MomBoth(60), paths, assets, dates, warmup_bars=120)
    batch_time = time.perf_counter() - t0

    assert batch_time < per_path_time / 2, (
        f"batch ({batch_time:.3f}s) not >=2x faster than per-path "
        f"({per_path_time:.3f}s)"
    )


def test_run_batch_weights_shape_validation(paths, assets, dates):
    class _BadShape(Strategy):
        rebalance_frequency = "monthly"
        def __repr__(self): return "_BadShape()"
        def generate_weights(self, data, t):
            return pd.Series(0.0, index=data.assets)
        def generate_weights_batch(self, view, t):
            return np.zeros((view.prices.shape[0], 99))  # wrong K

    with pytest.raises(ValueError, match="returned shape"):
        run_batch(_BadShape(), paths, assets, dates, warmup_bars=120)


# --- summary() / plot() smoke tests -------------------------------------

def test_batch_summary_returns_aggregate(paths, assets, dates, capsys):
    result = run_batch(_MomBoth(60), paths, assets, dates, warmup_bars=120)
    agg = result.summary()
    assert "sharpe_mean" in agg
    assert "max_drawdown_mean" in agg
    captured = capsys.readouterr()
    assert "BatchResult" in captured.out


def test_batch_plot_smoke(paths, assets, dates):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = run_batch(_MomBoth(60), paths, assets, dates, warmup_bars=120)
    fig = result.plot()
    assert fig is not None
    assert len(fig.axes) == 2
    plt.close(fig)

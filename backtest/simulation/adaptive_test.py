import numpy as np
import pandas as pd
import pytest

from backtest.risk import RiskConfig
from backtest.simulation import (
    GaussianGenerator,
    HistoricalReplayGenerator,
    MonteCarloEngine,
)
from backtest.simulation.adaptive import AdaptiveResult, run_until_converged
from backtest.strategy import Strategy


@pytest.fixture
def real_returns():
    rng = np.random.default_rng(42)
    return rng.normal(0.0003, 0.015, (1260, 5))


@pytest.fixture
def assets():
    return ["A", "B", "C", "D", "E"]


@pytest.fixture
def dates():
    return pd.date_range("2020-01-01", periods=520, freq="B")


class _Mom(Strategy):
    rebalance_frequency = "monthly"
    risk = RiskConfig(max_position=0.30, max_gross=1.0, max_net=1.0)

    def __init__(self, lookback: int = 60):
        self.lookback = lookback

    def __repr__(self) -> str:
        return f"_Mom(lookback={self.lookback})"

    def generate_weights(self, data, t):
        if len(data.prices) < self.lookback + 1:
            return pd.Series(0.0, index=data.assets)
        rets = data.prices.iloc[-1] / data.prices.iloc[-self.lookback - 1] - 1
        rets = rets.reindex(data.assets).dropna()
        if len(rets) < 2:
            return pd.Series(0.0, index=data.assets)
        median = rets.median()
        n = len(rets)
        w = pd.Series(0.0, index=rets.index)
        w[rets > median] = 1.0 / n
        w[rets < median] = -1.0 / n
        return w


@pytest.fixture
def base_engine(assets, dates, real_returns):
    gens = {
        "gaussian": GaussianGenerator(mu=0.0003, sigma=0.015),
        "historical": HistoricalReplayGenerator(real_returns),
    }
    return MonteCarloEngine(
        strategy=_Mom(lookback=60),
        generators=gens,
        assets=assets,
        dates=dates,
        warmup_bars=120,
    )


def test_adaptive_returns_valid_result(base_engine):
    res = run_until_converged(
        base_engine,
        metric="sharpe",
        target_se=10.0,
        batch_size=20,
        max_paths=200,
        min_paths=20,
        seed=0,
        n_jobs=1,
    )
    assert isinstance(res, AdaptiveResult)
    assert res.converged is True
    assert res.n_paths_used == 20
    assert res.metric == "sharpe"
    assert set(res.result.path_returns.keys()) == {"gaussian", "historical"}
    for name, df in res.result.path_returns.items():
        assert df.shape[1] == res.n_paths_used


def test_adaptive_reaches_max_when_unconvergeable(base_engine):
    res = run_until_converged(
        base_engine,
        metric="sharpe",
        target_se=1e-9,
        batch_size=20,
        max_paths=60,
        min_paths=20,
        seed=0,
        n_jobs=1,
    )
    assert res.converged is False
    assert res.n_paths_used == 60


def test_adaptive_se_trace_populated(base_engine):
    res = run_until_converged(
        base_engine,
        metric="sharpe",
        target_se=1e-9,
        batch_size=20,
        max_paths=60,
        min_paths=20,
        seed=0,
        n_jobs=1,
    )
    for name in base_engine.generators:
        assert len(res.se_trace[name]) >= 1
        for v in res.se_trace[name]:
            assert v >= 0


def test_adaptive_validation(assets, dates):
    gens = {"g": GaussianGenerator(mu=0.0, sigma=0.01)}
    engine = MonteCarloEngine(
        strategy=_Mom(20),
        generators=gens,
        assets=assets,
        dates=dates,
        warmup_bars=30,
    )
    with pytest.raises(ValueError, match="batch_size"):
        run_until_converged(engine, batch_size=0)
    with pytest.raises(ValueError, match="max_paths"):
        run_until_converged(engine, batch_size=100, max_paths=10)
    with pytest.raises(ValueError, match="min_paths"):
        run_until_converged(engine, min_paths=1)
    with pytest.raises(ValueError, match="target_se"):
        run_until_converged(engine, target_se=0)


def test_adaptive_reproducibility(base_engine):
    a = run_until_converged(
        base_engine,
        target_se=10.0,
        batch_size=20,
        max_paths=40,
        min_paths=20,
        seed=42,
        n_jobs=1,
    )
    b = run_until_converged(
        base_engine,
        target_se=10.0,
        batch_size=20,
        max_paths=40,
        min_paths=20,
        seed=42,
        n_jobs=1,
    )
    for name in base_engine.generators:
        pd.testing.assert_frame_equal(
            a.result.path_returns[name], b.result.path_returns[name]
        )

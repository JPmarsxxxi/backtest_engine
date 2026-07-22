import numpy as np
import pandas as pd
import pytest

from backtest.risk import RiskConfig
from backtest.simulation import (
    GaussianGenerator,
    HistoricalReplayGenerator,
    MonteCarloEngine,
    PermutationGenerator,
)
from backtest.strategy import Strategy


# --- shared fixtures -----------------------------------------------------

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
        "permutation": PermutationGenerator(real_returns),
    }
    return MonteCarloEngine(
        strategy=_Mom(lookback=60),
        generators=gens,
        assets=assets,
        dates=dates,
        warmup_bars=120,
    )


# --- core behavior -------------------------------------------------------

def test_runs_smoke(base_engine):
    result = base_engine.run(n_paths=4, seed=0, n_jobs=1)
    assert set(result.path_returns.keys()) == {"gaussian", "historical", "permutation"}
    for name, df in result.path_returns.items():
        assert df.shape[1] == 4
        assert df.shape[0] > 0
        assert list(df.columns) == [f"path_{i}" for i in range(4)]
    for name, df in result.path_equity.items():
        assert df.shape == result.path_returns[name].shape
    assert result.n_paths == 4
    assert result.seed == 0


def test_reproducibility_same_seed(base_engine):
    a = base_engine.run(n_paths=3, seed=42, n_jobs=1)
    b = base_engine.run(n_paths=3, seed=42, n_jobs=1)
    for name in a.path_returns:
        pd.testing.assert_frame_equal(a.path_returns[name], b.path_returns[name])
        pd.testing.assert_frame_equal(a.path_equity[name], b.path_equity[name])


def test_different_seeds_differ(base_engine):
    a = base_engine.run(n_paths=3, seed=1, n_jobs=1)
    b = base_engine.run(n_paths=3, seed=2, n_jobs=1)
    assert not a.path_returns["gaussian"].equals(b.path_returns["gaussian"])


def test_per_generator_seed_offset(base_engine):
    # Same base seed but per-gen offset means generators see independent noise.
    result = base_engine.run(n_paths=3, seed=0, n_jobs=1)
    gauss = result.path_returns["gaussian"].values
    perm = result.path_returns["permutation"].values
    assert not np.allclose(gauss, perm)


def test_n_jobs_deterministic(base_engine):
    a = base_engine.run(n_paths=4, seed=0, n_jobs=1)
    b = base_engine.run(n_paths=4, seed=0, n_jobs=2)
    for name in a.path_returns:
        pd.testing.assert_frame_equal(a.path_returns[name], b.path_returns[name])


# --- result methods ------------------------------------------------------

def test_metrics_per_path(base_engine):
    result = base_engine.run(n_paths=3, seed=0, n_jobs=1)
    mpp = result.metrics_per_path()
    assert set(mpp.keys()) == {"gaussian", "historical", "permutation"}
    for name, reports in mpp.items():
        assert len(reports) == 3
        for r in reports:
            assert hasattr(r, "sharpe")
            assert hasattr(r, "max_drawdown")


def test_aggregate_metrics(base_engine):
    result = base_engine.run(n_paths=3, seed=0, n_jobs=1)
    agg = result.aggregate_metrics()
    for name in ("gaussian", "historical", "permutation"):
        for stat in ("mean", "std", "min", "max"):
            assert f"sharpe_{stat}" in agg[name]
            assert f"max_drawdown_{stat}" in agg[name]


def test_sensitivity_table_shape(base_engine):
    result = base_engine.run(n_paths=10, seed=0, n_jobs=1)
    table = result.sensitivity_table(
        metrics=("sharpe", "max_drawdown"),
        quantiles=(0.05, 0.50, 0.95),
    )
    assert list(table.index) == ["gaussian", "historical", "permutation"]
    expected_cols = {
        "sharpe_mean", "sharpe_q05", "sharpe_q50", "sharpe_q95",
        "max_drawdown_mean", "max_drawdown_q05", "max_drawdown_q50", "max_drawdown_q95",
    }
    assert set(table.columns) == expected_cols
    assert table.index.name == "generator"


def test_sensitivity_table_quantile_ordering(base_engine):
    result = base_engine.run(n_paths=20, seed=0, n_jobs=1)
    table = result.sensitivity_table(metrics=("sharpe",), quantiles=(0.05, 0.50, 0.95))
    for name in table.index:
        q05 = table.loc[name, "sharpe_q05"]
        q50 = table.loc[name, "sharpe_q50"]
        q95 = table.loc[name, "sharpe_q95"]
        if all(np.isfinite([q05, q50, q95])):
            assert q05 <= q50 <= q95


def test_generator_configs_captured(base_engine):
    result = base_engine.run(n_paths=2, seed=0, n_jobs=1)
    assert set(result.generator_configs.keys()) == {"gaussian", "historical", "permutation"}
    assert result.generator_configs["gaussian"]["kind"] == "gaussian"
    assert result.generator_configs["permutation"]["kind"] == "permutation"


# --- strategy forms ------------------------------------------------------

def test_factory_strategy_works(assets, dates):
    gens = {"gaussian": GaussianGenerator(mu=0.0003, sigma=0.015)}
    engine_inst = MonteCarloEngine(
        strategy=_Mom(lookback=60),
        generators=gens,
        assets=assets,
        dates=dates,
        warmup_bars=120,
    )
    engine_fact = MonteCarloEngine(
        strategy=lambda: _Mom(lookback=60),
        generators=gens,
        assets=assets,
        dates=dates,
        warmup_bars=120,
    )
    a = engine_inst.run(n_paths=3, seed=0, n_jobs=1)
    b = engine_fact.run(n_paths=3, seed=0, n_jobs=1)
    pd.testing.assert_frame_equal(a.path_returns["gaussian"], b.path_returns["gaussian"])


# --- semantic sanity (the whole point of MC) -----------------------------

def test_permutation_null_centered_near_zero(real_returns, assets, dates):
    gens = {"permutation": PermutationGenerator(real_returns)}
    engine = MonteCarloEngine(
        strategy=_Mom(lookback=60),
        generators=gens,
        assets=assets,
        dates=dates,
        warmup_bars=120,
    )
    result = engine.run(n_paths=50, seed=0, n_jobs=1)
    table = result.sensitivity_table(metrics=("sharpe",), quantiles=(0.50,))
    median_sharpe = table.loc["permutation", "sharpe_q50"]
    # _Mom on permuted (no-autocorr) returns has no edge -> median Sharpe near 0.
    assert abs(median_sharpe) < 1.5


# --- validation ----------------------------------------------------------

def test_validation_errors(assets, dates):
    gens = {"gaussian": GaussianGenerator(mu=0.0, sigma=0.01)}
    with pytest.raises(ValueError, match="non-empty"):
        MonteCarloEngine(_Mom(60), {}, assets, dates)
    with pytest.raises(ValueError, match="positive"):
        MonteCarloEngine(_Mom(60), gens, assets, dates, initial_capital=-1)
    with pytest.raises(ValueError, match="warmup_bars"):
        MonteCarloEngine(_Mom(60), gens, assets, dates, warmup_bars=-1)
    with pytest.raises(ValueError, match="warmup_bars"):
        MonteCarloEngine(_Mom(60), gens, assets, dates, warmup_bars=10000)


def test_n_paths_validation(assets, dates):
    gens = {"gaussian": GaussianGenerator(mu=0.0, sigma=0.01)}
    engine = MonteCarloEngine(_Mom(60), gens, assets, dates, warmup_bars=120)
    with pytest.raises(ValueError, match="n_paths"):
        engine.run(n_paths=0)


# --- path-tensor cache integration --------------------------------------

def test_engine_populates_cache(tmp_path, assets, dates, real_returns):
    from backtest.simulation.cache import PathTensorCache
    cache = PathTensorCache(tmp_path / "cache")
    gens = {
        "gaussian": GaussianGenerator(mu=0.0003, sigma=0.015),
        "historical": HistoricalReplayGenerator(real_returns),
    }
    engine = MonteCarloEngine(
        strategy=_Mom(60),
        generators=gens,
        assets=assets,
        dates=dates,
        warmup_bars=120,
        path_cache=cache,
    )
    engine.run(n_paths=4, seed=0, n_jobs=1)
    assert cache.stats()["count"] == 2  # one file per generator


def test_engine_cache_hit_produces_same_results(tmp_path, assets, dates):
    from backtest.simulation.cache import PathTensorCache
    cache = PathTensorCache(tmp_path / "cache")
    gens = {"gaussian": GaussianGenerator(mu=0.0003, sigma=0.015)}
    engine = MonteCarloEngine(
        strategy=_Mom(60),
        generators=gens,
        assets=assets,
        dates=dates,
        warmup_bars=120,
        path_cache=cache,
    )
    a = engine.run(n_paths=4, seed=0, n_jobs=1)
    files_first = cache.stats()["count"]
    b = engine.run(n_paths=4, seed=0, n_jobs=1)
    files_second = cache.stats()["count"]

    assert files_first == files_second  # cache hit -> no new files
    pd.testing.assert_frame_equal(
        a.path_returns["gaussian"], b.path_returns["gaussian"]
    )


def test_engine_cache_miss_on_different_seed(tmp_path, assets, dates):
    from backtest.simulation.cache import PathTensorCache
    cache = PathTensorCache(tmp_path / "cache")
    gens = {"gaussian": GaussianGenerator(mu=0.0003, sigma=0.015)}
    engine = MonteCarloEngine(
        strategy=_Mom(60),
        generators=gens,
        assets=assets,
        dates=dates,
        warmup_bars=120,
        path_cache=cache,
    )
    engine.run(n_paths=4, seed=0, n_jobs=1)
    engine.run(n_paths=4, seed=1, n_jobs=1)
    assert cache.stats()["count"] == 2  # one per seed


def test_engine_no_cache_works_unchanged(base_engine):
    # Without path_cache the existing behavior is unchanged.
    assert base_engine.path_cache is None
    result = base_engine.run(n_paths=4, seed=0, n_jobs=1)
    assert result.n_paths == 4


# --- summary() ----------------------------------------------------------

def test_summary_returns_table_and_prints(base_engine, capsys):
    result = base_engine.run(n_paths=4, seed=0, n_jobs=1)
    table = result.summary()
    assert isinstance(table, pd.DataFrame)
    assert "gaussian" in table.index
    assert "historical" in table.index
    assert "permutation" in table.index
    captured = capsys.readouterr()
    assert "Monte Carlo summary" in captured.out
    assert "gaussian" in captured.out


def test_summary_custom_metrics(base_engine, capsys):
    result = base_engine.run(n_paths=4, seed=0, n_jobs=1)
    table = result.summary(metrics=("sharpe",), quantiles=(0.5,))
    expected_cols = {"sharpe_mean", "sharpe_q50"}
    assert set(table.columns) == expected_cols


# --- plot() smoke tests -------------------------------------------------

def test_mc_plot_smoke(base_engine):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = base_engine.run(n_paths=4, seed=0, n_jobs=1)
    fig = result.plot()
    assert fig is not None
    # 3 generators × 1 equity row + 1 sharpe + 1 dd = 5 axes minimum
    assert len(fig.axes) >= 5
    plt.close(fig)


def test_mc_plot_single_generator(assets, dates):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gens = {"gaussian": GaussianGenerator(mu=0.0, sigma=0.01)}
    engine = MonteCarloEngine(
        strategy=_Mom(60), generators=gens, assets=assets,
        dates=dates, warmup_bars=120,
    )
    result = engine.run(n_paths=4, seed=0, n_jobs=1)
    fig = result.plot()
    assert fig is not None
    plt.close(fig)

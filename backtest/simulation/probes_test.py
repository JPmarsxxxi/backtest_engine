import json

import numpy as np
import pandas as pd
import pytest

from backtest.data import DataPanel
from backtest.risk import RiskConfig
from backtest.simulation import JumpOverlayAdapter
from backtest.simulation.auto_select import select_generators
from backtest.simulation.fingerprint import FingerprintCache, fingerprint
from backtest.simulation.probes import Probe, ProbeBattery
from backtest.strategy import Strategy


# --- fixtures ------------------------------------------------------------

@pytest.fixture
def battery():
    return ProbeBattery(n_assets=5, n_steps=520, seed=42)


@pytest.fixture
def real_returns():
    rng = np.random.default_rng(0)
    return rng.normal(0.0003, 0.015, (1260, 5))


class _Mom(Strategy):
    rebalance_frequency = "monthly"
    risk = RiskConfig(max_position=0.30, max_gross=1.0, max_net=1.0)

    def __init__(self, lookback: int = 60):
        self.lookback = lookback

    def __repr__(self):
        return f"_Mom(lookback={self.lookback})"

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


EXPECTED_PROBE_NAMES = [
    "iid_gaussian", "momentum", "mean_reversion",
    "vol_clustering", "jump_diffusion", "cross_section",
]


# --- ProbeBattery --------------------------------------------------------

def test_battery_returns_six_probes(battery):
    probes = battery.probes()
    assert len(probes) == 6
    assert [p.name for p in probes] == EXPECTED_PROBE_NAMES


def test_each_probe_has_valid_panel(battery):
    for probe in battery.probes():
        assert isinstance(probe, Probe)
        assert isinstance(probe.panel, DataPanel)
        assert len(probe.panel.dates) == battery.n_steps
        assert len(probe.panel.assets_all) == battery.n_assets
        assert probe.isolates


def test_battery_reproducibility():
    a = ProbeBattery(n_assets=5, n_steps=200, seed=42)
    b = ProbeBattery(n_assets=5, n_steps=200, seed=42)
    for pa, pb in zip(a.probes(), b.probes()):
        np.testing.assert_array_equal(
            pa.panel._prices.values, pb.panel._prices.values
        )


def test_battery_id_stable_within_params():
    a = ProbeBattery(n_assets=5, n_steps=200, seed=42)
    b = ProbeBattery(n_assets=5, n_steps=200, seed=42)
    assert a.battery_id() == b.battery_id()


def test_battery_id_changes_with_params():
    a = ProbeBattery(n_assets=5, n_steps=200, seed=42)
    b = ProbeBattery(n_assets=5, n_steps=200, seed=43)
    c = ProbeBattery(n_assets=5, n_steps=300, seed=42)
    d = ProbeBattery(n_assets=4, n_steps=200, seed=42)
    ids = {a.battery_id(), b.battery_id(), c.battery_id(), d.battery_id()}
    assert len(ids) == 4


def test_baseline_is_iid_gaussian(battery):
    assert battery.baseline().name == "iid_gaussian"


def test_battery_validation():
    with pytest.raises(ValueError, match="n_assets"):
        ProbeBattery(n_assets=1)
    with pytest.raises(ValueError, match="n_steps"):
        ProbeBattery(n_steps=10)


# --- fingerprint() -------------------------------------------------------

def test_fingerprint_returns_score_per_probe(battery):
    scores = fingerprint(_Mom(60), battery=battery)
    assert set(scores.keys()) == set(EXPECTED_PROBE_NAMES)


def test_fingerprint_baseline_is_zero(battery):
    scores = fingerprint(_Mom(60), battery=battery)
    assert scores["iid_gaussian"] == 0.0


def test_fingerprint_momentum_signal_detected(battery):
    scores = fingerprint(_Mom(60), battery=battery)
    # Cross-sectional momentum profits on persistent alphas, loses on mean-reversion.
    assert scores["momentum"] > scores["mean_reversion"]


def test_fingerprint_caches(tmp_path, battery):
    cache = FingerprintCache(tmp_path / "fp_cache")
    a = fingerprint(_Mom(60), battery=battery, cache=cache)
    b = fingerprint(_Mom(60), battery=battery, cache=cache)
    assert a == b
    files = list((tmp_path / "fp_cache").glob("*.json"))
    assert len(files) == 1


def test_fingerprint_cache_misses_on_different_battery_id(tmp_path):
    cache = FingerprintCache(tmp_path / "fp_cache")
    b1 = ProbeBattery(n_assets=5, n_steps=200, seed=42)
    b2 = ProbeBattery(n_assets=5, n_steps=200, seed=43)
    fingerprint(_Mom(60), battery=b1, cache=cache)
    fingerprint(_Mom(60), battery=b2, cache=cache)
    files = list((tmp_path / "fp_cache").glob("*.json"))
    assert len(files) == 2


def test_fingerprint_cache_round_trip(tmp_path, battery):
    cache = FingerprintCache(tmp_path / "fp_cache")
    scores = fingerprint(_Mom(60), battery=battery, cache=cache)
    files = list((tmp_path / "fp_cache").glob("*.json"))
    loaded = json.loads(files[0].read_text())
    assert loaded == scores


def test_fingerprint_factory_strategy(battery):
    scores_inst = fingerprint(_Mom(60), battery=battery)
    scores_fact = fingerprint(lambda: _Mom(60), battery=battery)
    assert scores_inst == scores_fact


# --- select_generators() -------------------------------------------------

def test_select_always_includes_baseline_and_null(real_returns):
    gens = select_generators({}, real_returns)
    assert "gaussian" in gens
    assert "permutation" in gens


def test_select_zero_scores_no_extra(real_returns):
    scores = {name: 0.0 for name in EXPECTED_PROBE_NAMES}
    gens = select_generators(scores, real_returns)
    assert set(gens.keys()) == {"gaussian", "permutation"}


def test_select_adds_block_bootstrap_above_threshold(real_returns):
    scores = {"iid_gaussian": 0.0, "momentum": 1.0, "mean_reversion": 0.0,
              "vol_clustering": 0.0, "jump_diffusion": 0.0, "cross_section": 0.0}
    gens = select_generators(scores, real_returns, threshold=0.5)
    assert "block_bootstrap" in gens


def test_select_adds_jumps_when_jump_probe_high(real_returns):
    scores = {"iid_gaussian": 0.0, "momentum": 0.0, "mean_reversion": 0.0,
              "vol_clustering": 0.0, "jump_diffusion": 1.0, "cross_section": 0.0}
    gens = select_generators(scores, real_returns, threshold=0.5)
    assert "block_bootstrap_jumps" in gens
    assert "block_bootstrap" in gens
    assert isinstance(gens["block_bootstrap_jumps"], JumpOverlayAdapter)


def test_select_threshold_respected(real_returns):
    scores = {"iid_gaussian": 0.0, "momentum": 0.3,
              "mean_reversion": 0.0, "vol_clustering": 0.0,
              "jump_diffusion": 0.0, "cross_section": 0.0}
    gens_tight = select_generators(scores, real_returns, threshold=0.5)
    assert set(gens_tight.keys()) == {"gaussian", "permutation"}
    gens_loose = select_generators(scores, real_returns, threshold=0.1)
    assert "block_bootstrap" in gens_loose


def test_select_validates_input():
    with pytest.raises(ValueError, match="2-D"):
        select_generators({}, np.zeros((10,)))


def test_select_adds_garch_when_vol_clustering_high(real_returns):
    from backtest.simulation.generators import GarchGenerator
    scores = {"iid_gaussian": 0.0, "momentum": 0.0, "mean_reversion": 0.0,
              "vol_clustering": 1.0, "jump_diffusion": 0.0, "cross_section": 0.0}
    gens = select_generators(scores, real_returns, threshold=0.5)
    assert "garch" in gens
    assert isinstance(gens["garch"], GarchGenerator)


def test_select_adds_multivariate_when_cross_section_high(real_returns):
    from backtest.simulation.generators import MultivariateGenerator
    scores = {"iid_gaussian": 0.0, "momentum": 0.0, "mean_reversion": 0.0,
              "vol_clustering": 0.0, "jump_diffusion": 0.0, "cross_section": 1.0}
    gens = select_generators(scores, real_returns, threshold=0.5)
    assert "multivariate" in gens
    assert isinstance(gens["multivariate"], MultivariateGenerator)


def test_select_skips_multivariate_for_single_asset():
    rng = np.random.default_rng(99)
    single_asset = rng.normal(0, 0.01, (500, 1))
    scores = {"iid_gaussian": 0.0, "momentum": 0.0, "mean_reversion": 0.0,
              "vol_clustering": 0.0, "jump_diffusion": 0.0, "cross_section": 1.0}
    gens = select_generators(scores, single_asset, threshold=0.5)
    assert "multivariate" not in gens


def test_select_garch_and_multivariate_below_threshold(real_returns):
    scores = {"iid_gaussian": 0.0, "momentum": 0.0, "mean_reversion": 0.0,
              "vol_clustering": 0.3, "jump_diffusion": 0.0, "cross_section": 0.3}
    gens = select_generators(scores, real_returns, threshold=0.5)
    assert "garch" not in gens
    assert "multivariate" not in gens


def test_monte_carlo_aggregate_metrics_includes_extended_keys(real_returns):
    from backtest.simulation import MonteCarloEngine
    from backtest.simulation.generators import GaussianGenerator

    idx = pd.date_range("2020-01-01", periods=real_returns.shape[0], freq="B")
    assets = [f"A{i}" for i in range(real_returns.shape[1])]

    class _AllCash(Strategy):
        rebalance_frequency = "weekly"

        def generate_weights(self, data, t):
            return pd.Series(0.0, index=data.assets)

    mc = MonteCarloEngine(
        strategy=_AllCash(),
        generators={"gaussian": GaussianGenerator.fit(real_returns)},
        assets=assets, dates=idx, warmup_bars=10,
    ).run(n_paths=4, seed=0, n_jobs=1)
    agg = mc.aggregate_metrics()
    for k in ("hit_rate", "turnover", "information_ratio", "beta",
              "ann_return", "ann_vol", "total_return", "total_pnl",
              "total_costs", "longest_underwater"):
        for s in ("mean", "std", "min", "max"):
            assert f"{k}_{s}" in agg["gaussian"], f"missing {k}_{s}"

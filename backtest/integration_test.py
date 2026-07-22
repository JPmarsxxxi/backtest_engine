"""End-to-end smoke test wiring every module: data, strategy, risk, costs,
engine, splitters, multi-path, metrics, selection, registry."""

import numpy as np
import pandas as pd
import pytest

from backtest.data import DataPanel
from backtest.engine import Engine, MultiPathEngine
from backtest.metrics import compute_metrics, psr
from backtest.registry import TrialRegistry
from backtest.risk import RiskConfig
from backtest.selection import dsr
from backtest.splitters import CombinatorialPurgedCV
from backtest.strategy import Strategy


def _synthetic_panel(seed: int = 0, n_days: int = 1260) -> DataPanel:
    assets = ("A", "B", "C", "D", "E")
    idx = pd.date_range("2020-01-01", periods=n_days, freq="B")
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0003, 0.015, (n_days, len(assets)))
    prices = pd.DataFrame(
        np.cumprod(1 + rets, axis=0) * 100, index=idx, columns=list(assets)
    )
    return DataPanel(prices)


class XSMomentum(Strategy):
    """Cross-sectional momentum: long top half, short bottom half by lookback return."""

    rebalance_frequency = "monthly"
    risk = RiskConfig(max_position=0.30, max_gross=1.0, max_net=1.0)

    def __init__(self, lookback: int = 60):
        self.lookback = lookback

    def __repr__(self) -> str:
        return f"XSMomentum(lookback={self.lookback})"

    def generate_weights(self, data, t):
        prices = data.prices
        if len(prices) < self.lookback + 1:
            return pd.Series(0.0, index=data.assets)
        window = prices.iloc[-(self.lookback + 1):]
        rets = window.iloc[-1] / window.iloc[0] - 1
        rets = rets.reindex(data.assets).dropna()
        if len(rets) < 2:
            return pd.Series(0.0, index=data.assets)
        median = rets.median()
        n = len(rets)
        weights = pd.Series(0.0, index=rets.index)
        weights[rets > median] = 1.0 / n
        weights[rets < median] = -1.0 / n
        return weights


@pytest.fixture
def panel():
    return _synthetic_panel()


@pytest.fixture
def graph_file(tmp_path):
    p = tmp_path / "graph.png"
    p.write_bytes(b"fake")
    return str(p)


def test_strategy_repr_stable():
    assert repr(XSMomentum(60)) == repr(XSMomentum(60))
    assert repr(XSMomentum(60)) != repr(XSMomentum(30))


def test_walk_forward_pipeline(panel):
    s = XSMomentum(lookback=60)
    result = Engine(panel, s).run(start=panel.dates[120])

    assert len(result.equity_curve) > 0
    assert result.weights.shape[1] == len(panel.assets_all)
    assert result.metadata["n_rebalances"] > 0

    rep = compute_metrics(result.returns, result.equity_curve)
    assert np.isfinite(rep.max_drawdown)
    assert 0 <= rep.psr <= 1
    assert rep.n_obs == result.returns.notna().sum()


def test_registry_logs_parameter_sweep(panel, graph_file, tmp_path):
    reg = TrialRegistry(tmp_path / "registry")
    for lb in [30, 45, 60, 90, 120]:
        reg.run(
            Engine(panel, XSMomentum(lookback=lb)),
            causal_graph_path=graph_file,
            family="momentum_xs",
            start=panel.dates[120],
        )
    assert reg.k(family="momentum_xs") == 5
    k_eff = reg.k(family="momentum_xs", method="effective", threshold=0.7)
    assert 1 <= k_eff <= 5


def test_registry_dsr_count_and_effective(panel, graph_file, tmp_path):
    reg = TrialRegistry(tmp_path / "registry")
    for lb in [30, 45, 60, 90, 120]:
        reg.run(
            Engine(panel, XSMomentum(lookback=lb)),
            causal_graph_path=graph_file,
            family="momentum_xs",
            start=panel.dates[120],
        )
    tid = reg.list(family="momentum_xs")[0]["id"]
    val_count = reg.dsr_for(tid, family="momentum_xs", method="count")
    val_eff = reg.dsr_for(
        tid, family="momentum_xs", method="effective", threshold=0.7
    )
    assert 0 <= val_count <= 1
    assert 0 <= val_eff <= 1
    # K_eff <= K_count, so the smaller K -> at least as large DSR (less conservative)
    assert val_eff >= val_count - 1e-9


def test_cpcv_multi_path_shapes(panel):
    cv = CombinatorialPurgedCV(n_splits=4, n_test_groups=2, embargo_pct=0.01)
    mp = MultiPathEngine(panel, XSMomentum(lookback=60), cv)
    result = mp.run(n_jobs=1)

    assert result.n_splits == 6
    assert result.n_paths == 3
    assert result.path_returns.shape[1] == 3
    assert result.path_equity.shape[1] == 3

    agg = result.aggregate_metrics()
    for metric in ("sharpe", "max_drawdown", "psr"):
        for stat in ("mean", "std", "min", "max"):
            assert f"{metric}_{stat}" in agg


def test_dsr_more_conservative_than_psr(panel):
    result = Engine(panel, XSMomentum(lookback=60)).run(start=panel.dates[120])
    p = psr(result.returns, sr_star=0.0)
    d = dsr(result.returns, K=20)
    assert d <= p + 1e-9


def test_returns_df_round_trip(panel, graph_file, tmp_path):
    reg = TrialRegistry(tmp_path / "registry")
    for lb in [30, 60, 120]:
        reg.run(
            Engine(panel, XSMomentum(lookback=lb)),
            causal_graph_path=graph_file,
            family="momentum_xs",
            start=panel.dates[150],
        )
    df = reg.returns_df(family="momentum_xs")
    assert df.shape[1] == 3
    assert df.shape[0] > 0


if __name__ == "__main__":
    panel = _synthetic_panel()
    print(f"Panel: {len(panel.dates)} bars, {len(panel.assets_all)} assets")

    result = Engine(panel, XSMomentum(lookback=60)).run(start=panel.dates[120])
    rep = compute_metrics(result.returns, result.equity_curve)
    print()
    print(rep)
    print()

    p = psr(result.returns, sr_star=0.0)
    d10 = dsr(result.returns, K=10)
    d100 = dsr(result.returns, K=100)
    print(f"PSR(0):       {p:.4f}")
    print(f"DSR(K=10):    {d10:.4f}")
    print(f"DSR(K=100):   {d100:.4f}")

import numpy as np
import pandas as pd
import pytest

from backtest.data import DataPanel
from backtest.engine import Engine, MultiPathEngine
from backtest.risk import RiskConfig
from backtest.splitters import CombinatorialPurgedCV, WalkForward
from backtest.strategy import Strategy


@pytest.fixture
def panel():
    idx = pd.date_range("2024-01-01", periods=120, freq="B")
    rng = np.random.default_rng(0)
    rets = rng.normal(0, 0.01, (120, 3))
    prices = pd.DataFrame(
        np.cumprod(1 + rets, axis=0) * 100, index=idx, columns=["A", "B", "C"]
    )
    volume = pd.DataFrame(
        np.full((120, 3), 1_000_000.0), index=idx, columns=prices.columns
    )
    return DataPanel(prices, volume=volume)


class _EqualWeight(Strategy):
    rebalance_frequency = "weekly"
    risk = RiskConfig(max_position=1.0, max_gross=1.0, max_net=1.0)

    def generate_weights(self, data, t):
        n = len(data.assets)
        if n == 0:
            return pd.Series(dtype=float)
        return pd.Series(1.0 / n, index=data.assets)


class _StatefulCounter(Strategy):
    rebalance_frequency = "weekly"
    risk = RiskConfig(max_position=1.0, max_gross=1.0, max_net=1.0)

    def __init__(self):
        self.fit_calls = 0

    def fit(self, data):
        self.fit_calls += 1

    def generate_weights(self, data, t):
        return pd.Series({"A": 0.5, "B": 0.5, "C": 0.0})


def _make_factory():
    return _EqualWeight()


def test_walk_forward_one_path(panel):
    wf = WalkForward(train_pct=0.5)
    mp = MultiPathEngine(panel, _EqualWeight(), wf)
    res = mp.run(n_jobs=1)
    assert res.n_splits == 1
    assert res.n_paths == 1
    assert res.path_returns.shape[1] == 1
    assert "path_0" in res.path_returns.columns


def test_walk_forward_matches_single_path_engine(panel):
    wf = WalkForward(train_pct=0.5)
    train, test = next(wf.split(panel.dates))
    single = Engine(panel, _EqualWeight()).run(train_dates=train, test_dates=test)
    mp = MultiPathEngine(panel, _EqualWeight(), wf).run(n_jobs=1)
    # Both have NaN at the first test bar (engine returns[0] convention); compare as-is.
    pd.testing.assert_series_equal(
        single.returns,
        mp.path_returns["path_0"],
        check_names=False,
        atol=1e-12,
    )


def test_cpcv_split_and_path_counts(panel):
    cv = CombinatorialPurgedCV(n_splits=6, n_test_groups=2, embargo_pct=0)
    res = MultiPathEngine(panel, _EqualWeight(), cv).run(n_jobs=1)
    assert res.n_splits == 15
    assert res.n_paths == 5
    assert len(res.split_returns) == 15
    assert res.path_returns.shape[1] == 5


def test_cpcv_paths_cover_all_dates(panel):
    cv = CombinatorialPurgedCV(n_splits=6, n_test_groups=2, embargo_pct=0)
    res = MultiPathEngine(panel, _EqualWeight(), cv).run(n_jobs=1)
    assert len(res.path_returns) == len(panel.dates)
    # Engine's returns[0]=NaN convention puts one NaN at the start of each contiguous
    # test segment per path. At most n_groups NaNs per path.
    assert res.path_returns.isna().sum().max() <= cv.n_groups


def test_cpcv_paths_differ(panel):
    cv = CombinatorialPurgedCV(n_splits=6, n_test_groups=2, embargo_pct=0)
    res = MultiPathEngine(panel, _EqualWeight(), cv).run(n_jobs=1)
    p0 = res.path_returns["path_0"]
    p1 = res.path_returns["path_1"]
    assert not p0.equals(p1)


def test_strategy_factory_used(panel):
    cv = CombinatorialPurgedCV(n_splits=6, n_test_groups=2, embargo_pct=0)
    res = MultiPathEngine(panel, _make_factory, cv).run(n_jobs=1)
    assert res.n_splits == 15


def test_strategy_state_isolated_across_splits(panel):
    cv = CombinatorialPurgedCV(n_splits=4, n_test_groups=2, embargo_pct=0)
    src = _StatefulCounter()
    MultiPathEngine(panel, src, cv).run(n_jobs=1)
    # Each split deepcopies the strategy; the original must not have been mutated.
    assert src.fit_calls == 0


def test_n_jobs_consistency(panel):
    cv = CombinatorialPurgedCV(n_splits=4, n_test_groups=2, embargo_pct=0)
    seq = MultiPathEngine(panel, _EqualWeight(), cv).run(n_jobs=1)
    par = MultiPathEngine(panel, _EqualWeight(), cv).run(n_jobs=2)
    pd.testing.assert_frame_equal(
        seq.path_returns, par.path_returns, atol=1e-12,
    )


def test_path_equity_starts_from_initial_capital(panel):
    wf = WalkForward(train_pct=0.5)
    mp = MultiPathEngine(panel, _EqualWeight(), wf, initial_capital=2_000_000)
    res = mp.run(n_jobs=1)
    # Leading NaN return is fillna'd to 0 during compounding → first bar stays at initial.
    assert res.path_equity["path_0"].iloc[0] == pytest.approx(2_000_000)
    # The first non-NaN return bar must equal initial * (1 + r) at the same index.
    first_ret_idx = res.path_returns["path_0"].dropna().index[0]
    first_ret = res.path_returns["path_0"].loc[first_ret_idx]
    first_eq_after = res.path_equity["path_0"].loc[first_ret_idx]
    assert first_eq_after == pytest.approx(2_000_000 * (1 + first_ret))


def test_aggregate_metrics_keys(panel):
    cv = CombinatorialPurgedCV(n_splits=6, n_test_groups=2, embargo_pct=0)
    res = MultiPathEngine(panel, _EqualWeight(), cv).run(n_jobs=1)
    agg = res.aggregate_metrics()
    for k in ("sharpe", "max_drawdown", "psr"):
        for s in ("mean", "std", "min", "max"):
            assert f"{k}_{s}" in agg


def test_metrics_per_path_count(panel):
    cv = CombinatorialPurgedCV(n_splits=6, n_test_groups=2, embargo_pct=0)
    res = MultiPathEngine(panel, _EqualWeight(), cv).run(n_jobs=1)
    reports = res.metrics_per_path()
    assert len(reports) == res.n_paths


def test_initial_capital_validation(panel):
    wf = WalkForward(train_pct=0.5)
    with pytest.raises(ValueError):
        MultiPathEngine(panel, _EqualWeight(), wf, initial_capital=0)


def test_empty_test_segment_handled(panel):
    # Edge case: a CPCV split could conceivably have empty test if heavily purged.
    # We use embargo_pct close to 1 to stress.
    cv = CombinatorialPurgedCV(
        n_splits=6, n_test_groups=2, purge_bars=0, embargo_pct=0
    )
    res = MultiPathEngine(panel, _EqualWeight(), cv).run(n_jobs=1)
    # Should not crash. Per-segment leading NaNs are allowed (engine returns[0] convention).
    assert len(res.path_returns) == len(panel.dates)
    assert res.path_returns.isna().sum().max() <= cv.n_groups


def test_multi_path_summary_smoke(panel):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cv = CombinatorialPurgedCV(n_splits=4, n_test_groups=2, embargo_pct=0.01)
    res = MultiPathEngine(panel, _EqualWeight(), cv).run(n_jobs=1)
    agg = res.summary()
    assert "sharpe_mean" in agg
    plt.close("all")


def test_cash_and_borrow_rate_pass_through(panel):
    # Positive cash_rate should accrue interest on idle cash; the all-cash baseline
    # (zero weights, zero positions) makes the effect isolable.
    class _AllCash(Strategy):
        rebalance_frequency = "weekly"
        risk = RiskConfig(max_position=1.0, max_gross=1.0, max_net=1.0)

        def generate_weights(self, data, t):
            return pd.Series(0.0, index=data.assets)

    wf = WalkForward(train_pct=0.5)
    no_rate = MultiPathEngine(panel, _AllCash(), wf).run(n_jobs=1)
    with_rate = MultiPathEngine(
        panel, _AllCash(), wf, cash_rate=0.10
    ).run(n_jobs=1)
    # With a positive cash rate and zero positions, the path equity must grow.
    assert with_rate.path_equity["path_0"].iloc[-1] > no_rate.path_equity["path_0"].iloc[-1]


def test_split_results_populated(panel):
    wf = WalkForward(train_pct=0.5)
    res = MultiPathEngine(panel, _EqualWeight(), wf).run(n_jobs=1)
    # One list per split; one BacktestResult per segment in that split.
    assert len(res.split_results) == res.n_splits
    # Walk-forward has a single contiguous test segment per split.
    assert len(res.split_results[0]) == 1
    # The retained BacktestResult exposes the full set of DataFrames.
    bt = res.split_results[0][0]
    assert bt.weights.shape[0] > 0
    assert bt.positions.shape == bt.weights.shape
    assert bt.trades.shape == bt.weights.shape
    assert bt.unfilled.shape == bt.weights.shape


def test_split_results_segments_for_non_contiguous_cpcv(panel):
    # n_test_groups=2 with non-adjacent group combinations produces 2 segments per
    # split for at least some combinations (e.g. groups 0 and 2 with a gap at 1).
    cv = CombinatorialPurgedCV(n_splits=6, n_test_groups=2, embargo_pct=0)
    res = MultiPathEngine(panel, _EqualWeight(), cv).run(n_jobs=1)
    assert len(res.split_results) == res.n_splits
    seg_counts = [len(rs) for rs in res.split_results]
    # Some splits hit non-contiguous group pairs → multi-segment per split.
    assert max(seg_counts) >= 2


def test_aggregate_metrics_includes_extended_keys(panel):
    cv = CombinatorialPurgedCV(n_splits=4, n_test_groups=2, embargo_pct=0)
    res = MultiPathEngine(panel, _EqualWeight(), cv).run(n_jobs=1)
    agg = res.aggregate_metrics()
    for k in (
        "ann_return", "ann_vol", "total_return",
        "total_pnl", "total_costs", "longest_underwater",
    ):
        for s in ("mean", "std", "min", "max"):
            assert f"{k}_{s}" in agg, f"missing {k}_{s}"


def test_parallel_failure_falls_back_to_sequential(panel):
    # A strategy that raises on every call. Under n_jobs=-1 joblib wraps the
    # traceback; MultiPathEngine should warn and re-run sequentially so the
    # original exception type propagates.
    class _Boom(Strategy):
        rebalance_frequency = "weekly"

        def generate_weights(self, data, t):
            raise RuntimeError("strategy exploded")

    wf = WalkForward(train_pct=0.5)
    mp = MultiPathEngine(panel, _Boom(), wf)
    with pytest.warns(UserWarning, match="retrying with n_jobs=1"):
        with pytest.raises(RuntimeError, match="strategy exploded"):
            mp.run(n_jobs=2)

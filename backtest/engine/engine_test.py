import numpy as np
import pandas as pd
import pytest

from backtest.costs import (
    Commission,
    CompositeCostModel,
    LiquidityCap,
    Spread,
)
from backtest.data import DataPanel
from backtest.engine import Engine
from backtest.risk import RiskConfig
from backtest.strategy import Strategy


@pytest.fixture
def simple_panel():
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    prices = pd.DataFrame(
        {"A": [100, 110, 110, 110, 110], "B": [100, 100, 100, 100, 100]},
        index=idx, dtype=float,
    )
    volume = pd.DataFrame(np.full((5, 2), 1_000_000.0), index=idx, columns=prices.columns)
    return DataPanel(prices, volume=volume)


@pytest.fixture
def long_panel():
    idx = pd.date_range("2024-01-01", periods=60, freq="B")
    rng = np.random.default_rng(0)
    rets = rng.normal(0, 0.01, (60, 3))
    prices = pd.DataFrame(
        np.cumprod(1 + rets, axis=0) * 100,
        index=idx, columns=["A", "B", "C"],
    )
    volume = pd.DataFrame(
        np.full((60, 3), 1_000_000.0), index=idx, columns=prices.columns,
    )
    return DataPanel(prices, volume=volume)


class _Constant(Strategy):
    rebalance_frequency = "daily"
    risk = RiskConfig(max_position=1.0, max_gross=2.0, max_net=2.0)

    def __init__(self, w):
        self.w = w

    def generate_weights(self, data, t):
        return pd.Series(self.w)


def test_constant_50_50_first_bar(simple_panel):
    s = _Constant({"A": 0.5, "B": 0.5})
    r = Engine(simple_panel, s).run()
    assert r.equity_curve.iloc[0] == pytest.approx(1_000_000.0)
    assert r.positions.iloc[0]["A"] == pytest.approx(500_000.0)
    assert r.positions.iloc[0]["B"] == pytest.approx(500_000.0)


def test_constant_50_50_drift_then_rebalance(simple_panel):
    s = _Constant({"A": 0.5, "B": 0.5})
    r = Engine(simple_panel, s).run()
    # Day 1: A 100->110 (+10%), B flat. Pre: A=550K, B=500K, equity=1.05M.
    assert r.equity_curve.iloc[1] == pytest.approx(1_050_000.0)
    # After rebalance: 525K each.
    assert r.positions.iloc[1]["A"] == pytest.approx(525_000.0)
    assert r.positions.iloc[1]["B"] == pytest.approx(525_000.0)


def test_buy_and_hold(simple_panel):
    class S(_Constant):
        rebalance_frequency = staticmethod(lambda dates: dates[:1])

    s = S({"A": 1.0, "B": 0.0})
    r = Engine(simple_panel, s).run()
    # All-in A on first bar; never trade again. Day 4 price = 110, so equity = 1.1M.
    assert r.equity_curve.iloc[-1] == pytest.approx(1_100_000.0)
    assert r.metadata["n_rebalances"] == 1


def test_no_costs_no_liquidity_default(simple_panel):
    s = _Constant({"A": 0.5, "B": 0.5})
    r = Engine(simple_panel, s).run()
    assert (r.costs == 0).all()
    assert (r.unfilled == 0).all().all()


def test_costs_reduce_equity(simple_panel):
    s = _Constant({"A": 0.5, "B": 0.5})
    cm = CompositeCostModel([Commission(bps=10), Spread(half_bps=5)])
    r = Engine(simple_panel, s, costs=cm).run()
    # Day 0: trade $1M total at 15 bps -> $1500 cost, equity = 998500
    assert r.equity_curve.iloc[0] == pytest.approx(1_000_000 - 1_500)
    assert (r.costs.iloc[0]) == pytest.approx(1_500.0)


def test_risk_clips_max_position(long_panel):
    class S(Strategy):
        rebalance_frequency = "daily"
        risk = RiskConfig(max_position=0.10, max_gross=10, max_net=10)
        def generate_weights(self, data, t):
            return pd.Series({"A": 0.5, "B": 0.5, "C": 0.5})

    r = Engine(long_panel, S()).run()
    assert (r.weights.abs() <= 0.10 + 1e-9).all().all()


def test_liquidity_cap_creates_unfilled(long_panel):
    class Big(Strategy):
        rebalance_frequency = staticmethod(lambda dates: dates[:1])
        risk = RiskConfig(max_position=10, max_gross=10, max_net=10)
        def generate_weights(self, data, t):
            return pd.Series({"A": 5.0, "B": 0.0, "C": 0.0})

    cap = LiquidityCap(cap_pct=0.01)
    r = Engine(long_panel, Big(), liquidity=cap, initial_capital=1_000_000_000).run()
    assert r.unfilled.iloc[0]["A"] != 0


def test_rebalance_dates_honored(long_panel):
    class Weekly(_Constant):
        rebalance_frequency = "weekly"

    s = Weekly({"A": 1/3, "B": 1/3, "C": 1/3})
    r = Engine(long_panel, s).run()
    n_rb = r.metadata["n_rebalances"]
    n_trade_days = (r.trades.abs().sum(axis=1) > 0).sum()
    assert n_trade_days == n_rb
    assert n_rb < len(r.equity_curve)


def test_fit_called_when_train_dates_given(long_panel):
    class Recorder(Strategy):
        rebalance_frequency = "daily"
        risk = RiskConfig(max_position=1.0, max_gross=2.0, max_net=2.0)
        def __init__(self):
            self.fit_calls = 0
        def fit(self, data):
            self.fit_calls += 1
        def generate_weights(self, data, t):
            return pd.Series({"A": 0.0, "B": 0.0, "C": 0.0})

    s = Recorder()
    train = long_panel.dates[:30]
    test = long_panel.dates[30:]
    r = Engine(long_panel, s).run(train_dates=train, test_dates=test)
    assert s.fit_calls == 1
    assert r.metadata["fit_called"]
    assert r.metadata["start"] == test[0]


def test_fit_not_called_without_train(long_panel):
    class Recorder(Strategy):
        rebalance_frequency = "daily"
        def __init__(self):
            self.fit_calls = 0
        def fit(self, data):
            self.fit_calls += 1
        def generate_weights(self, data, t):
            return pd.Series({"A": 0.0, "B": 0.0, "C": 0.0})

    s = Recorder()
    Engine(long_panel, s).run()
    assert s.fit_calls == 0


def test_empty_universe_no_trading(long_panel):
    u = pd.DataFrame(False, index=long_panel.dates, columns=long_panel.assets_all)
    panel = DataPanel(long_panel._prices, volume=long_panel._volume, universe=u)
    s = _Constant({"A": 1.0, "B": 0.0, "C": 0.0})
    r = Engine(panel, s).run()
    assert (r.trades == 0).all().all()
    assert r.equity_curve.iloc[-1] == pytest.approx(1_000_000.0)


def test_weights_consistent_with_positions(long_panel):
    s = _Constant({"A": 0.4, "B": 0.3, "C": 0.3})
    r = Engine(long_panel, s).run()
    reconstructed = r.weights.multiply(r.equity_curve, axis=0)
    pd.testing.assert_frame_equal(
        reconstructed, r.positions, check_exact=False, atol=1e-6,
    )


def test_initial_capital_must_be_positive(simple_panel):
    s = _Constant({"A": 0.5, "B": 0.5})
    with pytest.raises(ValueError):
        Engine(simple_panel, s, initial_capital=0)
    with pytest.raises(ValueError):
        Engine(simple_panel, s, initial_capital=-100)


def test_returns_match_equity_diff(long_panel):
    s = _Constant({"A": 0.5, "B": 0.5, "C": 0.0})
    r = Engine(long_panel, s).run()
    expected = r.equity_curve.pct_change()  # NaN at index 0, matches engine convention
    pd.testing.assert_series_equal(
        r.returns, expected, check_names=False, atol=1e-12,
    )
    assert pd.isna(r.returns.iloc[0])


def test_start_end_filtering(long_panel):
    s = _Constant({"A": 0.5, "B": 0.5, "C": 0.0})
    start = long_panel.dates[10]
    end = long_panel.dates[40]
    r = Engine(long_panel, s).run(start=start, end=end)
    assert r.equity_curve.index[0] == start
    assert r.equity_curve.index[-1] == end


def test_summary_smoke(long_panel):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = _Constant({"A": 0.5, "B": 0.5, "C": 0.0})
    r = Engine(long_panel, s).run()
    rep = r.summary(rolling_window=20)
    assert rep.n_obs == r.returns.notna().sum()
    plt.close("all")


def test_pnl_property_sums_to_equity_change(long_panel):
    s = _Constant({"A": 0.5, "B": 0.5, "C": 0.0})
    r = Engine(long_panel, s).run()
    pnl_total = r.pnl.sum()
    eq_change = r.equity_curve.iloc[-1] - r.metadata["initial_capital"]
    assert pnl_total == pytest.approx(eq_change, rel=1e-9)


def test_gross_pnl_equals_pnl_plus_costs(simple_panel):
    s = _Constant({"A": 0.5, "B": 0.5})
    cm = CompositeCostModel([Commission(bps=10), Spread(half_bps=5)])
    r = Engine(simple_panel, s, costs=cm).run()
    assert (r.gross_pnl - r.pnl - r.costs).abs().max() < 1e-9
    assert r.costs.sum() > 0


def test_required_data_default_passes(simple_panel):
    s = _Constant({"A": 0.5, "B": 0.5})
    Engine(simple_panel, s)


def test_required_data_missing_feature_raises(simple_panel):
    class S(_Constant):
        def required_data(self):
            return {"prices": None, "eps": None}

    with pytest.raises(ValueError, match="eps"):
        Engine(simple_panel, S({"A": 0.5, "B": 0.5}))


def test_required_data_volume_missing_raises():
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    prices = pd.DataFrame(
        {"A": [100, 110, 110, 110, 110]}, index=idx, dtype=float,
    )
    panel_no_volume = DataPanel(prices)

    class S(_Constant):
        def required_data(self):
            return {"prices": None, "volume": None}

    with pytest.raises(ValueError, match="volume"):
        Engine(panel_no_volume, S({"A": 1.0}))


def test_required_data_feature_present_passes():
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    prices = pd.DataFrame(
        {"A": [100, 110, 110, 110, 110]}, index=idx, dtype=float,
    )
    eps = pd.DataFrame({"A": [1, 1, 2, 2, 2]}, index=idx, dtype=float)
    panel = DataPanel(prices, features={"eps": eps})

    class S(_Constant):
        def required_data(self):
            return {"prices": None, "eps": None}

    Engine(panel, S({"A": 1.0}))


def test_liquidity_cap_without_volume_raises():
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    prices = pd.DataFrame(
        {"A": [100, 110, 110, 110, 110]}, index=idx, dtype=float,
    )
    panel_no_volume = DataPanel(prices)
    s = _Constant({"A": 1.0})
    with pytest.raises(ValueError, match="LiquidityCap"):
        Engine(panel_no_volume, s, liquidity=LiquidityCap(cap_pct=0.10))


def test_liquidity_cap_with_volume_passes(simple_panel):
    s = _Constant({"A": 0.5, "B": 0.5})
    Engine(simple_panel, s, liquidity=LiquidityCap(cap_pct=0.10))

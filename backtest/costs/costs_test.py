import numpy as np
import pandas as pd
import pytest

from backtest.costs import (
    Commission,
    CompositeCostModel,
    LiquidityCap,
    MarketImpact,
    ShortBorrow,
    Spread,
)
from backtest.data import DataPanel


@pytest.fixture
def panel():
    idx = pd.date_range("2024-01-01", periods=40, freq="B")
    prices = pd.DataFrame(
        np.full((40, 3), [100.0, 50.0, 200.0]),
        index=idx,
        columns=["A", "B", "C"],
    )
    volume = pd.DataFrame(
        np.full((40, 3), [1_000_000.0, 500_000.0, 100_000.0]),
        index=idx,
        columns=["A", "B", "C"],
    )
    return DataPanel(prices, volume=volume)


@pytest.fixture
def view(panel):
    return panel.as_of(panel.dates[-1])


def test_commission_bps(view):
    c = Commission(bps=5)
    trades = pd.Series([10_000.0, -5_000.0, 0.0], index=["A", "B", "C"])
    assert c.trade_cost(trades, view) == pytest.approx(15_000.0 * 0.0005)


def test_commission_zero_trades(view):
    c = Commission(bps=10)
    trades = pd.Series([0.0, 0.0, 0.0], index=["A", "B", "C"])
    assert c.trade_cost(trades, view) == 0.0


def test_commission_per_share(view):
    c = Commission(per_share=0.005)
    trades = pd.Series([10_000.0, -5_000.0, 0.0], index=["A", "B", "C"])
    expected = (10_000 / 100 + 5_000 / 50) * 0.005
    assert c.trade_cost(trades, view) == pytest.approx(expected)


def test_commission_requires_exactly_one_arg():
    with pytest.raises(ValueError):
        Commission()
    with pytest.raises(ValueError):
        Commission(bps=5, per_share=0.005)


def test_spread_scalar(view):
    s = Spread(half_bps=2)
    trades = pd.Series([10_000.0, -10_000.0, 0.0], index=["A", "B", "C"])
    assert s.trade_cost(trades, view) == pytest.approx(20_000.0 * 0.0002)


def test_spread_per_asset(view):
    s = Spread(half_bps={"A": 1, "B": 5, "C": 0})
    trades = pd.Series([10_000.0, 10_000.0, 10_000.0], index=["A", "B", "C"])
    expected = 10_000 * (1 + 5 + 0) / 10_000
    assert s.trade_cost(trades, view) == pytest.approx(expected)


def test_spread_missing_asset_zero(view):
    s = Spread(half_bps={"A": 5})
    trades = pd.Series([10_000.0, 10_000.0, 10_000.0], index=["A", "B", "C"])
    assert s.trade_cost(trades, view) == pytest.approx(10_000 * 5 / 10_000)


def test_spread_series_input(view):
    rates = pd.Series([1.0, 5.0, 0.0], index=["A", "B", "C"])
    s = Spread(half_bps=rates)
    trades = pd.Series([10_000.0, 10_000.0, 10_000.0], index=["A", "B", "C"])
    expected = 10_000 * (1 + 5 + 0) / 10_000
    assert s.trade_cost(trades, view) == pytest.approx(expected)


def test_market_impact_requires_volume():
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    prices = pd.DataFrame(np.ones((10, 2)) * 100, index=idx, columns=["A", "B"])
    p = DataPanel(prices)  # no volume
    v = p.as_of(idx[-1])
    with pytest.raises(ValueError, match="volume"):
        MarketImpact(k=10).trade_cost(pd.Series([1000.0, 0.0], index=["A", "B"]), v)


def test_market_impact_sqrt_exact(view):
    # ADV_$_A = 1_000_000 shares * $100 = $100_000_000
    # ratio    = 1_000_000 / 100_000_000 = 0.01
    # impact_bps = 10 * sqrt(0.01) = 1.0
    # cost = 1_000_000 * 1.0 / 10_000 = 100.0
    mi = MarketImpact(k=10, kind="sqrt")
    trades = pd.Series([1_000_000.0, 0, 0], index=["A", "B", "C"])
    assert mi.trade_cost(trades, view) == pytest.approx(100.0)


def test_market_impact_linear_exact(view):
    # ratio = 0.01, impact_bps = 10 * 0.01 = 0.1, cost = 1_000_000 * 0.1 / 10_000 = 10.0
    mi = MarketImpact(k=10, kind="linear")
    trades = pd.Series([1_000_000.0, 0, 0], index=["A", "B", "C"])
    assert mi.trade_cost(trades, view) == pytest.approx(10.0)


def test_market_impact_sqrt_scales(view):
    mi = MarketImpact(k=10, kind="sqrt")
    small = mi.trade_cost(pd.Series([1_000.0, 0, 0], index=["A", "B", "C"]), view)
    big = mi.trade_cost(pd.Series([10_000_000.0, 0, 0], index=["A", "B", "C"]), view)
    assert big > small * 100


def test_market_impact_invalid_kind():
    with pytest.raises(ValueError):
        MarketImpact(k=10, kind="cubic")


def test_short_borrow_only_on_shorts(view):
    sb = ShortBorrow(annual_bps=200, trading_days=252)
    positions = pd.Series([10_000.0, -10_000.0, 0.0], index=["A", "B", "C"])
    expected_daily = 10_000 * 0.02 / 252
    assert sb.holding_cost(positions, view) == pytest.approx(expected_daily)
    assert sb.trade_cost(positions, view) == 0.0


def test_short_borrow_long_only_zero(view):
    sb = ShortBorrow(annual_bps=500)
    positions = pd.Series([10_000.0, 10_000.0, 10_000.0], index=["A", "B", "C"])
    assert sb.holding_cost(positions, view) == 0.0


def test_composite_sums_trade_costs(view):
    composite = CompositeCostModel([Commission(bps=5), Spread(half_bps=3)])
    trades = pd.Series([10_000.0, 0, 0], index=["A", "B", "C"])
    expected = 10_000 * 0.0005 + 10_000 * 0.0003
    assert composite.trade_cost(trades, view) == pytest.approx(expected)


def test_composite_sums_holding_costs(view):
    composite = CompositeCostModel([
        ShortBorrow(annual_bps=100),
        ShortBorrow(annual_bps=200),
    ])
    positions = pd.Series([0, -10_000.0, 0], index=["A", "B", "C"])
    expected = 10_000 * (0.01 + 0.02) / 252
    assert composite.holding_cost(positions, view) == pytest.approx(expected)


def test_liquidity_cap_caps_oversized(view):
    cap = LiquidityCap(cap_pct=0.10)
    huge = pd.Series([100_000_000.0, 0, 0], index=["A", "B", "C"])
    executed, unfilled = cap.apply(huge, view)
    adv_a = 1_000_000 * 100
    assert executed["A"] == pytest.approx(0.10 * adv_a)
    assert unfilled["A"] == pytest.approx(huge["A"] - executed["A"])


def test_liquidity_cap_passthrough_under_cap(view):
    cap = LiquidityCap(cap_pct=0.10)
    small = pd.Series([100.0, -100.0, 50.0], index=["A", "B", "C"])
    executed, unfilled = cap.apply(small, view)
    pd.testing.assert_series_equal(executed, small)
    assert (unfilled == 0).all()


def test_liquidity_cap_preserves_sign(view):
    cap = LiquidityCap(cap_pct=0.10)
    trades = pd.Series([-100_000_000.0, 0, 0], index=["A", "B", "C"])
    executed, _ = cap.apply(trades, view)
    assert executed["A"] < 0


def test_liquidity_cap_no_volume_passthrough():
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    prices = pd.DataFrame(np.ones((10, 2)) * 100, index=idx, columns=["A", "B"])
    p = DataPanel(prices)
    v = p.as_of(idx[-1])
    cap = LiquidityCap(cap_pct=0.05)
    trades = pd.Series([1e9, -1e9], index=["A", "B"])
    executed, unfilled = cap.apply(trades, v)
    pd.testing.assert_series_equal(executed, trades)
    assert (unfilled == 0).all()


def test_liquidity_cap_invalid_pct():
    with pytest.raises(ValueError):
        LiquidityCap(cap_pct=0)
    with pytest.raises(ValueError):
        LiquidityCap(cap_pct=1.5)

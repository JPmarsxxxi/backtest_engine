import numpy as np
import pandas as pd
import pytest

from backtest.data import DataPanel
from backtest.risk import RiskConfig, RiskManager


@pytest.fixture
def view():
    idx = pd.date_range("2024-01-01", periods=120, freq="B")
    rng = np.random.default_rng(42)
    prices = pd.DataFrame(
        np.cumprod(1 + rng.normal(0, 0.01, (120, 3)), axis=0) * 100,
        index=idx,
        columns=["A", "B", "C"],
    )
    return DataPanel(prices).as_of(idx[100])


def test_max_position_clip(view):
    rm = RiskManager(RiskConfig(max_position=0.10, max_gross=10, max_net=10))
    out = rm.apply(pd.Series([0.5, -0.4, 0.3], index=["A", "B", "C"]), {}, view)
    assert (out.abs() <= 0.10 + 1e-12).all()


def test_max_gross_rescale(view):
    rm = RiskManager(RiskConfig(max_position=None, max_gross=0.5, max_net=10))
    out = rm.apply(pd.Series([0.5, -0.5, 0.5], index=["A", "B", "C"]), {}, view)
    assert pytest.approx(out.abs().sum(), rel=1e-9) == 0.5


def test_max_net_rescale(view):
    rm = RiskManager(RiskConfig(max_position=None, max_gross=10, max_net=0.4))
    out = rm.apply(pd.Series([0.6, 0.6, 0.6], index=["A", "B", "C"]), {}, view)
    assert pytest.approx(abs(out.sum()), rel=1e-9) == 0.4


def test_max_leverage_overrides_gross(view):
    rm = RiskManager(RiskConfig(max_position=None, max_gross=10, max_net=10, max_leverage=2.0))
    out = rm.apply(pd.Series([2.0, -2.0, 2.0], index=["A", "B", "C"]), {}, view)
    assert pytest.approx(out.abs().sum(), rel=1e-9) == 2.0


def test_target_vol_scales(view):
    rm_no_target = RiskManager(
        RiskConfig(max_position=None, max_gross=10, max_net=10)
    )
    rm_target = RiskManager(
        RiskConfig(max_position=None, max_gross=10, max_net=10, target_vol=0.10)
    )
    proposed = pd.Series([1.0, 1.0, 1.0], index=["A", "B", "C"])
    untouched = rm_no_target.apply(proposed, {}, view)
    targeted = rm_target.apply(proposed, {}, view)
    assert not np.allclose(untouched.values, targeted.values)


def test_zero_input_is_zero_output(view):
    rm = RiskManager(RiskConfig())
    out = rm.apply(pd.Series([0.0, 0.0, 0.0], index=["A", "B", "C"]), {}, view)
    assert (out == 0.0).all()

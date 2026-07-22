import numpy as np
import pandas as pd
import pytest

from backtest.data import DataPanel
from backtest.risk import DEFAULT_RISK_CONFIG, RiskConfig
from backtest.strategy import Strategy


@pytest.fixture
def panel():
    idx = pd.date_range("2024-01-01", periods=80, freq="B")
    prices = pd.DataFrame(
        np.cumprod(1 + np.random.default_rng(0).normal(0, 0.01, (80, 3)), axis=0) * 100,
        index=idx,
        columns=["A", "B", "C"],
    )
    return DataPanel(prices)


class _DummyStrategy(Strategy):
    rebalance_frequency = "daily"

    def generate_weights(self, data, t):
        return pd.Series([0.3, -0.2, 0.5], index=["A", "B", "C"])


def test_must_set_rebalance_frequency(panel):
    class Broken(Strategy):
        def generate_weights(self, data, t):
            return pd.Series(0.0, index=["A"])

    with pytest.raises(ValueError, match="rebalance_frequency"):
        Broken().rebalance_dates(panel.dates)


def test_abstract_generate_weights():
    with pytest.raises(TypeError):
        Strategy()


def test_default_risk_is_default_config():
    s = _DummyStrategy()
    assert s.risk is DEFAULT_RISK_CONFIG


def test_required_data_default():
    s = _DummyStrategy()
    assert s.required_data() == {"prices": None}


def test_rebalance_daily(panel):
    s = _DummyStrategy()
    out = s.rebalance_dates(panel.dates)
    assert out.equals(panel.dates)


def test_rebalance_weekly(panel):
    class S(_DummyStrategy):
        rebalance_frequency = "weekly"

    out = S().rebalance_dates(panel.dates)
    weeks = out.isocalendar().week.values
    assert len(np.unique(weeks)) == len(out)


def test_rebalance_monthly(panel):
    class S(_DummyStrategy):
        rebalance_frequency = "monthly"

    out = S().rebalance_dates(panel.dates)
    assert len(np.unique(out.month.values)) == len(out)


def test_rebalance_callable(panel):
    class S(_DummyStrategy):
        rebalance_frequency = staticmethod(lambda dates: dates[::5])

    out = S().rebalance_dates(panel.dates)
    assert len(out) == len(panel.dates[::5])


def test_apply_risk_default_uses_self_risk(panel):
    class S(_DummyStrategy):
        risk = RiskConfig(max_position=0.1)

    s = S()
    proposed = pd.Series([0.5, -0.5, 0.5], index=["A", "B", "C"])
    out = s.apply_risk(proposed, {}, panel.as_of(panel.dates[20]))
    assert (out.abs() <= 0.1 + 1e-12).all()


def test_apply_risk_can_be_overridden(panel):
    class S(_DummyStrategy):
        def apply_risk(self, proposed, state, data):
            return proposed * 0.0

    s = S()
    out = s.apply_risk(
        pd.Series([1.0, 1.0, 1.0], index=["A", "B", "C"]),
        {},
        panel.as_of(panel.dates[20]),
    )
    assert (out == 0.0).all()

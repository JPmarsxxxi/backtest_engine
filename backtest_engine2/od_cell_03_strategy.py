# Cell 3 — Stage 2: paper-window overnight drift (trial 1, pre-registered replication).
# Long US500 2:00-3:00am ET every night (Boyarchenko/Larsen/Whelan SR917 headline window).
# Signal-free unconditional premium harvest; ET-anchored so DST is handled by tz conversion.
import numpy as np
from backtest.strategy import Strategy
from backtest.risk import RiskConfig

ET = "America/New_York"


def _et_hour(dates):
    return dates.tz_localize("UTC").tz_convert(ET)


def _od_rebalance_bars(dates):
    et = _et_hour(dates)
    pick = ((et.hour == 2) | (et.hour == 3)) & (et.minute == 0)
    return dates[pick]


class OvernightDriftWindow(Strategy):
    """SR917 replication: long the 2-3am ET window, flat otherwise."""
    rebalance_frequency = staticmethod(_od_rebalance_bars)

    def __init__(self, gross: float = 1.0):
        self.gross = gross
        self.risk = RiskConfig(max_position=gross, max_gross=1.0)

    def __repr__(self):
        return f"OvernightDriftWindow(window=02:00-03:00ET, long_only, gross={self.gross})"

    def generate_weights(self, data, t):
        et_t = t.tz_localize("UTC").tz_convert(ET)
        if et_t.hour == 2 and et_t.minute == 0:
            return pd.Series({"USA500": self.gross})
        return pd.Series(0.0, index=data.assets)


strat = OvernightDriftWindow(gross=1.0)
print(strat)
rb = strat.rebalance_dates(panel.dates)
years = (panel.dates[-1] - panel.dates[0]).days / 365.25
et_rb = _et_hour(rb)
print(f"Rebalance bars: {len(rb)} total, {len(rb) / years:.0f}/yr (expect ~2x250 = ~500/yr)")
print(f"  entries (ET 2:00): {int((et_rb.hour == 2).sum())}")
print(f"  exits   (ET 3:00): {int((et_rb.hour == 3).sum())}")

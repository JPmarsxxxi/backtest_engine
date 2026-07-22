# Cell 3 — Stage 2: paper-exact intraday TSM strategy.
# Signal: sign of first-half-hour return (00:00->00:30 UTC, close-time stamps).
# Trade: hold sign*gross over the last half-hour only (enter 23:30 bar, flat at 00:00 bar).
# 00:30 is a safety-exit bar: normally already flat -> no trade, no cost; it only
# matters on days where the midnight bar is missing from the grid.
import numpy as np
from backtest.strategy import Strategy
from backtest.risk import RiskConfig


def _itsm_rebalance_bars(dates):
    hm = dates.hour * 60 + dates.minute
    return dates[(hm == 23 * 60 + 30) | (hm == 0) | (hm == 30)]


class IntradayTSM(Strategy):
    """Gao/Han/Li/Zhou (JFE 2018) construction on BTC: first half-hour predicts last half-hour."""
    rebalance_frequency = staticmethod(_itsm_rebalance_bars)

    def __init__(self, gross: float = 0.5):
        self.gross = gross
        self.risk = RiskConfig(max_position=gross, max_gross=1.0)  # fields: 03-risk.md §2

    def __repr__(self):
        return f"IntradayTSM(signal=00:00-00:30, trade=23:30-24:00, gross={self.gross})"

    def generate_weights(self, data, t):
        zero = pd.Series(0.0, index=data.assets)
        if not (t.hour == 23 and t.minute == 30):
            return zero                                      # exit / safety bars: target flat
        day = t.normalize()
        px = data.prices["BTCUSDT"]
        try:
            p_mid = px.at[day]                               # price at 00:00 (close-time stamp)
            p_0030 = px.at[day + pd.Timedelta("30min")]      # price at 00:30
        except KeyError:
            return zero                                      # signal bars missing -> no trade
        if pd.isna(p_mid) or pd.isna(p_0030) or p_0030 == p_mid:
            return zero
        return pd.Series({"BTCUSDT": self.gross * np.sign(p_0030 / p_mid - 1.0)})


strat = IntradayTSM(gross=0.5)
print(strat)
rb = strat.rebalance_dates(panel.dates)
years = (panel.dates[-1] - panel.dates[0]).days / 365.25
print(f"Rebalance bars: {len(rb)} total, {len(rb) / years:.0f}/yr (expect ~3/day ~= 1096/yr)")
print(f"  entry bars (23:30): {int(((rb.hour == 23) & (rb.minute == 30)).sum())}")
print(f"  exit bars  (00:00): {int(((rb.hour == 0) & (rb.minute == 0)).sum())}")
print(f"  safety     (00:30): {int(((rb.hour == 0) & (rb.minute == 30)).sum())}")

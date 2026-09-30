"""EDA#2 costs (Cell 3). Spread = max(FTMO x098 measured half-spread, Abdi-Ranaldo estimate 21d, 5d) per coin per bar;
commission 3.25 bp/side; FTMO swap charged ONCE per night at the rollover with Friday triple (user decisions 2026-09-30)."""
import numpy as np
import pandas as pd

from backtest.costs.base import CostModel
from backtest.costs import spread_source as ss

SWAP_ANNUAL = -0.30                 # FTMO crypto, both directions (x098 menu)
NIGHT = 1.0 / 365.0                 # 8.22 bp per rollover at -30%/yr
# MT5 swap_rollover3days = 5 (Friday): the rollover that ENDS server-Friday charges 3 nights; Sat/Sun rollovers charge 0.
MULT = {0: 1, 1: 1, 2: 1, 3: 1, 4: 3, 5: 0, 6: 0}


def rollover_stamps(index):
    """Bar-close stamps (UTC) at which FTMO's server midnight falls. Server midnight = 17:00 New York (NY-close
    convention) = 21:00 UTC in US summer time, 22:00 UTC in winter. Returns {stamp: multiplier}."""
    ny = index.tz_convert("America/New_York")
    hit = (ny.hour == 17) & (ny.minute == 0)
    stamps = index[hit]
    # the server day that ENDS at this rollover = the New York calendar day of the 17:00 stamp
    days = ny[hit].dayofweek
    return pd.Series([MULT[int(d)] for d in days], index=stamps)


class FTMOSwapNightly(CostModel):
    """Charges |position| * 30%/365 * multiplier on positions held THROUGH the rollover bar. holding_cost is called before
    the bar's rebalance, so a position opened at this bar is not charged and one closed at this bar is."""

    def __init__(self, index, long_rate=SWAP_ANNUAL, short_rate=SWAP_ANNUAL):
        self.mult = rollover_stamps(index)
        self.long_rate, self.short_rate = long_rate, short_rate

    def trade_cost(self, trades, view):
        return 0.0

    def holding_cost(self, positions, view):
        m = self.mult.get(view.t, 0)
        if m == 0:
            return 0.0
        pos = positions[positions != 0]
        if pos.empty:
            return 0.0
        rate = np.where(pos > 0, self.long_rate, self.short_rate)
        return float((-rate * pos.abs() * NIGHT * m).sum())


def daily_ohlc(hourly):
    """hourly: dict coin -> DataFrame[open, high, low, close] on bar-CLOSE stamps. Daily bars labelled by their END stamp
    (00:00 UTC), complete days only (24 bars); incomplete days are NaN (never filled)."""
    out = {k: {} for k in ["open", "high", "low", "close"]}
    for c, df in hourly.items():
        key = df.index.ceil("D")
        g = df.groupby(key)
        n = g["close"].count()
        full = n >= 24
        out["open"][c] = g["open"].first().where(full)
        out["high"][c] = g["high"].max().where(full)
        out["low"][c] = g["low"].min().where(full)
        out["close"][c] = g["close"].last().where(full)
    return {k: pd.DataFrame(v) for k, v in out.items()}


def ar_half_bps(d, window):
    """Abdi-Ranaldo (2017) half-spread, bp. The engine's estimator uses the NEXT day's mid-range (eta.shift(-1)), so the raw
    value labelled E is only known at E+1D. Shifted by one daily row: the value labelled E uses data up to E only."""
    raw = ss.estimate(d["high"], d["low"], d["close"], method="abdi_ranaldo", window=window)
    return raw.shift(1)

"""EDA#2 costs (Cell 3). Spread = max(FTMO x098 measured half-spread, MEASURED Binance half-spread by hour) per coin per bar;
commission 3.25 bp/side; FTMO swap charged ONCE per night at the rollover with Friday triple (user decisions 2026-09-30)."""
import numpy as np
import pandas as pd

from backtest.costs.base import CostModel

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


def measured_half_frames(quotes, index, floor):
    """MEASURED spread (Tardis Binance quotes, 1st of each month) -> two hourly charged half-spread frames (bp).
    For each coin and bar-close stamp t:
      level  = the most recent sampled day that had fully ENDED by t (1st 00:00 -> 2nd 00:00): its median (low) or
               90th percentile (high) of per-minute time-weighted half-spread
      ratio  = hour-of-day shape from sampled days ended by t: median over days of (hour median / day median),
               evaluated at t's hour (the fill happens at t, inside hour t.hour)
      charged = max(FTMO floor, level x ratio)
    Before the first sampled day has ended (all of 2018, early 2019) the FIRST sampled day stands in: a cost
    ASSUMPTION using later data, never a signal input (flagged)."""
    low, high = {}, {}
    for c, q in quotes.items():
        q = q.copy()
        q["day"] = q["t"].dt.floor("D")
        q["hour"] = q["t"].dt.hour
        days = sorted(q["day"].unique())
        lev = q.groupby("day")["half_bp_tw"].agg(med="median", p90=lambda x: x.quantile(0.9))
        hm = q.groupby(["day", "hour"])["half_bp_tw"].median().unstack()
        rel = hm.div(lev["med"], axis=0)                       # per-day hour shape
        known = pd.DatetimeIndex(days) + pd.Timedelta("1D")    # a sampled day is known once it has ended
        prof = rel.expanding().median()                        # shape from days up to and including row k
        prof.index = known
        lev.index = known
        pos = known.searchsorted(index, side="right") - 1      # latest sampled day ended by t
        pos_c = np.clip(pos, 0, len(known) - 1)                # before the first: the first (assumption)
        hrs = index.hour
        ratio = prof.to_numpy()[pos_c, hrs]
        ratio = np.where(np.isfinite(ratio), ratio, 1.0)
        low[c] = np.maximum(floor[c], lev["med"].to_numpy()[pos_c] * ratio)
        high[c] = np.maximum(floor[c], lev["p90"].to_numpy()[pos_c] * ratio)
    lo = pd.DataFrame(low, index=index)
    hi = pd.DataFrame(high, index=index)
    return lo, hi

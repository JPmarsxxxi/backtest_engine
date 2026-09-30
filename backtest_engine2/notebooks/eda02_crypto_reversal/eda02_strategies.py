"""EDA#2 strategies (Step 2 expressions, hunts/2026-09-30_crypto-cost10/step2/EXPRESSIONS.md).

Both read ONLY the DataView they are handed (prices/volume up to t). Stamps are bar CLOSE times (Cell 1), so the daily
boundary E = t.floor("D") (00:00 UTC) is the close of the day's 23:00 bar, known at E. A day's return / volume counts only if
all 24 hourly closes of that day are present (gaps are never filled).
"""
import numpy as np
import pandas as pd

from backtest.risk.config import RiskConfig
from backtest.strategy.base import Strategy

RISK_G1 = RiskConfig(max_position=1.0, max_gross=1.0, max_net=1.0)   # G = 1.0 x equity (user, 2026-09-30)


def daily_panel(prices, volume=None, end=None):
    """Daily log returns (and quote volume) for complete UTC days ending at or before `end` (a 00:00 stamp).
    Day d is labelled by its END stamp (00:00 of d+1)."""
    p = prices.loc[:end] if end is not None else prices
    # hourly bar close at h belongs to the day ending at ceil(h, D); the 00:00 close closes the previous day
    key = p.index.ceil("D")
    cnt = p.notna().groupby(key).sum()
    at_end = p.reindex(cnt.index)                      # the 00:00 stamp = close of the day's 23:00 bar
    full = cnt >= 24
    r = np.log(at_end / at_end.shift(1)).where(full & full.shift(1, fill_value=False))
    r = r.where(r.index.to_series().diff().eq(pd.Timedelta("1D")), axis=0)
    out = {"r": r}
    if volume is not None:
        v = volume.loc[:end] if end is not None else volume
        vd = v.groupby(v.index.ceil("D")).sum(min_count=24)
        out["v"] = vd.reindex(cnt.index).where(full)
    return out


def h7_weights_table(r, q=0.10, lookback=365, minp=180, G=1.0):
    thr = r.rolling(lookback, min_periods=minp).quantile(q).shift(1)
    sig = (r < thr).astype(float)
    n = sig.sum(axis=1)
    return sig.div(n.where(n > 0), axis=0).fillna(0.0) * G


def h2_weights_table(r, v, q=0.67, vol_win=30, minp_vol=20, minp_q=90, G=1.0):
    med = v.rolling(vol_win, min_periods=minp_vol).median().shift(1)
    shock = np.log(v / med).mean(axis=1, skipna=True)
    thr = shock.expanding(minp_q).quantile(q).shift(1)
    high = shock > thr
    live = r.notna().sum(axis=1)
    w = (-np.sign(r)).div(live.where(live > 0), axis=0).fillna(0.0) * G
    return w.where(high, 0.0)


def hour_k_schedule(k):
    def f(dates):
        return dates[dates.hour == k]
    return f


class _Base(Strategy):
    risk = RISK_G1

    def __init__(self, k=0, G=1.0):
        self.k, self.G = k, G
        self.rebalance_frequency = hour_k_schedule(k)

    def _end(self, t):
        return t.floor("D")                         # last completed day boundary at or before t


class CrashRebound(_Base):
    """H7: long coins whose last complete UTC day returned below their trailing-365d 10th percentile; 24h hold."""

    def generate_weights(self, data, t):
        assert data.prices.index.max() <= t, "look-ahead: view extends past t"
        d = daily_panel(data.prices, end=self._end(t))
        w = h7_weights_table(d["r"], G=self.G)
        return w.iloc[-1] if len(w) and w.index[-1] == self._end(t) else pd.Series(0.0, index=data.prices.columns)


class VolShockReversal(_Base):
    """H2: on panel volume-shock days, short each coin's day move (-sign r) equal-weight; 24h hold."""

    def required_data(self):
        return {"prices": None, "volume": None}

    def generate_weights(self, data, t):
        assert data.prices.index.max() <= t, "look-ahead: view extends past t"
        d = daily_panel(data.prices, data.volume, end=self._end(t))
        w = h2_weights_table(d["r"], d["v"], G=self.G)
        return w.iloc[-1] if len(w) and w.index[-1] == self._end(t) else pd.Series(0.0, index=data.prices.columns)

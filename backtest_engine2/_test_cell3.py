import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, r'C:\Users\User\backtest_engine\backtest_engine2')

import numpy as np
import pandas as pd
import yfinance as yf
from backtest.data import DataPanel
from backtest.strategy import Strategy
from _mood_helpers import _mood_d_max_tau

MOOD_THRESHOLD = 4.256531

# ── minimal panel (same as Cell 1) ───────────────────────────────────────────
raw = yf.download(['^GSPC', '^VIX'], start='1990-01-01', auto_adjust=False, progress=False)['Close']
raw.columns = ['SPX', 'VIX']
raw = raw.dropna()
vol_raw = yf.download('^GSPC', start='1990-01-01', auto_adjust=False, progress=False)['Volume'].squeeze()
vol_raw.name = 'SPX'
idx = raw.index.intersection(vol_raw.index)
raw = raw.loc[idx]; vol_raw = vol_raw.loc[idx]
prices_df = pd.DataFrame({'SPX': raw['SPX'], 'CASH': 1.0}, index=raw.index)
volume_df = pd.DataFrame({'SPX': vol_raw, 'CASH': float('nan')}, index=raw.index)
vix_df    = pd.DataFrame({'SPX': raw['VIX'], 'CASH': raw['VIX']}, index=raw.index)
panel = DataPanel(prices_df, volume=volume_df, features={'VIX': vix_df}, check_outliers=True)

# ── strategy ─────────────────────────────────────────────────────────────────
class VIXChangePoint(Strategy):
    rebalance_frequency = "daily"

    def __init__(self, ewma_lambda=0.95, vol_threshold=0.20, warmup=21, threshold=MOOD_THRESHOLD):
        self.ewma_lambda   = ewma_lambda
        self.vol_threshold = vol_threshold
        self.warmup        = warmup
        self.threshold     = threshold
        self._reset_state()

    def __repr__(self):
        return (f"VIXChangePoint(ewma_lambda={self.ewma_lambda}, "
                f"vol_threshold={self.vol_threshold}, "
                f"warmup={self.warmup}, "
                f"threshold={self.threshold:.6f})")

    def required_data(self):
        return {"prices": None, "VIX": None}

    def _reset_state(self):
        self._vix_buf  = []
        self._spx_buf  = []
        self._ewma_var = None
        self._prev_w   = None

    def _ewma_from_scratch(self, rets):
        lam = self.ewma_lambda
        v = rets[0] ** 2
        for r in rets[1:]:
            v = lam * v + (1 - lam) * r ** 2
        return v

    def generate_weights(self, data, t):
        vix_s   = data.feature("VIX")["SPX"]
        price_s = data.prices["SPX"]
        if len(vix_s) < 2 or len(price_s) < 2:
            return pd.Series({"SPX": 0.0, "CASH": 1.0})
        vix_ret = float(np.log(vix_s.iloc[-1]   / vix_s.iloc[-2]))
        spx_ret = float(np.log(price_s.iloc[-1] / price_s.iloc[-2]))
        self._vix_buf.append(vix_ret)
        self._spx_buf.append(spx_ret)
        lam = self.ewma_lambda
        if self._ewma_var is None:
            self._ewma_var = spx_ret ** 2
        else:
            self._ewma_var = lam * self._ewma_var + (1 - lam) * spx_ret ** 2
        n = len(self._vix_buf)
        if n > self.warmup:
            buf_arr = np.array(self._vix_buf, dtype=np.float64)
            dmax, tau_f = _mood_d_max_tau(buf_arr, n)
            if dmax > self.threshold:
                tau = int(tau_f)
                post = self._spx_buf[tau:]
                if post:
                    self._ewma_var = self._ewma_from_scratch(post)
                self._vix_buf = []
                self._spx_buf = []
        ann_vol = np.sqrt(max(self._ewma_var, 0.0) * 252)
        if ann_vol < self.vol_threshold:
            proposed = pd.Series({"SPX": 1.0, "CASH": 0.0})
        else:
            proposed = pd.Series({"SPX": 0.0, "CASH": 1.0})
        weight       = self._prev_w if self._prev_w is not None else pd.Series({"SPX": 0.0, "CASH": 1.0})
        self._prev_w = proposed
        return weight


strat = VIXChangePoint()
print(strat)
print(f"required_data : {strat.required_data()}")
print(f"Rebalances/yr : {len(strat.rebalance_dates(panel.dates)) / (len(panel.dates) / 252):.1f}")

# quick smoke test: run generate_weights on the last few bars
view = panel.as_of(panel.dates[100])
w = strat.generate_weights(view, panel.dates[100])
print(f"Weight at bar 100: {w.to_dict()}")
view2 = panel.as_of(panel.dates[101])
w2 = strat.generate_weights(view2, panel.dates[101])
print(f"Weight at bar 101: {w2.to_dict()}")
print("OK")

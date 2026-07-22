import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, r'C:\Users\User\backtest_engine\backtest_engine2')

import numpy as np
import pandas as pd
import yfinance as yf
from backtest.data import DataPanel
from backtest.strategy import Strategy
from backtest.risk.config import RiskConfig
from backtest.costs import Commission, Spread, MarketImpact, CompositeCostModel, LiquidityCap
from backtest.engine import MultiPathEngine
from backtest.splitters import CombinatorialPurgedCV
from _mood_helpers import _mood_d_max_tau

MOOD_THRESHOLD = 4.256531

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


class VIXChangePoint(Strategy):
    rebalance_frequency = "daily"
    risk = RiskConfig(max_position=None, max_gross=1.0, max_net=1.0)

    def __init__(self, ewma_lambda=0.95, vol_threshold=0.20, warmup=21, threshold=MOOD_THRESHOLD):
        self.ewma_lambda   = ewma_lambda
        self.vol_threshold = vol_threshold
        self.warmup        = warmup
        self.threshold     = threshold
        self._reset_state()

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


costs = CompositeCostModel([
    Commission(bps=1),
    Spread(half_bps={'SPX': 1, 'CASH': 0}),
    MarketImpact(k=10, kind='sqrt', adv_lookback=20),
])
liq = LiquidityCap(cap_pct=0.10, adv_lookback=20)

# --- Cell 6a: splitter ---
splitter = CombinatorialPurgedCV(
    n_splits=6,
    n_test_groups=2,
    purge_bars=5,
    embargo_pct=0.01,
)

n = len(panel.dates)
embargo_bars = int(n * splitter.embargo_pct)
print(f"Timeline:           {n} bars  ({panel.dates[0].date()} -> {panel.dates[-1].date()})")
print(f"Groups:             {splitter.n_groups} contiguous groups of ~{n // splitter.n_groups} bars each")
print(f"Splits (trainings): {splitter.n_splits()}")
print(f"Paths (curves):     {splitter.n_paths()}")
print(f"Purge:              {splitter.purge_bars} bars on each side of every test run")
print(f"Embargo:            {embargo_bars} bars after each test run  (trailing buffer = {splitter.purge_bars + embargo_bars})")

first_train, first_test = next(iter(splitter.split(panel.dates)))
print(f"\nFirst split:  train={len(first_train)}, test={len(first_test)}, disjoint={len(first_train.intersection(first_test)) == 0}")

# --- Cell 6b: multipath run ---
print("\n--- Running MultiPathEngine ---")
mp = MultiPathEngine(
    panel,
    lambda: VIXChangePoint(),
    splitter,
    costs=costs,
    liquidity=liq,
    initial_capital=1_000_000,
)
res = mp.run(n_jobs=-1)

print(f"Splits run:           {res.n_splits}")
print(f"Paths assembled:      {res.n_paths}")
print(f"path_returns shape:   {res.path_returns.shape}")
print(f"path_equity shape:    {res.path_equity.shape}")
print(f"Date range:           {res.path_returns.index[0].date()} -> {res.path_returns.index[-1].date()}")
print(f"Per-path NaN count:   min={res.path_returns.isna().sum().min()}, max={res.path_returns.isna().sum().max()}")
print()
print("Final equity per path:")
print(res.path_equity.iloc[-1].rename("final_equity").map(lambda v: f"${v:,.0f}"))
print("OK")

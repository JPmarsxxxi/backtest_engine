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
from backtest.engine import Engine, MultiPathEngine
from backtest.splitters import CombinatorialPurgedCV
from backtest.metrics import compute_metrics
from backtest.selection import dsr, effective_k, expected_max_sr
from backtest.metrics.core import sharpe_var_term
from _mood_helpers import _mood_d_max_tau

MOOD_THRESHOLD = 4.256531

# ── setup (shared with cells 1-6) ────────────────────────────────────────────
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

# ── cell 5 result ─────────────────────────────────────────────────────────────
engine = Engine(panel, VIXChangePoint(), costs=costs, liquidity=liq, initial_capital=1_000_000)
result = engine.run()

# ── cell 6 result ─────────────────────────────────────────────────────────────
splitter = CombinatorialPurgedCV(n_splits=6, n_test_groups=2, purge_bars=5, embargo_pct=0.01)
mp = MultiPathEngine(panel, lambda: VIXChangePoint(), splitter, costs=costs, liquidity=liq, initial_capital=1_000_000)
res = mp.run(n_jobs=-1)

# ── Cell 7: metrics ───────────────────────────────────────────────────────────
print("=" * 60)
print("CELL 7 — METRICS")
print("=" * 60)

rep = compute_metrics(
    result.returns,
    result.equity_curve,
    costs=result.costs,
    trades=result.trades,
    ann_factor=252,
    ci_level=0.95,
)
print(rep)
print()
print(f"Sharpe:    {rep.sharpe:.3f}  ± {rep.sharpe_std:.3f}  "
      f"(95% CI: [{rep.sharpe_ci_low:.3f}, {rep.sharpe_ci_high:.3f}])")
print(f"PSR(0):    {rep.psr:.3f}  {'significant' if rep.psr > 0.95 else 'NOT significant'} at 95%")
print(f"MinTRL(0): {rep.min_trl:.0f} bars  (sample: {rep.n_obs} bars  "
      f"-> {'powered' if rep.min_trl < rep.n_obs else 'underpowered'})")

try:
    print("\nYearly breakdown:")
    print(rep.yearly_table())
except AttributeError:
    if rep.yearly is not None:
        print("\nYearly breakdown:")
        print(rep.yearly)

print("\n--- CPCV aggregate ---")
agg = res.aggregate_metrics()
print(f"Sharpe  mean={agg['sharpe_mean']:.3f}  std={agg['sharpe_std']:.3f}  "
      f"min={agg['sharpe_min']:.3f}  max={agg['sharpe_max']:.3f}")
print(f"Calmar  mean={agg['calmar_mean']:.3f}  std={agg['calmar_std']:.3f}")
print(f"Max DD  mean={agg['max_drawdown_mean']:.3f}  std={agg['max_drawdown_std']:.3f}")

# ── Cell 8: DSR ───────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("CELL 8 — DSR")
print("=" * 60)

K = 1  # first and only trial of VIXChangePoint this session

deflated = dsr(result.returns, K=K)

r = result.returns.dropna().to_numpy()
T = len(r)
sr_pb = r.mean() / r.std(ddof=1)
var_term = sharpe_var_term(r, sr_pb)
luck_max_pb  = expected_max_sr(K, var_term / T)
luck_max_ann = luck_max_pb * np.sqrt(252)

print(f"Trials (K):             {K}")
print(f"Realized Sharpe (ann.): {sr_pb * np.sqrt(252):.3f}")
print(f"Luck-max threshold:     {luck_max_ann:.3f}")
print(f"DSR:                    {deflated:.3f}   "
      f"{'significant' if deflated > 0.95 else 'NOT significant'} at 95%")

deflated_paths = [dsr(res.path_returns[c].dropna(), K=K) for c in res.path_returns.columns]
print(f"\nCPCV per-path DSR (K={K}):  "
      f"median={np.median(deflated_paths):.3f}  "
      f"min={np.min(deflated_paths):.3f}  max={np.max(deflated_paths):.3f}")

K_eff_paths = effective_k(res.path_returns.dropna(how='all'), threshold=0.5)
print(f"Effective-K from CPCV paths: {K_eff_paths}  "
      f"(paths are not independent trials; K stays {K})")

print("OK")

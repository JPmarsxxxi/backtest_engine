"""Run Cell 7 logic standalone: build PIT cache for HurstPairsStrategy."""
import sys, warnings, time
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from statsmodels.tsa.stattools import adfuller

REPO_ROOT = Path(r"C:\Users\User\backtest_engine")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from backtest.alpha_pipeline import frequency

DATA_DIR = REPO_ROOT / "backtest_engine2" / "data"
CACHE_DIR_C7 = DATA_DIR / "_cache"
CACHE_DIR_C7.mkdir(exist_ok=True, parents=True)
COINT_SERIES_CACHE = CACHE_DIR_C7 / "hurst_pairs_coint_series.parquet"
HURST_SERIES_CACHE = CACHE_DIR_C7 / "hurst_pairs_hurst_series.parquet"

# --- Cells 1+2+3 setup ---
raw = pd.read_parquet(DATA_DIR / "binance_hourly_top50_pool.parquet")
EXCLUDE = {"USD1USDT","USDSUSDT","RLUSDUSDT","BFUSDUSDT","USDEUSDT","XAUTUSDT","PAXGUSDT"}
filtered = raw[~raw["symbol"].isin(EXCLUDE)].copy()
prices = filtered.pivot(index="open_time", columns="symbol", values="close")
prices.index = pd.DatetimeIndex(prices.index).tz_convert(None)
prices = prices.sort_index()
prices_pit = frequency.run(prices, freq="intraday", delay=1)
prices_pit_clean = prices_pit

WINDOW_BARS = 8760; STEP_BARS = 720; MIN_OVERLAP = 12960; ADF_MAXLAG = 12
HEDGE_W = 720; HURST_W = 168
TAU_ARR_C7 = np.array([1,2,4,8,16,32]); Q_C7 = 1


if COINT_SERIES_CACHE.exists() and HURST_SERIES_CACHE.exists():
    print("Loading cached PIT series from disk...")
    coint_series_df = pd.read_parquet(COINT_SERIES_CACHE)
    hurst_series_df = pd.read_parquet(HURST_SERIES_CACHE)
else:
    print("Building PIT cache (~21 min on first run)...")
    log_pp = np.log(prices_pit_clean)
    assets_c7 = log_pp.columns.tolist()
    qualifying_pairs_c7 = []
    for i, a in enumerate(assets_c7):
        va = log_pp[a].notna()
        for b in assets_c7[i+1:]:
            ov = int((va & log_pp[b].notna()).sum())
            if ov >= MIN_OVERLAP:
                qualifying_pairs_c7.append((a, b, ov))
    print(f"  {len(qualifying_pairs_c7)} qualifying pairs (overlap >= {MIN_OVERLAP})")

    def _rolling_coint_series(av, bv, w, s, ml):
        out = []
        for end in range(w, len(av)+1, s):
            a_w = av[end-w:end]; b_w = bv[end-w:end]
            X = np.column_stack([np.ones_like(a_w), a_w])
            c, *_ = np.linalg.lstsq(X, b_w, rcond=None)
            r = b_w - c[0] - c[1]*a_w
            try: p = adfuller(r, maxlag=ml, regression="c")[1]
            except: p = np.nan
            out.append((end-1, p, float(c[1])))
        return out

    def _proc_coint(a, b, lpa, lpb):
        both = pd.concat([lpa, lpb], axis=1).dropna()
        if len(both) < WINDOW_BARS: return []
        ts = both.index
        res = _rolling_coint_series(both.iloc[:,0].values, both.iloc[:,1].values, WINDOW_BARS, STEP_BARS, ADF_MAXLAG)
        return [{"pair_id": f"{a}|{b}", "asset_a": a, "asset_b": b, "anchor_t": ts[idx], "p_value": p, "hedge_ratio": br} for idx, p, br in res]

    t0 = time.time()
    coint_chunks = Parallel(n_jobs=-1, verbose=0)(delayed(_proc_coint)(a, b, log_pp[a], log_pp[b]) for a, b, _ in qualifying_pairs_c7)
    coint_flat = [item for chunk in coint_chunks for item in chunk]
    coint_series_df = pd.DataFrame(coint_flat)
    print(f"  cointegration: {time.time()-t0:.1f}s ({len(coint_series_df):,} rows)")
    coint_series_df.to_parquet(COINT_SERIES_CACHE)

    def _hurst_win(x, tau, q=1):
        if np.isnan(x).any(): return np.nan
        abs_xq = np.abs(x)**q; denom = abs_xq.mean()
        if denom <= 0 or not np.isfinite(denom): return np.nan
        K = np.empty(len(tau))
        for i,t in enumerate(tau): K[i] = (np.abs(x[t:] - x[:-t])**q).mean() / denom
        v = (K > 0) & np.isfinite(K)
        if v.sum() < 2: return np.nan
        s,_ = np.polyfit(np.log(tau[v]), np.log(K[v]), 1)
        return s/q

    def _rolling_hurst(s, w, tau, q=1):
        x = s.values.astype(float)
        n = len(x); out = np.full(n, np.nan)
        for end in range(w, n+1): out[end-1] = _hurst_win(x[end-w:end], tau, q)
        return pd.Series(out, index=s.index)

    def _compute_spread(lpa, lpb, hw):
        lr_a = lpa.diff(); lr_b = lpb.diff()
        return lpa - (lr_a.rolling(hw, min_periods=hw).std() / lr_b.rolling(hw, min_periods=hw).std()) * lpb

    def _proc_hurst(a, b, lpa, lpb):
        both = pd.concat([lpa, lpb], axis=1).dropna()
        if len(both) < HEDGE_W + HURST_W: return None
        spread = _compute_spread(both.iloc[:,0], both.iloc[:,1], HEDGE_W).dropna()
        if len(spread) < HURST_W: return None
        h = _rolling_hurst(spread, HURST_W, TAU_ARR_C7, Q_C7)
        df = pd.DataFrame({"pair_id": f"{a}|{b}", "asset_a": a, "asset_b": b, "bar_t": spread.index, "hurst": h.values, "spread": spread.values})
        df = df.dropna(subset=["hurst"])
        return df

    t0 = time.time()
    hurst_chunks = Parallel(n_jobs=-1, verbose=0)(delayed(_proc_hurst)(a, b, log_pp[a], log_pp[b]) for a, b, _ in qualifying_pairs_c7)
    hurst_chunks = [c for c in hurst_chunks if c is not None]
    hurst_series_df = pd.concat(hurst_chunks, ignore_index=True)
    print(f"  hurst+spread: {time.time()-t0:.1f}s ({len(hurst_series_df):,} rows)")
    hurst_series_df.to_parquet(HURST_SERIES_CACHE)

print(f"\nCache state:")
print(f"  cointegration series: {len(coint_series_df):,} rows, {coint_series_df['pair_id'].nunique()} pairs")
print(f"  hurst+spread series:  {len(hurst_series_df):,} rows, {hurst_series_df['pair_id'].nunique()} pairs")
print(f"  median coint anchors per pair: {int(coint_series_df.groupby('pair_id').size().median())}")
print(f"  median hurst bars per pair:    {int(hurst_series_df.groupby('pair_id').size().median())}")
print(f"  coint p-value range: [{coint_series_df['p_value'].min():.4f}, {coint_series_df['p_value'].max():.4f}]")
print(f"  hurst range:         [{hurst_series_df['hurst'].min():.3f}, {hurst_series_df['hurst'].max():.3f}]")
print(f"\nCache files:")
print(f"  {COINT_SERIES_CACHE}")
print(f"  {HURST_SERIES_CACHE}")

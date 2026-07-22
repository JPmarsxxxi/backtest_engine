"""C7 — Cross-pair PnL correlation diagnostic.

Builds daily PnL per pair on the MEDIUM-gate + OU-exits strategy candidate,
then asks: are the 30 pairs really 30 independent trades, or are they all
proxies for 1-5 underlying factors?
"""
import sys, warnings, time
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

REPO_ROOT = Path(r"C:\Users\User\backtest_engine")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from backtest.alpha_pipeline import frequency

DATA_DIR = REPO_ROOT / "backtest_engine2" / "data"
JOINT_CACHE = REPO_ROOT / "backtest_engine2" / "data" / "_cache" / "hunt_2026-06-02_joint_scores.parquet"

raw = pd.read_parquet(DATA_DIR / "binance_hourly_top50_pool.parquet")
EXCLUDE = {"USD1USDT","USDSUSDT","RLUSDUSDT","BFUSDUSDT","USDEUSDT","XAUTUSDT","PAXGUSDT"}
filtered = raw[~raw["symbol"].isin(EXCLUDE)].copy()
prices = filtered.pivot(index="open_time", columns="symbol", values="close")
prices.index = pd.DatetimeIndex(prices.index).tz_convert(None)
prices = prices.sort_index()
prices_pit = frequency.run(prices, freq="intraday", delay=1)
log_prices_pit = np.log(prices_pit)

all_summary = pd.read_parquet(JOINT_CACHE)
def cap_select(df, k, n_target):
    sd = df.sort_values("joint_score", ascending=False).reset_index(drop=True)
    counts, picked = {}, []
    for _, row in sd.iterrows():
        if len(picked) >= n_target: break
        a, b = row["asset_a"], row["asset_b"]
        if counts.get(a,0) >= k or counts.get(b,0) >= k: continue
        picked.append(row); counts[a]=counts.get(a,0)+1; counts[b]=counts.get(b,0)+1
    return pd.DataFrame(picked).reset_index(drop=True), counts
new_top30, _ = cap_select(all_summary, k=3, n_target=30)
top_pairs = list(zip(new_top30["asset_a"], new_top30["asset_b"]))

HEDGE_W=720; HURST_W=168; Z_W=168
TAU=np.array([1,2,4,8,16,32]); Q=1
HOLD_MAX = 72; N_PATHS = 1000
CAL_FIT_WINDOW = HURST_W; CAL_FREQ = 24
PI_PLUS_GRID  = np.arange(0.5, 10.5, 0.5)
PI_MINUS_GRID = -np.arange(0.5, 10.5, 0.5)

def hurst_win(x, tau, q=1):
    if np.isnan(x).any(): return np.nan
    abs_xq = np.abs(x)**q; denom = abs_xq.mean()
    if denom <= 0 or not np.isfinite(denom): return np.nan
    K = np.empty(len(tau))
    for i,t in enumerate(tau): K[i] = (np.abs(x[t:] - x[:-t])**q).mean() / denom
    v = (K > 0) & np.isfinite(K)
    if v.sum() < 2: return np.nan
    s,_ = np.polyfit(np.log(tau[v]), np.log(K[v]), 1)
    return s/q

def rolling_hurst(s, w, tau, q=1):
    x = s.values.astype(float)
    n = len(x); out = np.full(n, np.nan)
    for end in range(w, n+1): out[end-1] = hurst_win(x[end-w:end], tau, q)
    return pd.Series(out, index=s.index)

def compute_spread(lpa, lpb, hw):
    lr_a = lpa.diff(); lr_b = lpb.diff()
    return lpa - (lr_a.rolling(hw, min_periods=hw).std() / lr_b.rolling(hw, min_periods=hw).std()) * lpb

def process_pair(pair):
    a, b = pair
    both = pd.concat([log_prices_pit[a], log_prices_pit[b]], axis=1).dropna()
    sp = compute_spread(both.iloc[:,0], both.iloc[:,1], HEDGE_W).dropna()
    h = rolling_hurst(sp, HURST_W, TAU, Q)
    return pair, sp, h

print("Recomputing series for top-30...")
t0 = time.time()
sr = Parallel(n_jobs=-1)(delayed(process_pair)(p) for p in top_pairs)
spread_by_pair = {p: sp for p, sp, h in sr}
hurst_by_pair = {p: h for p, sp, h in sr}
print("  done in {:.1f}s".format(time.time()-t0))


def calibrate_ou(spread_window):
    Y = np.diff(spread_window)
    s_prev = spread_window[:-1]
    m = float(np.mean(spread_window))
    X = s_prev - m
    var_X = float(np.var(X))
    if var_X <= 1e-12: return None
    slope = float(np.mean(Y * X)) / var_X
    phi = 1.0 + slope
    if not (-1.0 < phi < 1.0): return None
    resid = Y - slope * X
    sigma_sq = float(np.var(resid))
    if sigma_sq <= 0: return None
    return phi, float(np.sqrt(sigma_sq)), m


def find_best_thresholds(phi, sigma, m, s_0, sign_dir, n_paths=N_PATHS, n_steps=HOLD_MAX, seed=42):
    rng = np.random.default_rng(seed)
    paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    paths[:, 0] = s_0
    eps = rng.standard_normal((n_paths, n_steps))
    one_minus_phi_m = (1.0 - phi) * m
    for k in range(1, n_steps + 1):
        paths[:, k] = phi * paths[:, k-1] + one_minus_phi_m + sigma * eps[:, k-1]
    pnl_sigma = (sign_dir * (paths - s_0)) / sigma
    pnl_after_entry = pnl_sigma[:, 1:]
    cummax = np.maximum.accumulate(pnl_after_entry, axis=1)
    cummin = np.minimum.accumulate(pnl_after_entry, axis=1)
    cross_pp = np.full((n_paths, len(PI_PLUS_GRID)), n_steps - 1, dtype=np.int32)
    for j, pp in enumerate(PI_PLUS_GRID):
        mask = cummax >= pp
        any_hit = mask.any(axis=1)
        cross_pp[any_hit, j] = mask[any_hit].argmax(axis=1)
    cross_pm = np.full((n_paths, len(PI_MINUS_GRID)), n_steps - 1, dtype=np.int32)
    for j, pm in enumerate(PI_MINUS_GRID):
        mask = cummin <= pm
        any_hit = mask.any(axis=1)
        cross_pm[any_hit, j] = mask[any_hit].argmax(axis=1)
    best_sharpe = -np.inf; best_pp = best_pm = None
    for i, pp in enumerate(PI_PLUS_GRID):
        for j, pm in enumerate(PI_MINUS_GRID):
            exit_step = np.minimum(cross_pp[:, i], cross_pm[:, j])
            pnl_at_exit = pnl_after_entry[np.arange(n_paths), exit_step]
            mu = float(pnl_at_exit.mean()); sd = float(pnl_at_exit.std())
            if sd <= 1e-9: continue
            sh = mu / sd
            if sh > best_sharpe:
                best_sharpe = sh; best_pp = float(pp); best_pm = float(pm)
    return best_pp, best_pm, best_sharpe


def calibrate_pair(pair, spread, hurst, freq=CAL_FREQ):
    sp_arr = spread.values.astype(float); n = len(sp_arr)
    out = []
    for t in range(CAL_FIT_WINDOW, n, freq):
        window = sp_arr[t - CAL_FIT_WINDOW:t]
        if np.isnan(window).any(): continue
        s_0 = sp_arr[t]
        if np.isnan(s_0): continue
        res = calibrate_ou(window)
        if res is None: continue
        phi, sigma, m = res
        sign_dir = -np.sign(s_0 - m) if s_0 != m else 1.0
        if sign_dir == 0: sign_dir = 1.0
        pp, pm, sh = find_best_thresholds(phi, sigma, m, s_0, sign_dir)
        if pp is None: continue
        out.append({"t_idx": t, "t": spread.index[t], "phi": phi, "sigma": sigma, "m": m,
                    "pi_plus_star": pp, "pi_minus_star": pm, "sharpe": sh})
    return pd.DataFrame(out).set_index("t_idx") if out else pd.DataFrame()


def simulate_pair_ou(pair, spread, hurst, cal_table, hold_max=HOLD_MAX):
    sp_arr = spread.values.astype(float); n = len(sp_arr)
    if cal_table.empty: return pd.DataFrame()
    cal_sorted = cal_table.sort_index()
    cal_idx_arr = cal_sorted.index.values
    pp_arr = cal_sorted["pi_plus_star"].values
    pm_arr = cal_sorted["pi_minus_star"].values
    sigma_arr = cal_sorted["sigma"].values
    m_roll = spread.rolling(Z_W, min_periods=Z_W).mean().shift(1).values
    sd_roll = spread.rolling(Z_W, min_periods=Z_W).std().shift(1).values
    z_arr = (sp_arr - m_roll) / sd_roll
    abs_z_arr = np.abs(z_arr); sign_arr = np.sign(z_arr)
    h_arr = hurst.reindex(spread.index).values.astype(float)
    in_band = (abs_z_arr > 1).astype(int)
    tib = np.zeros(n, dtype=int); cur = 0
    for i in range(n):
        cur = cur + 1 if in_band[i] else 0
        tib[i] = cur if in_band[i] else 0
    entry_mask = (abs_z_arr > 1) & (abs_z_arr < 2) & (h_arr < 0.5) & ~np.isnan(m_roll) & ~np.isnan(sd_roll)
    entry_idx = np.where(entry_mask)[0]
    rows = []
    for t in entry_idx:
        cal_pos = np.searchsorted(cal_idx_arr, t, side="right") - 1
        if cal_pos < 0: continue
        pp, pm, sigma_cal = pp_arr[cal_pos], pm_arr[cal_pos], sigma_arr[cal_pos]
        if sigma_cal <= 0: continue
        sign_t = sign_arr[t]
        if sign_t == 0: continue
        s_0 = sp_arr[t]
        exit_k = None; exit_reason = None
        for k in range(1, hold_max + 1):
            idx = t + k
            if idx >= n: exit_k = k - 1; exit_reason = "data_end"; break
            cur_pnl_sigma = -sign_t * (sp_arr[idx] - s_0) / sigma_cal
            if cur_pnl_sigma >= pp: exit_k = k; exit_reason = "take_profit"; break
            if cur_pnl_sigma <= pm: exit_k = k; exit_reason = "stop_loss"; break
        if exit_k is None: exit_k = hold_max; exit_reason = "barrier"
        exit_spread = sp_arr[t + exit_k]
        pnl = -sign_t * (exit_spread - s_0)
        rows.append({"pair": str(pair), "t": spread.index[t],
                     "abs_z": abs_z_arr[t], "hurst": h_arr[t], "tib": tib[t],
                     "exit_k": exit_k, "exit_reason": exit_reason, "pnl": pnl})
    return pd.DataFrame(rows)


print("\nCalibrating OU thresholds...")
t0 = time.time()
cal_tables = Parallel(n_jobs=-1, verbose=0)(delayed(calibrate_pair)(p, spread_by_pair[p], hurst_by_pair[p]) for p in top_pairs)
print("  done in {:.1f}s".format(time.time()-t0))

print("Simulating OU-exit trades...")
t0 = time.time()
trade_dfs = Parallel(n_jobs=-1, verbose=0)(
    delayed(simulate_pair_ou)(p, spread_by_pair[p], hurst_by_pair[p], cal_tables[i])
    for i, p in enumerate(top_pairs)
)
trades = pd.concat([t for t in trade_dfs if not t.empty])
print("  {:,} trades in {:.1f}s".format(len(trades), time.time()-t0))

# Apply MEDIUM gate
MEDIUM = {"hurst_thr": 0.42, "tib_thr": 6, "z_lo": 1.0, "z_hi": 1.33}
medium = trades[
    (trades["hurst"] <= MEDIUM["hurst_thr"]) &
    (trades["tib"] <= MEDIUM["tib_thr"]) &
    (trades["abs_z"].between(MEDIUM["z_lo"], MEDIUM["z_hi"]))
].copy()
medium["date"] = pd.to_datetime(medium["t"]).dt.date
print("\nMEDIUM-gate trades: {:,}".format(len(medium)))

# Daily PnL per pair: pivot to date x pair, fill 0 (no-trade days)
daily_pnl = medium.groupby(["date", "pair"])["pnl"].sum().unstack(fill_value=0)
print("Daily PnL matrix shape: {} (dates x pairs)".format(daily_pnl.shape))

# Pair-pair PnL correlation matrix
corr = daily_pnl.corr()
n_pairs = len(corr)

print("\n" + "="*100)
print("C7 -- Cross-pair PnL correlation diagnostic")
print("="*100)

# Distribution of upper-triangle correlations
upper_idx = np.triu_indices_from(corr.values, k=1)
upper = corr.values[upper_idx]
upper = upper[~np.isnan(upper)]
print("\nPair-pair correlation distribution ({} unique pair-pair correlations):".format(len(upper)))
print("  min:    {:+.3f}".format(np.min(upper)))
print("  p5:     {:+.3f}".format(np.percentile(upper, 5)))
print("  p25:    {:+.3f}".format(np.percentile(upper, 25)))
print("  median: {:+.3f}".format(np.median(upper)))
print("  p75:    {:+.3f}".format(np.percentile(upper, 75)))
print("  p95:    {:+.3f}".format(np.percentile(upper, 95)))
print("  max:    {:+.3f}".format(np.max(upper)))
print("  mean:   {:+.3f}".format(np.mean(upper)))
print("\nFraction of pair-pair correlations above thresholds:")
for thr in (0.1, 0.2, 0.3, 0.5, 0.7):
    print("  >  {:.1f}: {:.3f}  ({} of {} pairs)".format(thr, (upper > thr).mean(), int((upper > thr).sum()), len(upper)))

# Top-15 highest correlations
print("\nTop-15 most-correlated pair-pair combinations:")
pairs_list = list(corr.columns)
records = []
for i in range(n_pairs):
    for j in range(i+1, n_pairs):
        c = corr.iloc[i, j]
        if np.isnan(c): continue
        records.append((pairs_list[i], pairs_list[j], c))
records.sort(key=lambda x: x[2], reverse=True)
for p1, p2, c in records[:15]:
    print("  {:+.3f}  {} <-> {}".format(c, p1, p2))

# PCA on standardized daily PnL
print("\n--- PCA on standardized daily PnL ---")
ret = daily_pnl.copy()
# Standardize each column
std_col = ret.std(); std_col[std_col == 0] = 1.0
ret_norm = (ret - ret.mean()) / std_col
ret_norm = ret_norm.fillna(0)
cov_mat = ret_norm.cov().values
eigvals = np.linalg.eigvalsh(cov_mat)
eigvals = np.sort(eigvals[eigvals > 0])[::-1]
total = eigvals.sum()
print("Top-5 eigenvalues (% of total variance):")
for i in range(min(5, len(eigvals))):
    print("  PC{}: {:.1f}%".format(i+1, 100 * eigvals[i] / total))

cumvar = np.cumsum(eigvals) / total * 100
n_for_50 = int(np.searchsorted(cumvar, 50) + 1)
n_for_80 = int(np.searchsorted(cumvar, 80) + 1)
n_for_90 = int(np.searchsorted(cumvar, 90) + 1)
print("Components needed for: 50% var = {}, 80% var = {}, 90% var = {}".format(n_for_50, n_for_80, n_for_90))
print("(If 30 pairs were perfectly independent, all three would equal 15, 24, 27.)")

# Decision rule
median_corr = np.median(upper)
if median_corr < 0.2:
    verdict = "DIVERSIFIED -- pairs are mostly independent"
elif median_corr < 0.4:
    verdict = "MODERATELY DIVERSIFIED"
else:
    verdict = "CONCENTRATED -- pairs move together"
print("\n=== C7 decision: {} (median pair-pair corr = {:+.3f}) ===".format(verdict, median_corr))

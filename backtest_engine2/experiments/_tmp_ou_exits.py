"""Test spec's O-U-calibrated per-trade exits.

At each calibration anchor (daily for tractability):
  1. Fit OU params {phi, sigma, m} on prior 168 bars of spread.
  2. Simulate N_PATHS OU paths from current s_0, length 72.
  3. Grid-search 20x20 {pi+, pi-} thresholds (in sigma units), pick best Sharpe.
  4. Cache (phi, sigma, m, pi_plus*, pi_minus*) for that anchor.

For each entry bar, use the most recent calibration; simulate the actual trade with
those thresholds; exit on first of TP / SL / 72h barrier.
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
HOLD_MAX = 72             # spec vertical barrier
N_PATHS = 1000            # synthetic paths per calibration (spec says 100k; 1k is fast and Sharpe-ranking-robust)
CAL_FIT_WINDOW = HURST_W  # 168 bars for OU fit; matches the spec's TW
CAL_FREQ = 24             # recalibrate every 24 bars (daily) - approximation
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
    """OLS fit: phi, sigma. Returns None if non-stationary."""
    Y = np.diff(spread_window)
    s_prev = spread_window[:-1]
    m = float(np.mean(spread_window))
    X = s_prev - m
    var_X = float(np.var(X))
    if var_X <= 1e-12: return None
    cov_YX = float(np.mean(Y * X))
    slope = cov_YX / var_X
    phi = 1.0 + slope
    if not (-1.0 < phi < 1.0): return None
    resid = Y - slope * X
    sigma_sq = float(np.var(resid))
    if sigma_sq <= 0: return None
    sigma = float(np.sqrt(sigma_sq))
    return phi, sigma, m


def find_best_thresholds(phi, sigma, m, s_0, sign_dir, n_paths=N_PATHS, n_steps=HOLD_MAX, seed=42):
    """Generate OU paths from s_0; find best {pi+, pi-} that maximizes PnL Sharpe.

    sign_dir: +1 for LONG entry (PnL = +(s - s_0)), -1 for SHORT entry (PnL = -(s - s_0)).
    PnL is normalized to sigma-units. Thresholds are also in sigma-units.
    """
    rng = np.random.default_rng(seed)
    # Vectorized OU simulation: paths shape (n_paths, n_steps+1)
    paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    paths[:, 0] = s_0
    eps = rng.standard_normal((n_paths, n_steps))
    one_minus_phi_m = (1.0 - phi) * m
    for k in range(1, n_steps + 1):
        paths[:, k] = phi * paths[:, k-1] + one_minus_phi_m + sigma * eps[:, k-1]

    # PnL per path/step in sigma-units (sign-corrected for direction)
    pnl_sigma = (sign_dir * (paths - s_0)) / sigma  # shape (n_paths, n_steps+1)
    # We only care about steps 1..n_steps for exits
    pnl_after_entry = pnl_sigma[:, 1:]  # shape (n_paths, n_steps)

    cummax = np.maximum.accumulate(pnl_after_entry, axis=1)
    cummin = np.minimum.accumulate(pnl_after_entry, axis=1)

    # For each pi+ in grid, first crossing step per path. If never crosses,
    # default to last index (n_steps - 1) so the lookup is in-bounds.
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

    # For each {pi+, pi-} grid pair: exit = min(cross_pp[pi+], cross_pm[pi-], n_steps-1)
    best_sharpe = -np.inf
    best_pp = best_pm = None
    for i, pp in enumerate(PI_PLUS_GRID):
        for j, pm in enumerate(PI_MINUS_GRID):
            exit_step = np.minimum(cross_pp[:, i], cross_pm[:, j])
            pnl_at_exit = pnl_after_entry[np.arange(n_paths), exit_step]
            mu = float(pnl_at_exit.mean())
            sd = float(pnl_at_exit.std())
            if sd <= 1e-9: continue
            sh = mu / sd
            if sh > best_sharpe:
                best_sharpe = sh
                best_pp = float(pp); best_pm = float(pm)
    return best_pp, best_pm, best_sharpe


def calibrate_pair(pair, spread, hurst, freq=CAL_FREQ):
    """For each calibration anchor (every `freq` bars), fit OU and find optimal thresholds.

    Note: thresholds are direction-aware. We calibrate ONE pair (pi+, pi-) per anchor
    by treating PnL = -sign(s_0 - m) * (path - s_0) -- i.e., assume the upcoming entry
    bets on reversion. (For any entry within `freq` bars, this is a valid approximation.)
    """
    sp_arr = spread.values.astype(float)
    n = len(sp_arr)
    out = []
    # Anchor only at points with at least CAL_FIT_WINDOW prior valid bars
    for t in range(CAL_FIT_WINDOW, n, freq):
        window = sp_arr[t - CAL_FIT_WINDOW:t]
        if np.isnan(window).any(): continue
        s_0 = sp_arr[t]
        if np.isnan(s_0): continue
        res = calibrate_ou(window)
        if res is None: continue
        phi, sigma, m = res
        # Direction assumed for calibration: bet on reversion
        sign_dir = -np.sign(s_0 - m) if s_0 != m else 1.0
        if sign_dir == 0: sign_dir = 1.0
        pp, pm, sh = find_best_thresholds(phi, sigma, m, s_0, sign_dir)
        if pp is None: continue
        out.append({
            "t_idx": t, "t": spread.index[t],
            "phi": phi, "sigma": sigma, "m": m,
            "pi_plus_star": pp, "pi_minus_star": pm, "sharpe": sh,
        })
    return pd.DataFrame(out).set_index("t_idx") if out else pd.DataFrame()


def simulate_pair_ou(pair, spread, hurst, cal_table, hold_max=HOLD_MAX):
    """Apply OU-calibrated exits to entries."""
    sp_arr = spread.values.astype(float)
    n = len(sp_arr)
    if cal_table.empty: return pd.DataFrame()

    # Build per-bar lookup of most recent calibration: forward-fill by t_idx
    cal_sorted = cal_table.sort_index()
    cal_idx_arr = cal_sorted.index.values
    pp_arr = cal_sorted["pi_plus_star"].values
    pm_arr = cal_sorted["pi_minus_star"].values
    sigma_arr = cal_sorted["sigma"].values
    m_arr = cal_sorted["m"].values

    # z-score using rolling stats (for entry filter)
    m_roll = spread.rolling(Z_W, min_periods=Z_W).mean().shift(1).values
    sd_roll = spread.rolling(Z_W, min_periods=Z_W).std().shift(1).values
    z_arr = (sp_arr - m_roll) / sd_roll
    abs_z_arr = np.abs(z_arr)
    sign_arr = np.sign(z_arr)
    h_arr = hurst.reindex(spread.index).values.astype(float)

    in_band = (abs_z_arr > 1).astype(int)
    in_band_diff = np.concatenate([[0], np.diff(in_band)])
    # Compute time-in-band counter
    tib = np.zeros(n, dtype=int)
    cur = 0
    for i in range(n):
        if in_band[i]:
            cur += 1
        else:
            cur = 0
        tib[i] = cur if in_band[i] else 0

    entry_mask = (abs_z_arr > 1) & (abs_z_arr < 2) & (h_arr < 0.5) & ~np.isnan(m_roll) & ~np.isnan(sd_roll)
    entry_idx = np.where(entry_mask)[0]

    rows = []
    for t in entry_idx:
        # Find most recent calibration (largest cal_idx <= t)
        cal_pos = np.searchsorted(cal_idx_arr, t, side="right") - 1
        if cal_pos < 0: continue
        pp = pp_arr[cal_pos]
        pm = pm_arr[cal_pos]
        sigma_cal = sigma_arr[cal_pos]
        if sigma_cal <= 0: continue

        sign_t = sign_arr[t]
        if sign_t == 0: continue
        s_0 = sp_arr[t]

        # Walk forward, exit on TP / SL / barrier
        exit_k = None
        exit_reason = None
        for k in range(1, hold_max + 1):
            idx = t + k
            if idx >= n:
                exit_k = k - 1; exit_reason = "data_end"; break
            cur_pnl_sigma = -sign_t * (sp_arr[idx] - s_0) / sigma_cal
            if cur_pnl_sigma >= pp:
                exit_k = k; exit_reason = "take_profit"; break
            if cur_pnl_sigma <= pm:
                exit_k = k; exit_reason = "stop_loss"; break
        if exit_k is None:
            exit_k = hold_max; exit_reason = "barrier"
        exit_spread = sp_arr[t + exit_k]
        pnl = -sign_t * (exit_spread - s_0)  # raw spread units, signed for direction

        rows.append({
            "pair": str(pair),
            "t": spread.index[t],
            "abs_z": abs_z_arr[t],
            "hurst": h_arr[t],
            "tib": tib[t],
            "pi_plus_used": pp,
            "pi_minus_used": pm,
            "sigma_cal": sigma_cal,
            "exit_k": exit_k,
            "exit_reason": exit_reason,
            "pnl": pnl,
            "pnl_sigma": pnl / sigma_cal,
        })
    return pd.DataFrame(rows)


# Run: calibrate then simulate, per pair
print("\nCalibrating OU thresholds (one per {} bars) for {} pairs...".format(CAL_FREQ, len(top_pairs)))
t0 = time.time()
cal_tables = Parallel(n_jobs=-1, verbose=0)(
    delayed(calibrate_pair)(p, spread_by_pair[p], hurst_by_pair[p]) for p in top_pairs
)
print("  done in {:.1f}s. Calibration counts per pair: min={}, max={}, median={}".format(
    time.time()-t0,
    int(min(len(c) for c in cal_tables)),
    int(max(len(c) for c in cal_tables)),
    int(np.median([len(c) for c in cal_tables]))
))

# Summary of calibrated thresholds
all_cal = pd.concat([c for c in cal_tables if not c.empty])
print("\nCalibrated threshold distribution (across all anchors, all pairs):")
print("  pi_plus*:  median={:.2f}  mean={:.2f}".format(all_cal["pi_plus_star"].median(), all_cal["pi_plus_star"].mean()))
print("  pi_minus*: median={:.2f}  mean={:.2f}".format(all_cal["pi_minus_star"].median(), all_cal["pi_minus_star"].mean()))
print("  OU phi:    median={:.4f}  mean={:.4f}".format(all_cal["phi"].median(), all_cal["phi"].mean()))
print("  OU sigma:  median={:.5f}".format(all_cal["sigma"].median()))

print("\nSimulating trades with OU-calibrated exits...")
t0 = time.time()
trade_dfs = Parallel(n_jobs=-1, verbose=0)(
    delayed(simulate_pair_ou)(p, spread_by_pair[p], hurst_by_pair[p], cal_tables[i])
    for i, p in enumerate(top_pairs)
)
trades = pd.concat([t for t in trade_dfs if not t.empty]).set_index("t").sort_index()
print("  {:,} simulated trades in {:.1f}s".format(len(trades), time.time()-t0))

# Splits
tmin, tmax = trades.index.min(), trades.index.max()
span = tmax - tmin
sp1 = tmin + span / 3
sp2 = tmin + 2 * span / 3
train = trades[trades.index < sp1]
val   = trades[(trades.index >= sp1) & (trades.index < sp2)]
test_held = trades[trades.index >= sp2]
print("\nSplits: TRAIN n={:,}  VAL n={:,}  TEST(held) n={:,}".format(len(train), len(val), len(test_held)))

def evaluate(df, label, hurst_thr=0.5, tib_thr=1e9, z_lo=1.0, z_hi=2.0):
    sub = df[(df["hurst"] <= hurst_thr) & (df["tib"] <= tib_thr) & (df["abs_z"].between(z_lo, z_hi))]
    if len(sub) == 0:
        return None
    return {
        "label": label,
        "n": len(sub),
        "mean_pnl": float(sub["pnl"].mean()),
        "mean_pnl_sigma": float(sub["pnl_sigma"].mean()),
        "median_pnl_sigma": float(sub["pnl_sigma"].median()),
        "win_rate": float((sub["pnl"] > 0).mean()),
        "tp_rate": float((sub["exit_reason"] == "take_profit").mean()),
        "sl_rate": float((sub["exit_reason"] == "stop_loss").mean()),
        "barrier_rate": float((sub["exit_reason"] == "barrier").mean()),
        "mean_hold": float(sub["exit_k"].mean()),
    }


GATES = [
    ("BASELINE (spec gate)",      {"hurst_thr": 0.5,  "tib_thr": 1e9, "z_lo": 1.0, "z_hi": 2.0}),
    ("STRICT  (Q1 cutoffs)",      {"hurst_thr": 0.38, "tib_thr": 2,   "z_lo": 1.0, "z_hi": 1.16}),
    ("MEDIUM  (top-2 quintiles)", {"hurst_thr": 0.42, "tib_thr": 6,   "z_lo": 1.0, "z_hi": 1.33}),
    ("LOOSE   (top-3 quintiles)", {"hurst_thr": 0.45, "tib_thr": 14,  "z_lo": 1.0, "z_hi": 1.52}),
]

print("\n" + "="*100)
print("PnL WITH O-U-CALIBRATED EXITS (TP at pi+*, SL at pi-*, barrier {}h)".format(HOLD_MAX))
print("="*100)

for label, params in GATES:
    print("\n--- {} ---".format(label))
    print("  thresholds: H<={hurst_thr}, tib<={tib_thr}, |z| in [{z_lo}, {z_hi}]".format(**params))
    for split_name, split_df in [("TRAIN", train), ("VAL", val)]:
        r = evaluate(split_df, split_name, **params)
        if r is None:
            print("  {}: no trades".format(split_name)); continue
        print("  {}: n={:>7,}  mean_pnl={:+.5f}  mean_pnl_sigma={:+.4f}  win={:.3f}  mean_hold={:.1f}h".format(
            split_name, r["n"], r["mean_pnl"], r["mean_pnl_sigma"], r["win_rate"], r["mean_hold"]))
        print("       exits: TP {:.1%}  SL {:.1%}  barrier {:.1%}".format(r["tp_rate"], r["sl_rate"], r["barrier_rate"]))

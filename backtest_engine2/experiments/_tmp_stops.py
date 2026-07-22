"""Simulate trades with simple stop rule:
exit on first of:
  (1) spread crosses entry-mean (take profit at completion)
  (2) |z_current| >= STOP_Z using entry's m and sd (stop-loss)
  (3) HOLD_MAX bars elapsed (vertical barrier)
Test on baseline gate vs refined gates.
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
HOLD_MAX = 72  # 72h vertical barrier
STOP_Z = 2.5   # stop-loss at |z|>=2.5 vs entry's m/sd

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


def simulate_pair(pair, spread, hurst, z_window=Z_W, hold_max=HOLD_MAX, stop_z=STOP_Z):
    """For each entry bar, simulate trade with TP/SL/barrier exit."""
    a, b = pair
    m = spread.rolling(z_window, min_periods=z_window).mean().shift(1)
    sd = spread.rolling(z_window, min_periods=z_window).std().shift(1)
    z = (spread - m) / sd
    abs_z = z.abs()
    sign_z = np.sign(z)

    in_band = (abs_z > 1).astype(int)
    grp = (in_band != in_band.shift()).cumsum()
    feat_tib = (in_band.groupby(grp).cumsum() * in_band).values

    spread_arr = spread.values.astype(float)
    m_arr = m.values.astype(float)
    sd_arr = sd.values.astype(float)
    sign_arr = sign_z.values.astype(float)
    abs_z_arr = abs_z.values.astype(float)
    hurst_arr = hurst.reindex(spread.index).values.astype(float)
    n = len(spread_arr)

    # Identify entry bars (spec gate: |z| in (1,2) AND H<0.5)
    entry_mask = (abs_z_arr > 1) & (abs_z_arr < 2) & (hurst_arr < 0.5) & ~np.isnan(m_arr) & ~np.isnan(sd_arr)
    entry_idx = np.where(entry_mask)[0]

    rows = []
    for t in entry_idx:
        m_t = m_arr[t]; sd_t = sd_arr[t]; sign_t = sign_arr[t]
        if sd_t <= 0 or sign_t == 0: continue
        entry_spread = spread_arr[t]
        exit_k = None
        exit_reason = None
        for k in range(1, hold_max + 1):
            idx = t + k
            if idx >= n:
                exit_k = k - 1
                exit_reason = "data_end"
                break
            cur_z = (spread_arr[idx] - m_t) / sd_t
            if sign_t > 0:
                # short pair: take profit when cur_z <= 0; stop when cur_z >= stop_z
                if cur_z <= 0.0:
                    exit_k = k; exit_reason = "take_profit"; break
                if cur_z >= stop_z:
                    exit_k = k; exit_reason = "stop_loss"; break
            else:
                if cur_z >= 0.0:
                    exit_k = k; exit_reason = "take_profit"; break
                if cur_z <= -stop_z:
                    exit_k = k; exit_reason = "stop_loss"; break
        if exit_k is None:
            exit_k = hold_max
            exit_reason = "barrier"
        exit_spread = spread_arr[t + exit_k]
        pnl = -sign_t * (exit_spread - entry_spread)
        rows.append({
            "pair": str(pair),
            "t": spread.index[t],
            "abs_z": abs_z_arr[t],
            "hurst": hurst_arr[t],
            "tib": feat_tib[t],
            "exit_k": exit_k,
            "exit_reason": exit_reason,
            "pnl": pnl,
        })
    return pd.DataFrame(rows)


print("\nSimulating trades for top-30 pairs (TP at z=0 cross, SL at |z|={}, barrier {}h)...".format(STOP_Z, HOLD_MAX))
t0 = time.time()
trade_dfs = Parallel(n_jobs=-1)(delayed(simulate_pair)(p, spread_by_pair[p], hurst_by_pair[p]) for p in top_pairs)
trades = pd.concat(trade_dfs).set_index("t").sort_index()
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
        return {"label": label, "n": 0}
    return {
        "label": label,
        "n": len(sub),
        "mean_pnl": float(sub["pnl"].mean()),
        "median_pnl": float(sub["pnl"].median()),
        "win_rate": float((sub["pnl"] > 0).mean()),
        "tp_rate": float((sub["exit_reason"] == "take_profit").mean()),
        "sl_rate": float((sub["exit_reason"] == "stop_loss").mean()),
        "barrier_rate": float((sub["exit_reason"] == "barrier").mean()),
        "mean_hold": float(sub["exit_k"].mean()),
        "mean_pnl_tp": float(sub.loc[sub["exit_reason"]=="take_profit","pnl"].mean()) if (sub["exit_reason"]=="take_profit").any() else None,
        "mean_pnl_sl": float(sub.loc[sub["exit_reason"]=="stop_loss","pnl"].mean()) if (sub["exit_reason"]=="stop_loss").any() else None,
        "mean_pnl_barrier": float(sub.loc[sub["exit_reason"]=="barrier","pnl"].mean()) if (sub["exit_reason"]=="barrier").any() else None,
    }


GATES = [
    ("BASELINE (spec gate)",           {"hurst_thr": 0.5,  "tib_thr": 1e9, "z_lo": 1.0, "z_hi": 2.0}),
    ("STRICT  (Q1 cutoffs)",           {"hurst_thr": 0.38, "tib_thr": 2,   "z_lo": 1.0, "z_hi": 1.16}),
    ("MEDIUM  (top-2 quintiles)",      {"hurst_thr": 0.42, "tib_thr": 6,   "z_lo": 1.0, "z_hi": 1.33}),
    ("LOOSE   (top-3 quintiles)",      {"hurst_thr": 0.45, "tib_thr": 14,  "z_lo": 1.0, "z_hi": 1.52}),
]

print("\n" + "="*100)
print("PNL WITH STOPS (take-profit at z=0 cross, stop-loss at |z|>={}, barrier at {}h)".format(STOP_Z, HOLD_MAX))
print("="*100)

for label, params in GATES:
    print("\n--- {} ---".format(label))
    print("  thresholds: H<={hurst_thr}, tib<={tib_thr}, |z| in [{z_lo}, {z_hi}]".format(**params))
    for split_name, split_df in [("TRAIN", train), ("VAL", val)]:
        r = evaluate(split_df, split_name, **params)
        if r["n"] == 0:
            print("  {}: no trades".format(split_name)); continue
        print("  {}: n={:>7,}  mean_pnl={:+.5f}  win={:.3f}  mean_hold={:.1f}h".format(
            split_name, r["n"], r["mean_pnl"], r["win_rate"], r["mean_hold"]))
        print("       exits: TP {:.1%}  SL {:.1%}  barrier {:.1%}".format(r["tp_rate"], r["sl_rate"], r["barrier_rate"]))
        print("       mean_pnl by exit: TP={:+.5f}  SL={:+.5f}  barrier={:+.5f}".format(
            r["mean_pnl_tp"] or 0, r["mean_pnl_sl"] or 0, r["mean_pnl_barrier"] or 0))

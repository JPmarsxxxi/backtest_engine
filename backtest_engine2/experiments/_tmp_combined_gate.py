"""Test combined refined gate: H<thr AND tib<=k AND |z|<=z_thr."""
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
HORIZONS = (24, 72)

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


def complete_reversion_arr(spread_arr, m_arr, sign_arr, h):
    n = len(spread_arr)
    out = np.full(n, np.nan)
    for t in range(n - h):
        s_t = sign_arr[t]; m_t = m_arr[t]
        if np.isnan(s_t) or np.isnan(m_t) or s_t == 0: continue
        window = spread_arr[t+1:t+h+1]
        if s_t > 0:
            out[t] = 1.0 if np.any(window < m_t) else 0.0
        else:
            out[t] = 1.0 if np.any(window > m_t) else 0.0
    return out


def build_features(pair, spread, hurst, z_window=Z_W):
    a, b = pair
    m = spread.rolling(z_window, min_periods=z_window).mean().shift(1)
    sd = spread.rolling(z_window, min_periods=z_window).std().shift(1)
    z = (spread - m) / sd
    abs_z = z.abs()
    sign_z = np.sign(z)

    in_band = (abs_z > 1).astype(int)
    grp = (in_band != in_band.shift()).cumsum()
    feat_tib = in_band.groupby(grp).cumsum() * in_band

    spread_arr = spread.values.astype(float)
    m_arr = m.values.astype(float)
    sign_arr = sign_z.values.astype(float)

    completed = {h: complete_reversion_arr(spread_arr, m_arr, sign_arr, h) for h in HORIZONS}
    fwd = {h: (spread.shift(-h) - spread) for h in HORIZONS}
    signed_pnl = {h: (-np.sign(z) * fwd[h]) for h in HORIZONS}

    data = {
        "pair": str(pair),
        "z": z, "abs_z": abs_z,
        "tib": feat_tib, "hurst": hurst,
    }
    for h in HORIZONS:
        data["complete_h{}".format(h)] = pd.Series(completed[h], index=spread.index)
        data["pnl_h{}".format(h)] = signed_pnl[h]

    df = pd.DataFrame(data)
    entry_mask = (df["abs_z"] > 1) & (df["abs_z"] < 2) & (df["hurst"] < 0.5)
    df = df[entry_mask].copy()
    df = df.dropna(subset=["complete_h24"])
    return df


print("\nBuilding feature datasets...")
t0 = time.time()
dfs = Parallel(n_jobs=-1)(delayed(build_features)(p, spread_by_pair[p], hurst_by_pair[p]) for p in top_pairs)
all_df = pd.concat(dfs).sort_index()
print("  {:,} entry events across {} pairs in {:.1f}s".format(len(all_df), len(top_pairs), time.time()-t0))

tmin, tmax = all_df.index.min(), all_df.index.max()
span = tmax - tmin
sp1 = tmin + span / 3
sp2 = tmin + 2 * span / 3
train = all_df[all_df.index < sp1]
val   = all_df[(all_df.index >= sp1) & (all_df.index < sp2)]
test_held = all_df[all_df.index >= sp2]
print("\nSplits: TRAIN n={:,}  VAL n={:,}  TEST(held) n={:,}".format(len(train), len(val), len(test_held)))


def evaluate_gate(df, label, hurst_thr, tib_thr, z_lo, z_hi):
    sub = df[(df["hurst"] <= hurst_thr) & (df["tib"] <= tib_thr) & (df["abs_z"].between(z_lo, z_hi))]
    n = len(sub)
    if n == 0:
        return {"split": label, "n": 0}
    p24 = sub["complete_h24"].mean()
    p72 = sub["complete_h72"].mean() if "complete_h72" in sub.columns else None
    # Drop NaN PnLs for the mean
    pnl24 = sub["pnl_h24"].dropna()
    pnl72 = sub["pnl_h72"].dropna()
    return {
        "split": label, "n": n,
        "p_complete_24": float(p24) if not np.isnan(p24) else None,
        "p_complete_72": float(p72) if not np.isnan(p72) else None,
        "mean_pnl_24": float(pnl24.mean()) if len(pnl24) else None,
        "mean_pnl_72": float(pnl72.mean()) if len(pnl72) else None,
        "win_rate_24": float((pnl24 > 0).mean()) if len(pnl24) else None,
        "win_rate_72": float((pnl72 > 0).mean()) if len(pnl72) else None,
    }


# Need complete_h72 col -- build_features only added pnl_h72 but not complete_h72 if we forgot
# Yes both are present. OK.

# Baseline: full entry set (no extra filter beyond the original gate)
print("\n" + "="*100)
print("BASELINE: original spec gate (|z| in [1,2] AND H<0.5)")
print("="*100)
for split_name, split_df in [("TRAIN", train), ("VAL", val)]:
    r = evaluate_gate(split_df, split_name, hurst_thr=0.5, tib_thr=1e9, z_lo=1.0, z_hi=2.0)
    print("  {}: n={:>7,}  P(complete h24)={:.3f}  P(complete h72)={:.3f}  mean_pnl_24={:+.5f}  mean_pnl_72={:+.5f}  win_h24={:.3f}".format(
        split_name, r["n"], r["p_complete_24"] or 0, r["p_complete_72"] or 0, r["mean_pnl_24"] or 0, r["mean_pnl_72"] or 0, r["win_rate_24"] or 0))

GATES = [
    ("STRICT (Q1 cutoffs)",      {"hurst_thr": 0.38, "tib_thr": 2,  "z_lo": 1.0, "z_hi": 1.16}),
    ("MEDIUM (top-2 quintiles)", {"hurst_thr": 0.42, "tib_thr": 6,  "z_lo": 1.0, "z_hi": 1.33}),
    ("LOOSE  (top-3 quintiles)", {"hurst_thr": 0.45, "tib_thr": 14, "z_lo": 1.0, "z_hi": 1.52}),
    ("PARTIAL: Hurst only",      {"hurst_thr": 0.38, "tib_thr": 1e9, "z_lo": 1.0, "z_hi": 2.0}),
    ("PARTIAL: TIB only",        {"hurst_thr": 0.5,  "tib_thr": 2,  "z_lo": 1.0, "z_hi": 2.0}),
    ("PARTIAL: |z| only",        {"hurst_thr": 0.5,  "tib_thr": 1e9, "z_lo": 1.0, "z_hi": 1.16}),
]

print("\n" + "="*100)
print("REFINED GATES")
print("="*100)

for label, params in GATES:
    print("\n--- {} (H<={hurst_thr}, tib<={tib_thr}, |z| in [{z_lo}, {z_hi}]) ---".format(label, **params))
    for split_name, split_df in [("TRAIN", train), ("VAL", val)]:
        r = evaluate_gate(split_df, split_name, **params)
        if r["n"] == 0:
            print("  {}: no entries".format(split_name)); continue
        baseline_n = len(split_df)
        keep_pct = 100.0 * r["n"] / baseline_n
        print("  {}: n={:>7,} ({:5.1f}% of baseline)  P(complete h24)={:.3f}  P(complete h72)={:.3f}  mean_pnl_24={:+.5f}  mean_pnl_72={:+.5f}  win_h24={:.3f}".format(
            split_name, r["n"], keep_pct,
            r["p_complete_24"] or 0, r["p_complete_72"] or 0,
            r["mean_pnl_24"] or 0, r["mean_pnl_72"] or 0,
            r["win_rate_24"] or 0))

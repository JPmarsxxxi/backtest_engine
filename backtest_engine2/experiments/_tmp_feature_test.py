"""Feature analysis using P(completed reversion) as outcome metric."""
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
    """For each bar t, does spread cross BACK through m[t] within next h bars?

    sign>0 (spread above mean) -> complete if any spread[t+1..t+h] < m[t]
    sign<0 (spread below mean) -> complete if any spread[t+1..t+h] > m[t]
    """
    n = len(spread_arr)
    out = np.full(n, np.nan)
    for t in range(n - h):
        s_t = sign_arr[t]
        m_t = m_arr[t]
        if np.isnan(s_t) or np.isnan(m_t) or s_t == 0:
            continue
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

    feat_mag = abs_z
    feat_vel = abs_z - abs_z.shift(4)
    feat_acc = feat_vel - feat_vel.shift(4)
    in_band = (abs_z > 1).astype(int)
    grp = (in_band != in_band.shift()).cumsum()
    feat_tib = in_band.groupby(grp).cumsum() * in_band
    feat_hurst = hurst
    feat_spv = spread.diff().rolling(24, min_periods=24).std().shift(1)
    lr_a = log_prices_pit[a].diff()
    lr_b = log_prices_pit[b].diff()
    feat_corr = lr_a.rolling(24, min_periods=24).corr(lr_b).shift(1)

    spread_arr = spread.values.astype(float)
    m_arr = m.values.astype(float)
    sign_arr = sign_z.values.astype(float)

    completed = {}
    for h in HORIZONS:
        completed[h] = complete_reversion_arr(spread_arr, m_arr, sign_arr, h)

    data = {
        "pair": str(pair),
        "z": z, "abs_z": abs_z,
        "mag": feat_mag, "vel": feat_vel, "acc": feat_acc,
        "tib": feat_tib, "hurst": feat_hurst,
        "spv": feat_spv, "corr": feat_corr,
    }
    for h in HORIZONS:
        data["complete_h{}".format(h)] = pd.Series(completed[h], index=spread.index)

    df = pd.DataFrame(data)
    entry_mask = (df["abs_z"] > 1) & (df["abs_z"] < 2) & (df["hurst"] < 0.5)
    df = df[entry_mask].copy()
    df = df.dropna(subset=["complete_h24", "complete_h72"])
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
test  = all_df[all_df.index >= sp2]
print("\nTime splits ({} to {}, total span {} days):".format(tmin.date(), tmax.date(), span.days))
print("  TRAIN ({} to {}): n={:,}".format(tmin.date(), sp1.date(), len(train)))
print("  VAL   ({} to {}): n={:,}".format(sp1.date(), sp2.date(), len(val)))
print("  TEST  ({} to {}): n={:,}  *** HELD OUT ***".format(sp2.date(), tmax.date(), len(test)))

# Baseline P(complete) -- no feature filter, just entry conditions
print("\n--- Baseline P(complete reversion) given entry conditions fire ---")
for split_name, split_df in [("TRAIN", train), ("VAL", val)]:
    for h in HORIZONS:
        p = split_df["complete_h{}".format(h)].mean()
        print("  {} h={:>3}: P(complete) = {:.3f}  ({:,} entries)".format(split_name, h, p, len(split_df)))

FEATURES = ["mag","vel","acc","tib","hurst","spv","corr"]
LABELS = {"mag":"|z| magnitude","vel":"|z| velocity (delta4h)","acc":"|z| acceleration","tib":"time in band","hurst":"Hurst value","spv":"spread vol (24h)","corr":"leg corr (24h)"}

def quintile_complete(df, feat, h, n_bins=5):
    df = df.dropna(subset=[feat, "complete_h{}".format(h)]).copy()
    if len(df) < 500: return None
    try:
        df["bin"] = pd.qcut(df[feat], n_bins, labels=False, duplicates="drop")
    except ValueError:
        return None
    g = df.groupby("bin").agg(
        n=("complete_h{}".format(h), "size"),
        feat_lo=(feat, "min"),
        feat_hi=(feat, "max"),
        p_complete=("complete_h{}".format(h), "mean"),
    ).sort_index()
    return g


def report_feature(feat, h):
    print("\n--- {} ({}) -- horizon h={}h ---".format(LABELS[feat], feat, h))
    train_q = quintile_complete(train, feat, h)
    val_q = quintile_complete(val, feat, h)
    if train_q is None or val_q is None:
        print("  insufficient data")
        return None
    # Q5-Q1 spread on each split
    t_diff = train_q.iloc[-1]["p_complete"] - train_q.iloc[0]["p_complete"]
    v_diff = val_q.iloc[-1]["p_complete"] - val_q.iloc[0]["p_complete"]
    t_mono = "UP" if (np.diff(train_q["p_complete"].values) >= 0).all() else "DOWN" if (np.diff(train_q["p_complete"].values) <= 0).all() else "MIXED"
    v_mono = "UP" if (np.diff(val_q["p_complete"].values) >= 0).all() else "DOWN" if (np.diff(val_q["p_complete"].values) <= 0).all() else "MIXED"
    print("  TRAIN: Q5-Q1 P(complete) spread = {:+.4f}  monotone={}".format(t_diff, t_mono))
    for i, row in train_q.iterrows():
        print("           Q{}: feat=[{:+.4f},{:+.4f}] n={:>7,} P(complete)={:.3f}".format(int(i)+1, row["feat_lo"], row["feat_hi"], int(row["n"]), row["p_complete"]))
    print("  VAL  : Q5-Q1 P(complete) spread = {:+.4f}  monotone={}".format(v_diff, v_mono))
    for i, row in val_q.iterrows():
        print("           Q{}: feat=[{:+.4f},{:+.4f}] n={:>7,} P(complete)={:.3f}".format(int(i)+1, row["feat_lo"], row["feat_hi"], int(row["n"]), row["p_complete"]))
    sign_agree = (t_diff > 0 and v_diff > 0) or (t_diff < 0 and v_diff < 0)
    return {"feature": feat, "label": LABELS[feat], "horizon": h,
            "train_diff": t_diff, "val_diff": v_diff,
            "train_mono": t_mono, "val_mono": v_mono,
            "sign_agree": sign_agree}


print("\n" + "="*100)
print("FEATURE EVALUATION using P(completed reversion) -- spread snaps back to mean")
print("="*100)

rows = []
for h in HORIZONS:
    print("\n\n###### HORIZON h={}h ######".format(h))
    for feat in FEATURES:
        r = report_feature(feat, h)
        if r is not None:
            rows.append(r)

print("\n" + "="*100)
print("SUMMARY -- sorted by absolute train_diff; sign_agree=TRUE means val confirms train direction")
print("="*100)
sm = pd.DataFrame(rows).sort_values(["horizon","train_diff"], key=lambda x: x.abs() if x.name=="train_diff" else x, ascending=[True,False])
print(sm.to_string(index=False))

print("\nReminder: TEST split (33% of data) NOT touched. To be used once on the final candidate(s).")

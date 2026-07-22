"""Standalone runner for Cell 6's trade-level diagnostic.

Loads `close` from the cached parquet, runs the strategy simulation, and saves
the analysis text + figure to disk. Avoids the 35-min engine re-run.
"""
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display
import matplotlib.pyplot as plt
from itertools import combinations
from statsmodels.tsa.stattools import coint

# ---------------- Load `close` from the Cell 1 cache ----------------
print("Loading panel from cache...")
raw = pd.read_parquet("data/binance_hourly.parquet")
MATIC_TO_POL_DATE = pd.Timestamp("2024-09-04", tz="UTC")
matic = raw[raw["symbol"] == "MATICUSDT"]
pol = raw[raw["symbol"] == "POLUSDT"]
matic_keep = matic[matic["open_time"] < MATIC_TO_POL_DATE]
pol_keep = pol[pol["open_time"] >= MATIC_TO_POL_DATE]
stitched = pd.concat([matic_keep, pol_keep], ignore_index=True).assign(symbol="MATICUSDT")
raw_x = pd.concat([raw[~raw["symbol"].isin(["MATICUSDT", "POLUSDT"])], stitched], ignore_index=True)

TICKERS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "SHIBUSDT", "AVAXUSDT", "DOTUSDT",
    "LINKUSDT", "TRXUSDT", "BCHUSDT", "NEARUSDT", "MATICUSDT",
    "UNIUSDT", "LTCUSDT", "ICPUSDT", "ETCUSDT", "HBARUSDT",
]
close = (raw_x.pivot(index="open_time", columns="symbol", values="close")
              .reindex(columns=TICKERS).sort_index())
full_idx = pd.date_range(close.index.min(), close.index.max(), freq="1h", tz="UTC")
close = close.reindex(full_idx)
close.index = close.index.tz_convert("UTC").tz_localize(None)
print(f"  panel: {len(close):,} bars, {len(close.columns)} assets")

# ---------------- Strategy params ----------------
TW = 168
HL = 720
HOLD_MAX = 72
COINT_LB = 4320
TOP_K = 20
H_THRESH = 0.5
ENTRY_LOW, ENTRY_HIGH = 1.0, 2.0
STOP_STD = 2.0
TAUS = np.array([1, 2, 4, 8, 16, 32])

def _hurst(x):
    if np.isnan(x).any(): return np.nan
    denom = np.mean(np.abs(x))
    if denom <= 0: return np.nan
    lt, lk = [], []
    for tau in TAUS:
        incr = np.abs(x[tau:] - x[:-tau])
        k = incr.mean() / denom
        if k <= 0: return np.nan
        lt.append(np.log(tau)); lk.append(np.log(k))
    return np.polyfit(np.array(lt), np.array(lk), 1)[0]

def compute_pair_series(a, b):
    sub = close[[a, b]].dropna()
    if len(sub) < HL + TW + 2:
        return None
    log_a = np.log(sub[a]); log_b = np.log(sub[b])
    lr_a = log_a.diff(); lr_b = log_b.diff()
    vol_a = lr_a.rolling(HL).std(); vol_b = lr_b.rolling(HL).std()
    b_series = vol_a.shift(1) / vol_b.shift(1)
    s = log_a - b_series * log_b
    s1 = s.shift(1)
    m = s1.rolling(TW).mean()
    std = s1.rolling(TW).std()
    h = s1.rolling(TW).apply(_hurst, raw=True)
    return pd.DataFrame({"s": s, "m": m, "std": std, "h": h}, index=sub.index)

# ---------------- Step 1: Top-20 candidates by coint over a stable window ----------------
print("\n" + "=" * 70)
print("Step 1: Top-20 candidate pairs by coint p-value")
print("=" * 70)
ANCHOR_START = 20_000
ANCHOR_LEN = COINT_LB
window = close.iloc[ANCHOR_START:ANCHOR_START + ANCHOR_LEN]
cols = [c for c in window.columns if window[c].notna().sum() >= ANCHOR_LEN * 0.95]
scores = []
for a, b in combinations(cols, 2):
    sub = window[[a, b]].dropna()
    if len(sub) < ANCHOR_LEN * 0.95: continue
    try:
        _, p, _ = coint(np.log(sub[a]), np.log(sub[b]), trend="c", autolag=None, maxlag=1)
    except Exception:
        continue
    if np.isfinite(p):
        scores.append(((a, b), p))
scores.sort(key=lambda kv: kv[1])
candidates = [pair for pair, _ in scores[:TOP_K]]
for i, ((a, b), p) in enumerate(zip(candidates, [s[1] for s in scores[:TOP_K]]), 1):
    print(f"  {i:2d}. {a:9s} / {b:9s}   coint p = {p:.4f}")

# ---------------- Step 2: Precompute signal series ----------------
print("\n" + "=" * 70)
print("Step 2: Precompute (s, m, std, H) per pair")
print("=" * 70)
pair_data = {}
t0 = time.time()
for i, (a, b) in enumerate(candidates, 1):
    print(f"  [{i:2d}/{len(candidates)}] {a:9s}/{b:9s}...", end=" ", flush=True)
    df = compute_pair_series(a, b)
    if df is None:
        print("SKIP"); continue
    df = df.reindex(close.index)
    pair_data[(a, b)] = {
        "s": df["s"].values, "m": df["m"].values,
        "std": df["std"].values, "h": df["h"].values,
    }
    print(f"valid bars: {df['h'].notna().sum():,}")
print(f"  total precompute time: {time.time() - t0:.1f}s")

# ---------------- Step 3: Simulate ----------------
print("\n" + "=" * 70)
print("Step 3: Simulate strategy")
print("=" * 70)
asset_arr = {col: close[col].values for col in close.columns}
n_bars = len(close)
warmup_idx = 5400

open_trades = {}
completed = []

t_sim = time.time()
for t_idx in range(warmup_idx, n_bars):
    # Exit
    for pair in list(open_trades.keys()):
        a, b = pair
        pd_ = pair_data.get(pair)
        if pd_ is None: continue
        s_t = pd_["s"][t_idx]
        if not np.isfinite(s_t): continue
        trade = open_trades[pair]
        m0 = trade["m0"]; std0 = trade["std0"]; side = trade["side"]
        hours_held = t_idx - trade["entry_idx"]
        exit_reason = None
        if side == +1 and s_t <= m0: exit_reason = "mean_revert"
        elif side == -1 and s_t >= m0: exit_reason = "mean_revert"
        elif side == +1 and s_t >= m0 + STOP_STD * std0: exit_reason = "stop_2sigma"
        elif side == -1 and s_t <= m0 - STOP_STD * std0: exit_reason = "stop_2sigma"
        elif hours_held >= HOLD_MAX: exit_reason = "timeout_72h"
        if exit_reason is not None:
            A_entry = asset_arr[a][trade["entry_idx"]]; B_entry = asset_arr[b][trade["entry_idx"]]
            A_exit = asset_arr[a][t_idx]; B_exit = asset_arr[b][t_idx]
            if not (np.isfinite(A_entry) and np.isfinite(B_entry) and np.isfinite(A_exit) and np.isfinite(B_exit)):
                del open_trades[pair]; continue
            A_ret = A_exit / A_entry - 1.0
            B_ret = B_exit / B_entry - 1.0
            pnl_pct = side * (B_ret - A_ret)
            completed.append({
                "pair": f"{a}/{b}", "entry_idx": trade["entry_idx"], "exit_idx": t_idx,
                "hold_h": hours_held, "side": side, "H_entry": trade["H_entry"],
                "s_entry": trade["s_entry"], "s_exit": s_t, "m0": m0, "std0": std0,
                "exit_reason": exit_reason, "pnl_pct": pnl_pct,
                "A_ret": A_ret, "B_ret": B_ret,
            })
            del open_trades[pair]
    # Entry
    for pair in candidates:
        if pair in open_trades: continue
        pd_ = pair_data.get(pair)
        if pd_ is None: continue
        s_t = pd_["s"][t_idx]; m_t = pd_["m"][t_idx]; std_t = pd_["std"][t_idx]; h_t = pd_["h"][t_idx]
        if not (np.isfinite(s_t) and np.isfinite(m_t) and np.isfinite(std_t) and np.isfinite(h_t)):
            continue
        if h_t >= H_THRESH: continue
        side = 0
        if (m_t + ENTRY_LOW * std_t) < s_t < (m_t + ENTRY_HIGH * std_t): side = +1
        elif (m_t - ENTRY_HIGH * std_t) < s_t < (m_t - ENTRY_LOW * std_t): side = -1
        if side != 0:
            open_trades[pair] = {
                "entry_idx": t_idx, "side": side, "m0": m_t, "std0": std_t,
                "s_entry": s_t, "H_entry": h_t,
            }
print(f"  sim time: {time.time() - t_sim:.1f}s,  trades: {len(completed):,}, open at end: {len(open_trades)}")

log = pd.DataFrame(completed)
log.to_parquet("data/trade_log.parquet")
print(f"  trade log saved → data/trade_log.parquet")

# ---------------- Step 4: Analysis ----------------
print("\n" + "=" * 70)
print("Step 4: Trade-level breakdown")
print("=" * 70)

print(f"\nOverall:")
print(f"  Total round-trips     : {len(log):,}")
print(f"  Mean hold (hours)     : {log['hold_h'].mean():.1f}")
print(f"  Median hold (hours)   : {log['hold_h'].median():.0f}")
print(f"  Win rate              : {(log['pnl_pct'] > 0).mean():.1%}")
print(f"  Mean gross pnl_pct    : {log['pnl_pct'].mean():+.4%}")
print(f"  Median gross pnl_pct  : {log['pnl_pct'].median():+.4%}")
print(f"  Stdev gross pnl_pct   : {log['pnl_pct'].std():.4%}")
print(f"  Sum of pnl_pct        : {log['pnl_pct'].sum():+.2%}  (additive total)")

print(f"\nBy exit reason:")
print(f"  {'reason':<15s} {'N':>7s} {'%':>6s} {'win%':>7s} {'mean pnl':>11s} {'median pnl':>12s} {'mean hold':>11s}")
print("  " + "-" * 77)
for reason, sub in log.groupby("exit_reason"):
    print(f"  {reason:<15s} {len(sub):>7,} {len(sub)/len(log)*100:>5.1f}% "
          f"{(sub['pnl_pct']>0).mean()*100:>6.1f}% {sub['pnl_pct'].mean():>+10.4%} "
          f"{sub['pnl_pct'].median():>+11.4%} {sub['hold_h'].mean():>10.1f}h")

print(f"\nBy Hurst bucket at entry (entries fire only when H<0.5):")
print(f"  {'bucket':<22s} {'N':>7s} {'win%':>7s} {'mean pnl':>11s} {'median pnl':>12s} {'%revert':>9s} {'%stop':>8s} {'%timeout':>10s}")
print("  " + "-" * 95)
buckets = [
    ("Strong anti (H<0.2)",     log["H_entry"] < 0.2),
    ("Med anti (0.2<=H<0.3)",   (log["H_entry"] >= 0.2) & (log["H_entry"] < 0.3)),
    ("Mild anti (0.3<=H<0.4)",  (log["H_entry"] >= 0.3) & (log["H_entry"] < 0.4)),
    ("Edge anti (0.4<=H<0.5)",  (log["H_entry"] >= 0.4) & (log["H_entry"] < 0.5)),
]
for name, mask in buckets:
    sub = log[mask]
    if len(sub) == 0: continue
    er = sub["exit_reason"].value_counts(normalize=True)
    print(f"  {name:<22s} {len(sub):>7,} {(sub['pnl_pct']>0).mean()*100:>6.1f}% "
          f"{sub['pnl_pct'].mean():>+10.4%} {sub['pnl_pct'].median():>+11.4%} "
          f"{er.get('mean_revert', 0)*100:>8.1f}% {er.get('stop_2sigma', 0)*100:>7.1f}% "
          f"{er.get('timeout_72h', 0)*100:>9.1f}%")

print(f"\nBreak-even cost analysis:")
mean_pnl = log["pnl_pct"].mean()
break_even_bps = mean_pnl * 2500
print(f"  Mean gross pnl_pct per trade = {mean_pnl:+.4%}")
print(f"  Break-even per-leg cost      = {break_even_bps:+.2f} bps (one-way: commission + spread)")
print(f"  Current per-leg cost (Cell 3)= 15 bps")
if break_even_bps < 0:
    print(f"  -> Strategy has NEGATIVE gross edge. No cost level makes it profitable.")
elif break_even_bps < 15:
    print(f"  -> Need cost < {break_even_bps:.1f} bps/leg to profit (current 15 bps -> loss)")
else:
    print(f"  -> Strategy IS profitable at current 15 bps/leg")

# ---------------- Plots ----------------
print("\nSaving plots to data/cell6_analysis.png ...")
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

ax = axes[0, 0]
for reason in ["mean_revert", "stop_2sigma", "timeout_72h"]:
    sub = log[log["exit_reason"] == reason]
    if len(sub) > 0:
        ax.hist(sub["pnl_pct"].clip(-0.20, 0.20) * 100, bins=40, alpha=0.5,
                label=f"{reason} (n={len(sub):,}, mean={sub['pnl_pct'].mean()*100:+.2f}%)")
ax.axvline(0, color="black", linestyle="--", alpha=0.5)
ax.set_xlabel("PnL per trade (% of leg notional)")
ax.set_ylabel("Count"); ax.set_title("PnL distribution by exit reason")
ax.legend(); ax.grid(alpha=0.3)

ax = axes[0, 1]
for reason in ["mean_revert", "stop_2sigma", "timeout_72h"]:
    sub = log[log["exit_reason"] == reason]
    if len(sub) > 0:
        ax.hist(sub["hold_h"], bins=24, alpha=0.5, label=f"{reason} (n={len(sub):,})")
ax.axvline(HOLD_MAX, color="red", linestyle="--", alpha=0.5)
ax.set_xlabel("Hold time (hours)"); ax.set_ylabel("Count")
ax.set_title("Hold-time by exit reason")
ax.legend(); ax.grid(alpha=0.3)

ax = axes[1, 0]
h_names, h_win, h_pnl = [], [], []
for name, mask in buckets:
    sub = log[mask]
    if len(sub) >= 30:
        h_names.append(name.replace(" (", "\n("))
        h_win.append((sub["pnl_pct"] > 0).mean() * 100)
        h_pnl.append(sub["pnl_pct"].mean() * 100)
x = np.arange(len(h_names))
ax.bar(x - 0.2, h_win, width=0.4, label="Win rate (%)", color="green", alpha=0.7)
ax2 = ax.twinx()
ax2.bar(x + 0.2, h_pnl, width=0.4, label="Mean pnl (%)", color="blue", alpha=0.7)
ax.set_xticks(x); ax.set_xticklabels(h_names, fontsize=8)
ax.set_ylabel("Win rate (%)"); ax2.set_ylabel("Mean pnl (%)")
ax.set_title("Win rate and mean PnL by Hurst bucket at entry")
ax.legend(loc="upper left"); ax2.legend(loc="upper right"); ax.grid(alpha=0.3)

ax = axes[1, 1]
log_sorted = log.sort_values("exit_idx").reset_index(drop=True)
log_sorted["cum_pnl"] = log_sorted["pnl_pct"].cumsum()
ax.plot(log_sorted["exit_idx"], log_sorted["cum_pnl"] * 100, lw=1)
ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel("Trade index (chronological by exit)"); ax.set_ylabel("Cumulative gross pnl_pct (%)")
ax.set_title(f"Cumulative gross PnL: ends at {log_sorted['cum_pnl'].iloc[-1]*100:+.2f}%")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("data/cell6_analysis.png", dpi=110)
print("Done.")

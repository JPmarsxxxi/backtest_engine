# Cell 5 — Stage 4 diagnostics: cost breakdown + Hurst hypothesis test
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import mannwhitneyu

# Re-use names from earlier cells: panel, result, close, qvol, strat, costs, liq, HOURS_PER_YEAR, BORROW_ANNUAL_BPS
# All cost params are kept consistent with Cell 3.

# ============================================================
# PART A — Cost breakdown (re-evaluate each component vectorized)
# ============================================================
print("=" * 70)
print("PART A — COST BREAKDOWN ($836k total)")
print("=" * 70)

trades = result.trades.reindex(columns=close.columns).fillna(0.0)
positions = result.positions.reindex(columns=close.columns).fillna(0.0)
# Align prices/volume to result.equity_curve.index (post-warmup window)
prices_run = close.reindex(result.equity_curve.index)
volume_run = qvol.reindex(result.equity_curve.index)

# Commission(bps=10): cost = |trades|.sum() * bps/1e4
commission_per_bar = trades.abs().sum(axis=1) * 10 / 1e4
# Spread(half_bps=5): cost = |trades|.sum() * half_bps/1e4
spread_per_bar = trades.abs().sum(axis=1) * 5 / 1e4
# ShortBorrow: cost = |shorts|.sum() * daily_rate, where daily_rate = 1500/1e4/HOURS_PER_YEAR
DAILY_RATE = BORROW_ANNUAL_BPS / 1e4 / HOURS_PER_YEAR
borrow_per_bar = positions.clip(upper=0.0).abs().sum(axis=1) * DAILY_RATE
# MarketImpact(k=10, sqrt, adv_lookback=168):
#   adv_shares_t = volume.rolling(168).median()   (inclusive of t — matches view.volume.tail(168))
#   adv_notional = adv_shares * price
#   ratio = |trade| / adv_notional, NaN/inf -> 0
#   cost = sum(|trade| * k * sqrt(ratio) / 1e4)
adv_shares = volume_run.rolling(168).median()
adv_notional = adv_shares * prices_run
with np.errstate(divide="ignore", invalid="ignore"):
    ratio = (trades.abs() / adv_notional).replace([np.inf, -np.inf], np.nan).fillna(0.0)
impact_per_bar = (trades.abs() * 10 * np.sqrt(ratio) / 1e4).sum(axis=1)

total_recomputed = commission_per_bar + spread_per_bar + borrow_per_bar + impact_per_bar
actual_total = result.costs.sum()

print(f"\n{'Component':<22s} {'Total ($)':>14s} {'% of costs':>11s} {'% of capital':>14s}")
print("-" * 70)
for name, series in [
    ("Commission (10 bps)", commission_per_bar),
    ("Spread (5 bps half)", spread_per_bar),
    ("MarketImpact (k=10)", impact_per_bar),
    ("ShortBorrow (15%)", borrow_per_bar),
]:
    tot = series.sum()
    print(f"{name:<22s} ${tot:>13,.0f} {tot/actual_total*100:>10.1f}% {tot/1_000_000*100:>13.1f}%")
print("-" * 70)
print(f"{'TOTAL (re-computed)':<22s} ${total_recomputed.sum():>13,.0f}")
print(f"{'TOTAL (result.costs)':<22s} ${actual_total:>13,.0f}  ← engine canonical")
diff = abs(total_recomputed.sum() - actual_total) / actual_total
print(f"Reconciliation diff   :   {diff:.2%}")

# ============================================================
# PART B — Trade statistics
# ============================================================
print("\n" + "=" * 70)
print("PART B — TRADE STATISTICS")
print("=" * 70)

n_bars_with_trades = (trades.abs().sum(axis=1) > 0).sum()
total_notional = trades.abs().sum().sum()
gross_per_bar = result.weights.abs().sum(axis=1)
net_per_bar = result.weights.sum(axis=1)

print(f"\nBars with non-zero trades  : {n_bars_with_trades:,} / {len(trades):,}  ({n_bars_with_trades/len(trades):.1%})")
print(f"Total notional traded      : ${total_notional:>14,.0f}")
print(f"Avg notional / trade-bar   : ${total_notional / max(1, n_bars_with_trades):>14,.0f}")
print(f"Avg gross exposure         : {gross_per_bar.mean():.2%}")
print(f"Avg net exposure (signed)  : {net_per_bar.mean():+.2%}")
print(f"Avg |net exposure|         : {net_per_bar.abs().mean():.2%}")
print(f"Unfilled / total notional  : {result.unfilled.abs().sum().sum() / total_notional:.1%}  ← LiquidityCap throttling rate")

# ============================================================
# PART C — Equity, drawdown, cumulative costs
# ============================================================
print("\n" + "=" * 70)
print("PART C — EQUITY / DD / COSTS plot")
print("=" * 70)

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
axes[0].plot(result.equity_curve.index, result.equity_curve.values, lw=1)
axes[0].axhline(1_000_000, color="gray", ls="--", alpha=0.5, label="initial")
axes[0].set_yscale("log")
axes[0].set_ylabel("Equity ($, log)")
axes[0].set_title("Equity curve")
axes[0].legend(); axes[0].grid(alpha=0.3)

peak = result.equity_curve.cummax()
dd = (result.equity_curve - peak) / peak
axes[1].fill_between(dd.index, dd.values, 0, color="red", alpha=0.5)
axes[1].set_ylabel("DD")
axes[1].set_title(f"Drawdown  (max = {dd.min():.1%})")
axes[1].grid(alpha=0.3)

axes[2].plot(result.costs.cumsum().index, result.costs.cumsum().values, color="orange", lw=1)
axes[2].set_ylabel("Cum costs ($)")
axes[2].set_title(f"Cumulative costs  (total = ${result.costs.sum():,.0f})")
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# PART D — Hurst-vs-HMR hypothesis test (paper's core thesis)
# ============================================================
print("\n" + "=" * 70)
print("PART D — DOES H<0.5 ACTUALLY CORRELATE WITH FASTER MEAN REVERSION?")
print("=" * 70)
print("\nPaper's claim: 'pairs with anti-persistent H revert significantly faster'")
print("Test: among ALL bars where spread is in the [1σ, 2σ] entry band (regardless of H),")
print("      stratify time-to-mean-reversion (HMR, capped at 72h) by Hurst bucket.")

# 1) Pick top-10 most-cointegrated pairs over a mid-sample window (no selection bias on
#    HMR — we're choosing pairs for stationarity, then measuring HMR for ALL their signals).
ANCHOR_START = 20_000   # ~Apr 2021 (well after early-history dropouts)
ANCHOR_LEN = 4320
print(f"\nRanking pairs by coint p-value over panel.dates[{ANCHOR_START}:{ANCHOR_START+ANCHOR_LEN}]...")
from statsmodels.tsa.stattools import coint

window = close.iloc[ANCHOR_START:ANCHOR_START + ANCHOR_LEN]
cols = [c for c in window.columns if window[c].notna().sum() >= ANCHOR_LEN * 0.95]
scores = []
for a, b in combinations(cols, 2):
    sub = window[[a, b]].dropna()
    if len(sub) < ANCHOR_LEN * 0.95:
        continue
    try:
        _, p, _ = coint(np.log(sub[a]), np.log(sub[b]), trend="c", autolag=None, maxlag=1)
    except Exception:
        continue
    if np.isfinite(p):
        scores.append(((a, b), p))
scores.sort(key=lambda kv: kv[1])
top_pairs = [pair for pair, _ in scores[:10]]
print(f"Top-10 pairs:")
for i, ((a, b), p) in enumerate(zip(top_pairs, [s[1] for s in scores[:10]]), 1):
    print(f"  {i:2d}. {a:9s} / {b:9s}   coint p = {p:.4f}")

# 2) Vectorized signal series per pair
TW = 168
HL = 720
HOLD = 72
TAUS = np.array([1, 2, 4, 8, 16, 32])

def _hurst_vec(x):
    if np.isnan(x).any():
        return np.nan
    denom = np.mean(np.abs(x))
    if denom <= 0:
        return np.nan
    log_tau, log_k = [], []
    for tau in TAUS:
        incr = np.abs(x[tau:] - x[:-tau])
        k = incr.mean() / denom
        if k <= 0:
            return np.nan
        log_tau.append(np.log(tau)); log_k.append(np.log(k))
    return np.polyfit(np.array(log_tau), np.array(log_k), 1)[0]

def compute_pair_series(a, b):
    """Full-panel vectorized s, m, std, H series for pair (a, b). Returns DataFrame or None."""
    sub = close[[a, b]].dropna()
    if len(sub) < HL + TW + 2:
        return None
    log_a = np.log(sub[a])
    log_b = np.log(sub[b])
    lr_a = log_a.diff()
    lr_b = log_b.diff()
    vol_a = lr_a.rolling(HL).std()
    vol_b = lr_b.rolling(HL).std()
    # b at bar t uses vol of returns ending at t-1
    b_series = vol_a.shift(1) / vol_b.shift(1)
    s = log_a - b_series * log_b
    s1 = s.shift(1)
    m = s1.rolling(TW).mean()
    std = s1.rolling(TW).std()           # = std of s_window (paper's std(m-s) = std(s) over window)
    h = s1.rolling(TW).apply(_hurst_vec, raw=True)
    return pd.DataFrame({"s": s, "m": m, "std": std, "h": h}, index=sub.index)

records = []  # (pair, H_at_signal, side, HMR)
t0 = time.time()
for a, b in top_pairs:
    print(f"  Scanning {a}/{b}...", end="", flush=True)
    df = compute_pair_series(a, b)
    if df is None:
        print(" SKIP")
        continue
    s_arr = df["s"].values; m_arr = df["m"].values; std_arr = df["std"].values; h_arr = df["h"].values
    n = len(s_arr)
    valid = np.isfinite(s_arr) & np.isfinite(m_arr) & np.isfinite(std_arr) & np.isfinite(h_arr)
    upper_mask = valid & (s_arr > m_arr + std_arr) & (s_arr < m_arr + 2*std_arr)  # sell signal
    lower_mask = valid & (s_arr < m_arr - std_arr) & (s_arr > m_arr - 2*std_arr)  # buy signal
    # Forward HMR scan
    n_sigs_pair = 0
    for i in np.where(upper_mask)[0]:
        m_i = m_arr[i]
        hmr = HOLD + 1
        end = min(i + HOLD + 1, n)
        for j in range(i + 1, end):
            if s_arr[j] <= m_i:
                hmr = j - i
                break
        records.append((f"{a}/{b}", h_arr[i], +1, hmr))
        n_sigs_pair += 1
    for i in np.where(lower_mask)[0]:
        m_i = m_arr[i]
        hmr = HOLD + 1
        end = min(i + HOLD + 1, n)
        for j in range(i + 1, end):
            if s_arr[j] >= m_i:
                hmr = j - i
                break
        records.append((f"{a}/{b}", h_arr[i], -1, hmr))
        n_sigs_pair += 1
    print(f"  signals={n_sigs_pair:,}")
print(f"\nTotal signals: {len(records):,}  (scan took {time.time()-t0:.1f}s)")

sig_df = pd.DataFrame(records, columns=["pair", "H", "side", "HMR"])

# 3) Stratify by H bucket
buckets = [
    ("Strongly anti-persistent (H<0.3)",   sig_df["H"] < 0.3),
    ("Mild anti-persistent   (0.3≤H<0.5)", (sig_df["H"] >= 0.3) & (sig_df["H"] < 0.5)),
    ("Near-random            (0.5≤H<0.7)", (sig_df["H"] >= 0.5) & (sig_df["H"] < 0.7)),
    ("Strongly persistent    (H≥0.7)",     sig_df["H"] >= 0.7),
]
print(f"\n{'Bucket':<40s} {'N':>7s} {'Mean HMR':>10s} {'Median HMR':>11s} {'% revert<72h':>14s}")
print("-" * 90)
for name, mask in buckets:
    sub = sig_df[mask]
    n = len(sub)
    if n == 0:
        print(f"{name:<40s} {0:>7d}")
        continue
    reverted = (sub["HMR"] <= HOLD).sum()
    print(f"{name:<40s} {n:>7,} {sub['HMR'].mean():>10.1f} {sub['HMR'].median():>11.1f} {reverted/n*100:>13.1f}%")

# 4) Histogram
fig, ax = plt.subplots(1, 1, figsize=(11, 5))
colors = ["darkgreen", "lightgreen", "orange", "red"]
for (name, mask), c in zip(buckets, colors):
    sub = sig_df[mask]
    if len(sub) > 0:
        ax.hist(sub["HMR"].clip(upper=HOLD+1), bins=24, alpha=0.5, label=f"{name} (n={len(sub):,})", color=c)
ax.axvline(HOLD, color="black", linestyle="--", alpha=0.5, label="72h max-hold")
ax.set_xlabel("HMR (bars to mean reversion)")
ax.set_ylabel("Count")
ax.set_title("Hurst hypothesis: does H<0.5 imply faster mean reversion?\n(left-shifted distribution = paper's claim)")
ax.legend(loc="upper right")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 5) Statistical test
anti = sig_df[sig_df["H"] < 0.5]["HMR"].values
pers = sig_df[sig_df["H"] >= 0.5]["HMR"].values
print(f"\nMann-Whitney U test (one-sided)")
print(f"  H0: HMR(H<0.5) is NOT stochastically less than HMR(H≥0.5)")
print(f"  H1: HMR(H<0.5) IS less (paper's claim)")
if len(anti) > 0 and len(pers) > 0:
    u, p = mannwhitneyu(anti, pers, alternative="less")
    print(f"  N(anti)={len(anti):,}, N(pers)={len(pers):,}")
    print(f"  Mean HMR: anti={anti.mean():.2f}h, pers={pers.mean():.2f}h  (diff = {anti.mean()-pers.mean():+.2f}h)")
    print(f"  U = {u:.0f}, p-value = {p:.6f}")
    if p < 0.01:
        print(f"  → REJECT H0 at α=0.01:  H<0.5 DOES revert faster (paper's claim supported)")
    elif p < 0.05:
        print(f"  → REJECT H0 at α=0.05:  H<0.5 DOES revert faster (paper's claim supported)")
    else:
        print(f"  → FAIL TO REJECT H0:    no significant speed advantage for H<0.5 in this data")

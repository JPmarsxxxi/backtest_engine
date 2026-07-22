"""Analyze the trade log from run_cell6_standalone.py."""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Force UTF-8 stdout on Windows
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

HOLD_MAX = 72

log = pd.read_parquet("data/trade_log.parquet")
print(f"Loaded trade log: {len(log):,} round-trips")

print("\n" + "=" * 70)
print("OVERALL STATS")
print("=" * 70)
print(f"  Total round-trips     : {len(log):,}")
print(f"  Mean hold (hours)     : {log['hold_h'].mean():.1f}")
print(f"  Median hold (hours)   : {log['hold_h'].median():.0f}")
print(f"  Win rate              : {(log['pnl_pct'] > 0).mean():.1%}")
print(f"  Mean gross pnl_pct    : {log['pnl_pct'].mean()*100:+.4f}%")
print(f"  Median gross pnl_pct  : {log['pnl_pct'].median()*100:+.4f}%")
print(f"  Stdev gross pnl_pct   : {log['pnl_pct'].std()*100:.4f}%")
print(f"  Sum of pnl_pct        : {log['pnl_pct'].sum()*100:+.2f}%  (additive total)")
print(f"  Max single-trade gain : {log['pnl_pct'].max()*100:+.2f}%  ({log.loc[log['pnl_pct'].idxmax(), 'pair']})")
print(f"  Max single-trade loss : {log['pnl_pct'].min()*100:+.2f}%  ({log.loc[log['pnl_pct'].idxmin(), 'pair']})")

print("\n" + "=" * 70)
print("BY EXIT REASON")
print("=" * 70)
print(f"  {'reason':<15s} {'N':>7s} {'%':>6s} {'win%':>7s} {'mean pnl':>11s} {'median pnl':>12s} {'mean hold':>11s}")
print("  " + "-" * 77)
for reason, sub in log.groupby("exit_reason"):
    pct = len(sub) / len(log) * 100
    win = (sub["pnl_pct"] > 0).mean() * 100
    mean_pnl = sub["pnl_pct"].mean() * 100
    med_pnl = sub["pnl_pct"].median() * 100
    mean_hold = sub["hold_h"].mean()
    print(f"  {reason:<15s} {len(sub):>7,} {pct:>5.1f}% {win:>6.1f}% "
          f"{mean_pnl:>+10.4f}% {med_pnl:>+11.4f}% {mean_hold:>10.1f}h")

print("\n" + "=" * 70)
print("BY HURST BUCKET AT ENTRY (entries fire only when H<0.5)")
print("=" * 70)
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
    if len(sub) == 0:
        continue
    er = sub["exit_reason"].value_counts(normalize=True)
    win = (sub["pnl_pct"] > 0).mean() * 100
    mean_pnl = sub["pnl_pct"].mean() * 100
    med_pnl = sub["pnl_pct"].median() * 100
    print(f"  {name:<22s} {len(sub):>7,} {win:>6.1f}% "
          f"{mean_pnl:>+10.4f}% {med_pnl:>+11.4f}% "
          f"{er.get('mean_revert', 0)*100:>8.1f}% {er.get('stop_2sigma', 0)*100:>7.1f}% "
          f"{er.get('timeout_72h', 0)*100:>9.1f}%")

print("\n" + "=" * 70)
print("BY PAIR (top 10 by trade count)")
print("=" * 70)
print(f"  {'pair':<25s} {'N':>7s} {'win%':>7s} {'mean pnl':>11s} {'sum pnl':>11s}")
print("  " + "-" * 65)
by_pair = log.groupby("pair").agg(
    n=("pnl_pct", "size"),
    win=("pnl_pct", lambda x: (x > 0).mean() * 100),
    mean_pnl=("pnl_pct", "mean"),
    sum_pnl=("pnl_pct", "sum"),
).sort_values("n", ascending=False)
for pair, row in by_pair.head(10).iterrows():
    print(f"  {pair:<25s} {int(row['n']):>7,} {row['win']:>6.1f}% "
          f"{row['mean_pnl']*100:>+10.4f}% {row['sum_pnl']*100:>+10.2f}%")

print("\n" + "=" * 70)
print("BREAK-EVEN COST ANALYSIS")
print("=" * 70)
# Per round-trip: 4 leg-trades (open A, open B, close A, close B), each $X notional.
# Total round-trip cost in dollars = 4X * c_bps/1e4, where c_bps is the per-leg cost.
# Gross profit per round-trip = pnl_pct * X (where X is leg notional).
# Break-even: pnl_pct = 4 * c_bps / 1e4  =>  c_bps = pnl_pct * 2500
mean_pnl = log["pnl_pct"].mean()
break_even_bps = mean_pnl * 2500
print(f"  Mean gross pnl_pct per trade : {mean_pnl*100:+.4f}%")
print(f"  Break-even per-leg cost      : {break_even_bps:+.2f} bps (commission + spread, one-way)")
print(f"  Current per-leg cost (Cell 3): ~15 bps  (10 commission + 5 spread)")
print(f"  Paper-likely institutional   : ~3 bps   (2 commission + 1 spread)")
if break_even_bps < 0:
    print(f"  -> NEGATIVE gross edge. NO cost level makes this profitable.")
elif break_even_bps < 3:
    print(f"  -> Needs cost < {break_even_bps:.1f} bps/leg. Below even institutional rates.")
elif break_even_bps < 15:
    print(f"  -> Profitable at institutional but NOT retail. Current 15 -> loss.")
else:
    print(f"  -> Profitable at current retail costs.")

# ---- Save plots ----
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

ax = axes[0, 0]
for reason in ["mean_revert", "stop_2sigma", "timeout_72h"]:
    sub = log[log["exit_reason"] == reason]
    if len(sub) > 0:
        ax.hist(sub["pnl_pct"].clip(-0.20, 0.20) * 100, bins=40, alpha=0.5,
                label=f"{reason} (n={len(sub):,}, mean={sub['pnl_pct'].mean()*100:+.2f}%)")
ax.axvline(0, color="black", linestyle="--", alpha=0.5)
ax.set_xlabel("PnL per trade (% of leg notional)")
ax.set_ylabel("Count")
ax.set_title("PnL distribution by exit reason")
ax.legend()
ax.grid(alpha=0.3)

ax = axes[0, 1]
for reason in ["mean_revert", "stop_2sigma", "timeout_72h"]:
    sub = log[log["exit_reason"] == reason]
    if len(sub) > 0:
        ax.hist(sub["hold_h"], bins=24, alpha=0.5, label=f"{reason} (n={len(sub):,})")
ax.axvline(HOLD_MAX, color="red", linestyle="--", alpha=0.5, label="72h cap")
ax.set_xlabel("Hold time (hours)")
ax.set_ylabel("Count")
ax.set_title("Hold time by exit reason")
ax.legend()
ax.grid(alpha=0.3)

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
ax.set_xticks(x)
ax.set_xticklabels(h_names, fontsize=8)
ax.set_ylabel("Win rate (%)")
ax2.set_ylabel("Mean pnl (%)")
ax.set_title("Win rate and mean PnL by Hurst bucket at entry")
ax.legend(loc="upper left")
ax2.legend(loc="upper right")
ax.grid(alpha=0.3)

ax = axes[1, 1]
log_sorted = log.sort_values("exit_idx").reset_index(drop=True)
log_sorted["cum_pnl"] = log_sorted["pnl_pct"].cumsum()
ax.plot(log_sorted["exit_idx"], log_sorted["cum_pnl"] * 100, lw=1)
ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel("Trade index (chronological by exit)")
ax.set_ylabel("Cumulative gross pnl_pct (%)")
ax.set_title(f"Cumulative gross PnL (sum of per-trade %): ends at {log_sorted['cum_pnl'].iloc[-1]*100:+.2f}%")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("data/cell6_analysis.png", dpi=110)
print("\nPlot saved -> data/cell6_analysis.png")

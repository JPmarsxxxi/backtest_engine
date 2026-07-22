"""Hunt for other signals with edge in the trade log."""
import sys
import numpy as np
import pandas as pd

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

log = pd.read_parquet("data/trade_log.parquet")
print(f"Loaded trade log: {len(log):,} round-trips\n")

# ---------- Add derived columns ----------
# Entry distance in sigma units (always positive, measures band-position from mean)
log["entry_dist_sigma"] = (log["s_entry"] - log["m0"]).abs() / log["std0"]

# Side label
log["side_label"] = log["side"].map({+1: "sell_pair (s>m)", -1: "buy_pair (s<m)"})

# Spread move (in sigma units, signed: positive = toward mean)
log["spread_move_sigma"] = (log["s_entry"] - log["s_exit"]) * log["side"] / log["std0"]

# ====================================================================
print("=" * 70)
print("SIGNAL 1: DIRECTION (sell-pair vs buy-pair)")
print("=" * 70)
print(f"  {'side':<22s} {'N':>7s} {'win%':>7s} {'mean pnl':>11s} {'median pnl':>12s} {'%revert':>9s} {'%stop':>8s} {'%timeout':>10s}")
print("  " + "-" * 95)
for side_label, sub in log.groupby("side_label"):
    er = sub["exit_reason"].value_counts(normalize=True)
    print(f"  {side_label:<22s} {len(sub):>7,} {(sub['pnl_pct']>0).mean()*100:>6.1f}% "
          f"{sub['pnl_pct'].mean()*100:>+10.4f}% {sub['pnl_pct'].median()*100:>+11.4f}% "
          f"{er.get('mean_revert',0)*100:>8.1f}% {er.get('stop_2sigma',0)*100:>7.1f}% "
          f"{er.get('timeout_72h',0)*100:>9.1f}%")

# ====================================================================
print("\n" + "=" * 70)
print("SIGNAL 2: ENTRY DISTANCE FROM MEAN (sigma units)")
print("=" * 70)
print(f"  {'entry distance':<22s} {'N':>7s} {'win%':>7s} {'mean pnl':>11s} {'median pnl':>12s} {'%revert':>9s} {'%stop':>8s}")
print("  " + "-" * 88)
dist_buckets = [
    ("1.00-1.25σ (near 1σ)", (log["entry_dist_sigma"] >= 1.0) & (log["entry_dist_sigma"] < 1.25)),
    ("1.25-1.50σ          ", (log["entry_dist_sigma"] >= 1.25) & (log["entry_dist_sigma"] < 1.50)),
    ("1.50-1.75σ          ", (log["entry_dist_sigma"] >= 1.50) & (log["entry_dist_sigma"] < 1.75)),
    ("1.75-2.00σ (near 2σ)", (log["entry_dist_sigma"] >= 1.75) & (log["entry_dist_sigma"] < 2.00)),
]
for name, mask in dist_buckets:
    sub = log[mask]
    if len(sub) == 0: continue
    er = sub["exit_reason"].value_counts(normalize=True)
    print(f"  {name:<22s} {len(sub):>7,} {(sub['pnl_pct']>0).mean()*100:>6.1f}% "
          f"{sub['pnl_pct'].mean()*100:>+10.4f}% {sub['pnl_pct'].median()*100:>+11.4f}% "
          f"{er.get('mean_revert',0)*100:>8.1f}% {er.get('stop_2sigma',0)*100:>7.1f}%")

# ====================================================================
print("\n" + "=" * 70)
print("SIGNAL 3: H bucket × Entry-distance bucket (best combos)")
print("=" * 70)
print(f"  {'H bucket':<18s} {'distance':<22s} {'N':>7s} {'win%':>7s} {'mean pnl':>11s}")
print("  " + "-" * 72)
h_buckets = [
    ("H<0.2",       log["H_entry"] < 0.2),
    ("0.2≤H<0.3",   (log["H_entry"] >= 0.2) & (log["H_entry"] < 0.3)),
    ("0.3≤H<0.4",   (log["H_entry"] >= 0.3) & (log["H_entry"] < 0.4)),
    ("0.4≤H<0.5",   (log["H_entry"] >= 0.4) & (log["H_entry"] < 0.5)),
]
results = []
for h_name, h_mask in h_buckets:
    for d_name, d_mask in dist_buckets:
        sub = log[h_mask & d_mask]
        if len(sub) < 30:
            continue
        win = (sub["pnl_pct"] > 0).mean() * 100
        mean_pnl = sub["pnl_pct"].mean() * 100
        results.append((h_name, d_name, len(sub), win, mean_pnl))
# Sort by mean pnl desc
results.sort(key=lambda r: -r[4])
print(f"  Top 10 H × distance combos by mean PnL (min 30 trades):")
for r in results[:10]:
    print(f"  {r[0]:<18s} {r[1]:<22s} {r[2]:>7,} {r[3]:>6.1f}% {r[4]:>+10.4f}%")
print(f"\n  Bottom 5 (worst combos):")
for r in results[-5:]:
    print(f"  {r[0]:<18s} {r[1]:<22s} {r[2]:>7,} {r[3]:>6.1f}% {r[4]:>+10.4f}%")

# ====================================================================
print("\n" + "=" * 70)
print("SIGNAL 4: PAIR-LEVEL — which pairs work, which don't?")
print("=" * 70)
by_pair = log.groupby("pair").agg(
    n=("pnl_pct", "size"),
    win_rate=("pnl_pct", lambda x: (x > 0).mean() * 100),
    mean_pnl=("pnl_pct", "mean"),
    sum_pnl=("pnl_pct", "sum"),
    median_pnl=("pnl_pct", "median"),
)
by_pair = by_pair.sort_values("mean_pnl", ascending=False)
print(f"\n  Top 10 pairs by mean PnL per trade:")
print(f"  {'pair':<26s} {'N':>7s} {'win%':>7s} {'mean pnl':>11s} {'sum pnl':>11s}")
print("  " + "-" * 65)
for pair, row in by_pair.head(10).iterrows():
    print(f"  {pair:<26s} {int(row['n']):>7,} {row['win_rate']:>6.1f}% "
          f"{row['mean_pnl']*100:>+10.4f}% {row['sum_pnl']*100:>+10.2f}%")
print(f"\n  Bottom 10 pairs by mean PnL per trade:")
for pair, row in by_pair.tail(10).iterrows():
    print(f"  {pair:<26s} {int(row['n']):>7,} {row['win_rate']:>6.1f}% "
          f"{row['mean_pnl']*100:>+10.4f}% {row['sum_pnl']*100:>+10.2f}%")

# Profitable pairs (mean PnL > 0)
profitable = by_pair[by_pair["mean_pnl"] > 0]
unprofitable = by_pair[by_pair["mean_pnl"] <= 0]
print(f"\n  Profitable pairs : {len(profitable)} / {len(by_pair)}  ({len(profitable)/len(by_pair)*100:.0f}%)")
print(f"  Profitable trades: {profitable['n'].sum():,} / {len(log):,}")
print(f"  → If we'd only traded profitable pairs: mean PnL = {profitable['mean_pnl'].mean()*100:+.4f}%, total = {profitable['sum_pnl'].sum()*100:+.2f}%")

# ====================================================================
print("\n" + "=" * 70)
print("SIGNAL 5: SPREAD MOVE distribution at exit (what really happens)")
print("=" * 70)
# How far does the spread actually move from entry, in σ units?
print(f"  Spread move from entry to exit (positive = toward mean):")
print(f"    Mean    : {log['spread_move_sigma'].mean():+.3f}σ")
print(f"    Median  : {log['spread_move_sigma'].median():+.3f}σ")
print(f"    25th %  : {log['spread_move_sigma'].quantile(0.25):+.3f}σ")
print(f"    75th %  : {log['spread_move_sigma'].quantile(0.75):+.3f}σ")

# Conditional on exit reason
print(f"\n  Spread move by exit reason:")
for reason, sub in log.groupby("exit_reason"):
    print(f"    {reason:<15s}: mean={sub['spread_move_sigma'].mean():+.3f}σ  median={sub['spread_move_sigma'].median():+.3f}σ")

# ====================================================================
print("\n" + "=" * 70)
print("SIGNAL 6: Would WIDER STOP (e.g. 3σ) help?")
print("=" * 70)
# Trades that stopped at 2σ — what was their PnL distribution?
# If stops fire close to entry but spreads later revert, a wider stop could help.
# Without intra-trade spread history we can't simulate exactly, but we can check
# the PnL of stopped trades: were they near the stop boundary or worse?
stopped = log[log["exit_reason"] == "stop_2sigma"]
print(f"  Of {len(stopped):,} stopped trades:")
print(f"    Mean PnL    : {stopped['pnl_pct'].mean()*100:+.4f}%")
print(f"    Median PnL  : {stopped['pnl_pct'].median()*100:+.4f}%")
print(f"    Win rate    : {(stopped['pnl_pct']>0).mean()*100:.1f}%  (some stops happen with positive PnL")
print(f"                    if the long leg outperformed the short leg unexpectedly)")
print(f"    Min PnL     : {stopped['pnl_pct'].min()*100:+.4f}%  (catastrophic blowouts)")
print(f"    p5 PnL      : {stopped['pnl_pct'].quantile(0.05)*100:+.4f}%")
print(f"    Stdev       : {stopped['pnl_pct'].std()*100:.4f}%")
# Look at the worst stops — if their losses are heavily tailed, wider stop might amplify the issue.
big_losses = (stopped["pnl_pct"] < -0.05).sum()  # worse than -5%
print(f"    Stops with loss > 5%: {big_losses:,} ({big_losses/len(stopped)*100:.1f}%)")

# ====================================================================
print("\n" + "=" * 70)
print("SIGNAL 7: Short-hold trades vs long-hold")
print("=" * 70)
# Short holds (quick exit) — were they good or bad?
hold_buckets = [
    ("1-6h (quick)",       (log["hold_h"] >= 1) & (log["hold_h"] <= 6)),
    ("7-24h (1 day)",      (log["hold_h"] >= 7) & (log["hold_h"] <= 24)),
    ("25-48h (2 day)",     (log["hold_h"] >= 25) & (log["hold_h"] <= 48)),
    ("49-71h (near timeout)", (log["hold_h"] >= 49) & (log["hold_h"] <= 71)),
    ("72h (timeout)",      log["hold_h"] >= 72),
]
print(f"  {'hold bucket':<26s} {'N':>7s} {'win%':>7s} {'mean pnl':>11s} {'%revert':>9s} {'%stop':>8s}")
print("  " + "-" * 75)
for name, mask in hold_buckets:
    sub = log[mask]
    if len(sub) == 0: continue
    er = sub["exit_reason"].value_counts(normalize=True)
    print(f"  {name:<26s} {len(sub):>7,} {(sub['pnl_pct']>0).mean()*100:>6.1f}% "
          f"{sub['pnl_pct'].mean()*100:>+10.4f}% "
          f"{er.get('mean_revert',0)*100:>8.1f}% {er.get('stop_2sigma',0)*100:>7.1f}%")

print("\nDone.")

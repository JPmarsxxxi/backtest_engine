# Cell 6 — Loser EDA: what do losing trades have in common? + tighter-filter ideas
# Uses saved trade logs (re-run Cell 4 if missing). Filter sims are on *closed trades* (hypothesis only).

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TRADE_LOG_PATH = Path("data/ou_trend_trade_log.parquet")
trades = pd.read_parquet(TRADE_LOG_PATH)
ctx = pd.json_normalize(trades["entry_ctx"])
df = pd.concat([trades.drop(columns=["entry_ctx"]), ctx], axis=1)
df["pi_width"] = df["pi_plus"] - df["pi_minus"]

# Only calculate dip_pct if m0 exists (old OU model had this)
if "m0" in df.columns:
    df["dip_pct"] = (df["p0"] - df["m0"]) / df["p0"]

win = df[df["win"]].copy()
lose = df[~df["win"]].copy()

print("=== Losers vs winners (headline) ===")
print(f"Trades {len(df):,}  |  Losers {len(lose):,} ({len(lose)/len(df):.1%})")
print("\nHow losers exited:")
print(lose["exit_reason"].value_counts().to_string())
print("\nHow winners exited:")
print(win["exit_reason"].value_counts().to_string())

# --- 1) Side-by-side stats ---
# Use columns that exist in the current model
metrics = ["roll_trend", "ou_sharpe", "mu", "sigma", "pi_width", "pi_plus", "pi_minus", "bars_held", "ret_pct"]

# Only include columns that actually exist
metrics = [c for c in metrics if c in df.columns]

rows = []
for c in metrics:
    rows.append({
        "feature": c,
        "win_median": win[c].median(),
        "lose_median": lose[c].median(),
        "win_p25": win[c].quantile(0.25),
        "lose_p25": lose[c].quantile(0.25),
        "win_p75": win[c].quantile(0.75),
        "lose_p75": lose[c].quantile(0.75),
    })
cmp = pd.DataFrame(rows)
print("\n=== 1) Feature comparison (price-space trades) ===")
print(cmp.round(3).to_string(index=False))

print("\nPlain English:")
print("- Losers hit STOP or TIMED OUT; winners mostly hit TAKE PROFIT.")
print("- Check if losers have TIGHTER stop (pi- closer to 0) or WORSE drift (lower mu).")
print("- Losers held LONGER (more max-hold timeouts).")
print("- Losers had WORSE MC sim at entry (lower ou_sharpe).")

# --- 2) Loser-only deep dive ---
print("\n=== 2) Loser breakdown ===")
stop_l = lose[lose["exit_reason"] == "pi_minus"]
time_l = lose[lose["exit_reason"] == "timeout"]
print(f"Stop losers: {len(stop_l):,}  mean ret {stop_l['ret_pct'].mean():+.2%}  median hold {stop_l['bars_held'].median():.0f}h")
print(f"Timeout losers: {len(time_l):,}  mean ret {time_l['ret_pct'].mean():+.2%}  median hold {time_l['bars_held'].median():.0f}h")

# --- 3) Filter experiments (on past trades — not a full re-backtest) ---
def _sim(sub, label):
    if len(sub) == 0:
        print(f"  {label}: no trades")
        return
    print(
        f"  {label}: n={len(sub):,}  win%={sub['win'].mean():.1%}  "
        f"mean ret/unit={sub['ret_pct'].mean():+.3%}  "
        f"stops={(sub['exit_reason']=='pi_minus').mean():.1%}  "
        f"timeouts={(sub['exit_reason']=='timeout').mean():.1%}"
    )

print("\n=== 3) Tighter entry filters (simulated on closed trades) ===")
print("Baseline:")
_sim(df, "all trades")

if "roll_trend" in df.columns:
    _sim(df[df["roll_trend"] >= 0.06], "stronger 1w uptrend (roll_trend >= 6%)")
    _sim(df[df["roll_trend"] >= 0.08], "very strong uptrend (roll_trend >= 8%)")

if "ou_sharpe" in df.columns:
    _sim(df[df["ou_sharpe"] >= 0.0], "positive MC Sharpe (ou_sharpe >= 0)")
    _sim(df[df["ou_sharpe"] >= 0.05], "strong MC Sharpe (ou_sharpe >= 0.05)")

if "mu" in df.columns:
    _sim(df[df["mu"] >= 0.0], "positive drift (mu >= 0)")
    mu_25 = df["mu"].quantile(0.25)
    mu_50 = df["mu"].quantile(0.50)
    _sim(df[df["mu"] >= mu_50], f"strong drift (mu >= {mu_50:.4f}, median)")
    
    # SNR filter
    df["snr"] = df["mu"] / df["sigma"]
    _sim(df[df["snr"] >= 0.1], "strong signal-to-noise (snr >= 0.1)")

if "pi_width" in df.columns:
    _sim(df[df["pi_width"] >= 5.0], "wide exit corridor (pi+ - pi- >= 5)")
    _sim(df[df["pi_width"] >= 7.0], "very wide corridor (pi+ - pi- >= 7)")

# Combo filter
combo_conditions = []
if "roll_trend" in df.columns:
    combo_conditions.append(df["roll_trend"] >= 0.06)
if "ou_sharpe" in df.columns:
    combo_conditions.append(df["ou_sharpe"] >= 0.0)
if "mu" in df.columns:
    combo_conditions.append(df["mu"] >= df["mu"].quantile(0.25))
if "pi_width" in df.columns:
    combo_conditions.append(df["pi_width"] >= 5.0)

if combo_conditions:
    combo = df[np.all(combo_conditions, axis=0)]
    _sim(combo, "COMBO of filters above")

# --- 4) Charts ---
plot_cols = []
plot_titles = []

if "roll_trend" in df.columns:
    plot_cols.append("roll_trend")
    plot_titles.append("1w trend strength at entry")
if "ou_sharpe" in df.columns:
    plot_cols.append("ou_sharpe")
    plot_titles.append("MC Sharpe at entry")
if "mu" in df.columns:
    plot_cols.append("mu")
    plot_titles.append("Drift (mu) at entry")
if "bars_held" in df.columns:
    plot_cols.append("bars_held")
    plot_titles.append("Hours held")
if "pi_width" in df.columns:
    plot_cols.append("pi_width")
    plot_titles.append("Exit corridor width")

# Select first 4 for plotting
plot_cols = plot_cols[:4]
plot_titles = plot_titles[:4]

if len(plot_cols) >= 4:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    
    for ax, col, title in zip(axes.flat, plot_cols, plot_titles):
        ax.hist(win[col].dropna(), bins=40, alpha=0.5, density=True, label="win", color="seagreen")
        ax.hist(lose[col].dropna(), bins=40, alpha=0.5, density=True, label="lose", color="indianred")
        ax.set_title(title)
        ax.legend()
    
    plt.suptitle("Winners (green) vs Losers (red)", y=1.02)
    plt.tight_layout()
    plt.show()

# --- 5) Suggested code knobs for Cell 1 / 2 ---
print("\n=== 5) Suggested tighter rules (for next backtest) ===")
print("  Cell 1:  R_MIN = 0.08 or 0.10  (currently 0.06 — require even stronger uptrend)")
print("  Cell 2:  skip entry if ou_sharpe < 0.0 after calibration (positive MC Sharpe only)")
print("  Cell 2:  skip if mu < 50th percentile (only take strong drift)")
print("  Cell 2:  skip if snr < 0.1 (drift must be 10% of volatility)")
print("  Cell 2:  optional: tighten pi_width to >= 7 if still getting tight corridors")
print("\nRe-run Cell 1→2→4 to test these filters; Cell 6 sim is only a rough guide on old trades.")
# Cell 5 — Trade / decision EDA (requires Cell 4 parquet or in-memory logs)
# Note: ret_pct / pi_exit are per-unit price P&L, not engine $ after costs.

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DECISION_LOG_PATH = Path("data/ou_trend_decision_log.parquet")
TRADE_LOG_PATH = Path("data/ou_trend_trade_log.parquet")

if TRADE_LOG_PATH.exists():
    trades = pd.read_parquet(TRADE_LOG_PATH)
    print(f"Loaded trades: {len(trades):,} from {TRADE_LOG_PATH}")
elif getattr(strat, "trade_log", None):
    trades = pd.DataFrame(strat.trade_log)
    print(f"In-memory trades: {len(trades):,}")
else:
    raise RuntimeError("No trade log — re-run Cell 4 first.")

if DECISION_LOG_PATH.exists():
    decisions = pd.read_parquet(DECISION_LOG_PATH)
    print(f"Loaded decisions: {len(decisions):,} from {DECISION_LOG_PATH}")
elif getattr(strat, "decision_log", None):
    decisions = pd.DataFrame(strat.decision_log)
else:
    decisions = pd.DataFrame()

trades["entry_t"] = pd.to_datetime(trades["entry_t"])
trades["exit_t"] = pd.to_datetime(trades["exit_t"])
trades["pi_width"] = trades["pi_plus"] - trades["pi_minus"]
trades["ret_pct_bps"] = trades["ret_pct"] * 1e4

print("\n=== Overview ===")
print(f"Trades: {len(trades):,}  |  Win rate: {trades['win'].mean():.1%}")
print(f"Mean ret_pct: {trades['ret_pct'].mean():+.4f}  ({trades['ret_pct_bps'].mean():+.1f} bps per unit)")
print(f"Median bars held: {trades['bars_held'].median():.0f}")
print("\nExit reasons:")
print(trades["exit_reason"].value_counts().to_string())

# Model type breakdown (ABM vs GBM)
if "model_type" in trades.columns:
    print("\nModel type distribution:")
    print(trades["model_type"].value_counts().to_string())

# --- 1) By OOS year (walk-forward windows) ---
print("\n=== 1) By OOS year ===")
by_year = trades.groupby("oos_year").agg(
    n=("win", "size"),
    win_rate=("win", "mean"),
    mean_ret=("ret_pct", "mean"),
    med_ret=("ret_pct", "median"),
    med_hold=("bars_held", "median"),
).round(4)
print(by_year.to_string())

fig, ax = plt.subplots(figsize=(8, 3.5))
colors = np.where(by_year["mean_ret"] >= 0, "seagreen", "indianred")
ax.bar(by_year.index.astype(str), by_year["mean_ret"] * 100, color=colors)
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("Mean ret_pct (%)")
ax.set_title("Mean per-unit return by OOS year (not engine $)")
plt.tight_layout()
plt.show()

# --- 2) Exit reason: wins vs losses ---
print("\n=== 2) Exit reason ===")
exit_tbl = trades.groupby("exit_reason").agg(
    n=("win", "size"),
    win_rate=("win", "mean"),
    mean_ret=("ret_pct", "mean"),
    med_hold=("bars_held", "median"),
    mean_pi_plus=("pi_plus", "mean"),
    mean_pi_minus=("pi_minus", "mean"),
).sort_values("n", ascending=False)
print(exit_tbl.round(4).to_string())

losers = trades[~trades["win"]]
print(f"\nLosers: {len(losers):,} ({len(losers)/len(trades):.1%})")
print("Losers by exit reason (%):")
print(
    losers["exit_reason"]
    .value_counts(normalize=True)
    .mul(100)
    .round(1)
    .astype(str)
    .radd("% ")
    .to_string()
)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, col in zip(axes, ["exit_reason", "oos_year"]):
    ct = trades.groupby(col)["win"].mean()
    ax.bar(ct.index.astype(str), ct.values, color="steelblue")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Win rate")
    ax.set_title(f"Win rate by {col}")
    ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.show()

# --- 3) Trend calibration vs outcomes ---
print("\n=== 3) Trend model at entry vs exit ===")

# Use columns that exist in trend model
trend_cols = ["mu", "sigma", "ou_sharpe", "pi_plus", "pi_minus", "pi_width", "bars_held", "ret_pct"]
existing_cols = [c for c in trend_cols if c in trades.columns]

print("Means by exit reason:")
print(trades.groupby("exit_reason")[existing_cols].mean(numeric_only=True).round(4).to_string())

print("\nMeans by win/loss:")
print(trades.groupby("win")[existing_cols].mean(numeric_only=True).round(4).to_string())

# If model_type exists, show breakdown
if "model_type" in trades.columns:
    print("\nMeans by model type (ABM vs GBM):")
    print(trades.groupby("model_type")[existing_cols].mean(numeric_only=True).round(4).to_string())

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
plot_cols = []
plot_titles = []

if "ou_sharpe" in trades.columns:
    plot_cols.append("ou_sharpe")
    plot_titles.append("MC Sharpe at entry")
if "pi_width" in trades.columns:
    plot_cols.append("pi_width")
    plot_titles.append("Threshold width (pi+ - pi-)")
if "bars_held" in trades.columns:
    plot_cols.append("bars_held")
    plot_titles.append("Bars held")

for ax, col, title in zip(axes, plot_cols, plot_titles):
    for reason, sub in trades.groupby("exit_reason"):
        if len(sub) < 30:
            continue
        ax.hist(sub[col].dropna(), bins=25, alpha=0.45, label=reason, density=True)
    ax.set_title(title)
    ax.legend(fontsize=7)
plt.tight_layout()
plt.show()

# --- 4) Worst / best assets ---
print("\n=== 4) By asset (min 20 trades) ===")
asset_tbl = (
    trades.groupby("asset")
    .agg(n=("win", "size"), win_rate=("win", "mean"), mean_ret=("ret_pct", "mean"))
    .query("n >= 20")
    .sort_values("mean_ret")
)
print("Worst 10:")
print(asset_tbl.head(10).round(4).to_string())
print("\nBest 10:")
print(asset_tbl.tail(10).round(4).to_string())

# --- 5) Decision funnel (rejects) ---
if len(decisions):
    print("\n=== 5) Decision funnel ===")
    print(decisions["decision"].value_counts().to_string())
    rej = decisions[decisions["decision"] == "entry_reject"]
    if len(rej):
        print("\nEntry rejects by reason:")
        print(rej["reason"].value_counts().to_string())
        if "oos_year" in rej.columns:
            print("\nRejects by year x reason (top):")
            print(
                rej.groupby(["oos_year", "reason"])
                .size()
                .sort_values(ascending=False)
                .head(12)
                .to_string()
            )

# --- 6) Drift (mu) analysis ---
if "mu" in trades.columns:
    print("\n=== 6) Drift (mu) analysis ===")
    print(f"Mean drift (winners): {trades[trades['win']]['mu'].mean():.6f}")
    print(f"Mean drift (losers):  {trades[~trades['win']]['mu'].mean():.6f}")
    
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    
    # Drift by outcome
    axes[0].hist(trades[trades["win"]]["mu"], bins=30, alpha=0.6, label="Winners", color="green")
    axes[0].hist(trades[~trades["win"]]["mu"], bins=30, alpha=0.6, label="Losers", color="red")
    axes[0].set_xlabel("Drift (mu)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Drift distribution by outcome")
    axes[0].legend()
    
    # Drift vs return scatter
    axes[1].scatter(trades["mu"], trades["ret_pct"], alpha=0.3, s=10)
    axes[1].axhline(0, color="k", lw=0.8, ls="--")
    axes[1].axvline(0, color="k", lw=0.8, ls="--")
    axes[1].set_xlabel("Drift (mu) at entry")
    axes[1].set_ylabel("Return (ret_pct)")
    axes[1].set_title("Drift vs actual return")
    
    plt.tight_layout()
    plt.show()

print(
    "\nNote: Trade ret_pct is (P_exit - P_0)/P_0 in price space. "
    "Engine year returns in Cell 4 include costs, overlap, and 5% weights."
)
# Cell 2 — Strategies H7 (CrashRebound) + H2 (VolShockReversal), with a truncation look-ahead test
# Stage 2. Strategy · skills/02-strategy.md, skills/03-risk.md · Depends on: Cell 1
# Decisions (user, 2026-09-30): callable rebalance at hour k (k=0 here); RiskConfig(max_position=1, max_gross=1, max_net=1);
# truncation test included.
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path[:0] = [ROOT, os.path.dirname(ROOT)]
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import warnings  # noqa: E402
warnings.filterwarnings("ignore", message=".*outlier.*")
from cellplot import cellplot, setup, PAL  # noqa: E402
from backtest.data.panel import DataPanel  # noqa: E402
from eda02_strategies import CrashRebound, VolShockReversal, daily_panel, h2_weights_table, h7_weights_table  # noqa: E402

setup()
prices = pd.read_parquet("prices_close_stamped.parquet")
volume = pd.read_parquet("volume_close_stamped.parquet")
panel = DataPanel(prices, volume=volume, check_outliers=False)

# full-sample vectorised weight tables (what a fast backtest would use)
d = daily_panel(prices, volume)
W7 = h7_weights_table(d["r"])
W2 = h2_weights_table(d["r"], d["v"])
RUN_START = pd.Timestamp("2018-01-01", tz="UTC")

# ---- truncation test: the strategy sees ONLY panel.as_of(t); the full-sample table must agree exactly ----
rng = np.random.default_rng(49)
days = W7.index[(W7.index >= RUN_START)]
pick = pd.DatetimeIndex(sorted(rng.choice(days, size=200, replace=False)))
# over-sample signal days so the test is not mostly all-zero rows
sig_days = W7.index[(W7.abs().sum(axis=1) > 0) & (W7.index >= RUN_START)]
sig_days2 = W2.index[(W2.abs().sum(axis=1) > 0) & (W2.index >= RUN_START)]
pick7 = pick.union(pd.DatetimeIndex(rng.choice(sig_days, 100, replace=False)))
pick2 = pick.union(pd.DatetimeIndex(rng.choice(sig_days2, 100, replace=False)))
s7, s2 = CrashRebound(k=0), VolShockReversal(k=0)
fail7 = sum(not np.allclose(s7.generate_weights(panel.as_of(t), t).reindex(W7.columns).fillna(0).values,
                            W7.loc[t].values, atol=1e-12) for t in pick7)
fail2 = sum(not np.allclose(s2.generate_weights(panel.as_of(t), t).reindex(W2.columns).fillna(0).values,
                            W2.loc[t].values, atol=1e-12) for t in pick2)
print(f"TRUNCATION TEST  H7: {len(pick7)} days checked, {fail7} mismatches   H2: {len(pick2)} days checked, {fail2} mismatches")
assert fail7 == 0 and fail2 == 0, "look-ahead: full-sample weights differ from as-of weights"

# schedule check: rebalances fall on hour k only
sched = s7.rebalance_dates(prices.index[prices.index >= RUN_START])
print(f"rebalance dates (k=0): {len(sched):,}, hours used: {sorted(set(sched.hour))}")

# ---- what the signals do (from 2018) ----
def summary(W, name):
    W = W[W.index >= RUN_START]
    act = W.abs().sum(axis=1) > 0
    n_pos = (W != 0).sum(axis=1)
    by = pd.DataFrame({"signal_days": act.groupby(W.index.year).sum(),
                       "days": act.groupby(W.index.year).size(),
                       "avg_coins_when_on": n_pos[act].groupby(W[act].index.year).mean().round(2),
                       "avg_net_when_on": W.sum(axis=1)[act].groupby(W[act].index.year).mean().round(2)})
    print(f"\n{name}\n{by.to_string()}")
    return by

b7, b2 = summary(W7, "H7 CrashRebound (long only)"), summary(W2, "H2 VolShockReversal")
print("\nH7 triggers by coin (from 2018):", (W7[W7.index >= RUN_START] > 0).sum().to_dict())
pd.concat({"H7": b7, "H2": b2}, axis=1).to_csv("cell_02_signal_counts.csv")

import matplotlib.pyplot as plt  # noqa: E402
fig, ax = plt.subplots(figsize=(9, 3.6))
x = np.arange(len(b7))
ax.bar(x - 0.2, b7["signal_days"] / b7["days"], width=0.38, color=PAL[0], label="H7 crash rebound")
ax.bar(x + 0.2, b2["signal_days"] / b2["days"], width=0.38, color=PAL[1], label="H2 volume-shock reversal")
ax.set_xticks(x)
ax.set_xticklabels([str(y) if y != 2023 else "2023 (to 03-19)" for y in b7.index])
ax.set_ylabel("share of days with a position")
ax.set_title("Cell 2 — how often each signal is on, by year (truncation test: 0 mismatches)", loc="left")
ax.legend(frameon=False)
cellplot(fig, 2, "signal_activity")

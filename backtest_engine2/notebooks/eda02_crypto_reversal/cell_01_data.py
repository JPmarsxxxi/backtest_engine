# Cell 1 — Data panel (EDA#2: crypto crash-rebound H7 + volume-shock reversal H2)
# Stage 1. Data · skills/01-data.md, skills/13-universe.md
# Decisions (user, 2026-09-30): bars RE-STAMPED to their close time (open + 1h) so a price sits at the moment it is known;
# gaps LEFT as gaps (never filled); data COPIED into backtest_engine2/data/eda02_crypto/; survivorship accepted + noted.
import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(ROOT))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from cellplot import cellplot, setup  # noqa: E402
from backtest.data.panel import DataPanel  # noqa: E402

setup()
DATA = os.path.join(ROOT, "data", "eda02_crypto")
COINS = ["BTCUSD", "ETHUSD", "BNBUSD", "SOLUSD", "XRPUSD", "DOGEUSD", "ADAUSD", "LTCUSD", "BCHUSD", "DOTUSD"]
CUTOFF = pd.Timestamp("2023-03-19 23:59:59", tz="UTC")   # TRAIN end; VAL (2023-03-25..) is NOT loaded

close, qvol, cost = {}, {}, {}
for c in COINS:
    df = pd.read_parquet(os.path.join(DATA, f"binance_{c}.parquet"))
    df["t_close"] = df["t"] + pd.Timedelta("1h")          # re-stamp: the 13:00 bar (13:00-14:00) becomes 14:00
    df = df[df["t_close"] <= CUTOFF].set_index("t_close").sort_index()
    close[c], qvol[c] = df["close"], df["quote_vol"]
    cost[c] = {"spread_rt_bp": float(df["ftmo_spread_rt_bp"].iloc[0]),
               "commission_rt_bp": float(df["ftmo_commission_rt_bp"].iloc[0])}

prices = pd.DataFrame(close)          # union of timestamps; a coin missing an hour is NaN there (never filled)
volume = pd.DataFrame(qvol).reindex(prices.index)
assert prices.index.max() <= CUTOFF, "look-ahead guard: a row past the TRAIN cutoff was loaded"
assert prices.index.is_monotonic_increasing and prices.index.is_unique
assert (prices.index.minute == 0).all(), "every stamp must be a whole hour (bar close)"

panel = DataPanel(prices, volume=volume)

# persist for the next cells (scripts, not a live kernel)
prices.to_parquet("prices_close_stamped.parquet")
volume.to_parquet("volume_close_stamped.parquet")
pd.DataFrame(cost).T.to_csv("ftmo_costs.csv")

# ---- output to validate ----
print(f"prices {prices.shape}  {prices.index.min()} -> {prices.index.max()}  (stamps = bar CLOSE time, UTC)")
first = prices.apply(lambda s: s.first_valid_index())
span_h = ((prices.index.max() - first) / pd.Timedelta("1h")).astype(int) + 1
gaps = pd.DataFrame({"first_bar_close": first.dt.strftime("%Y-%m-%d %H:%M"), "hours_present": prices.notna().sum(),
                     "hours_in_span": span_h, "missing_hours": span_h - prices.notna().sum()})
print(gaps.join(pd.DataFrame(cost).T).to_string())
print(f"global rows (union of timestamps): {len(prices):,}; rows where ALL coins missing: "
      f"{int(prices.isna().all(axis=1).sum())}")
live = prices.notna().groupby(prices.index.year).mean()
print("\nshare of hours with a price, by year:\n", live.round(3).to_string())

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

fig, ax = plt.subplots(figsize=(9, 3.6))
sns.heatmap(live.T, ax=ax, cmap="Blues", vmin=0, vmax=1, annot=True, fmt=".2f", annot_kws={"size": 7},
            cbar_kws={"label": "share of hours with a price"}, linewidths=1, linecolor="#fcfcfb")
ax.set_title("Cell 1 — hourly coverage by coin and year (TRAIN, bar-close stamps, gaps not filled)", loc="left")
ax.set_xlabel("")
ax.set_ylabel("")
cellplot(fig, 1, "coverage")

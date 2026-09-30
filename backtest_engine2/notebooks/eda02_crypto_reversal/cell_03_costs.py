# Cell 3 — Costs: time-varying spread (Abdi-Ranaldo estimate, FTMO floor), commission, nightly FTMO swap with Friday triple
# Stage 4. Costs · skills/04-costs.md §4, §4b · Depends on: Cell 1
# Decisions (user, 2026-09-30): spread source (a) ESTIMATED from Binance OHLC with the FTMO x098 half-spread as a FLOOR;
# estimator ABDI-RANALDO (user, replacing EDGE after EDGE read BTC at ~70x the FTMO measurement), lagged 1 day
# because the engine's AR uses the next day's range; window: BOTH 21d and 5d, charge the larger; swap (b) discrete nightly,
# Friday x3, weekend 0.
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path[:0] = [ROOT, os.path.dirname(ROOT)]
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from cellplot import cellplot, setup, PAL, GRAY  # noqa: E402
from backtest.costs import Commission, CompositeCostModel, RealizedSpread  # noqa: E402
from eda02_costs import FTMOSwapNightly, daily_ohlc, ar_half_bps, rollover_stamps  # noqa: E402
from eda02_strategies import daily_panel, h7_weights_table  # noqa: E402

setup()
DATA = os.path.join(ROOT, "data", "eda02_crypto")
CUTOFF = pd.Timestamp("2023-03-19 23:59:59", tz="UTC")
prices = pd.read_parquet("prices_close_stamped.parquet")
ftmo = pd.read_csv("ftmo_costs.csv", index_col=0)
COINS = list(prices.columns)

hourly = {}
for c in COINS:
    df = pd.read_parquet(os.path.join(DATA, f"binance_{c}.parquet"))
    df.index = df["t"] + pd.Timedelta("1h")                       # bar-close stamps (Cell 1 convention)
    hourly[c] = df.loc[df.index <= CUTOFF, ["open", "high", "low", "close"]]
d = daily_ohlc(hourly)

e21, e5 = ar_half_bps(d, 21), ar_half_bps(d, 5)

# ---- look-ahead test: the estimate labelled at day-end E must be computable from data up to E only ----
rng = np.random.default_rng(3)
ends = d["close"].index[d["close"].index >= pd.Timestamp("2018-03-01", tz="UTC")]
fails = 0
for E in rng.choice(ends, 60, replace=False):
    dt = {k: v.loc[:E] for k, v in d.items()}
    t21, t5 = ar_half_bps(dt, 21).iloc[-1], ar_half_bps(dt, 5).iloc[-1]
    fails += (not np.allclose(t21.fillna(-1), e21.loc[E].fillna(-1))) + (not np.allclose(t5.fillna(-1), e5.loc[E].fillna(-1)))
print(f"SPREAD TRUNCATION TEST: 60 days x 2 windows, {fails} mismatches")
assert fails == 0, "look-ahead in the spread estimate"

floor = (ftmo["spread_rt_bp"] / 2.0).reindex(COINS)          # measured FTMO half-spread, bp
est = np.fmax(e21, e5)                                       # larger of the two windows (NaN-aware)
charged_daily = est.clip(lower=floor, axis=1).fillna(floor)  # never below the FTMO measurement
# a value labelled E (00:00) is known at E and applies to bars E .. E+23h (carrying a known estimate forward, not a price)
half_bp = charged_daily.reindex(prices.index, method="ffill").fillna(floor)
half_bp.to_parquet("half_spread_bp_hourly.parquet")

costs = CompositeCostModel([RealizedSpread(half_bps=half_bp), Commission(bps=3.25), FTMOSwapNightly(prices.index)])

# ---- checks ----
roll = rollover_stamps(prices.index)
wk = roll.groupby(roll.index.tz_convert("America/New_York").dayofweek).agg(["count", "first"])
print("rollover bars by New-York weekday (0=Mon): count, multiplier\n", wk.rename(columns={"first": "mult"}).T.to_string())
print("rollover UTC hours used:", sorted(set(roll.index.hour)))

listed = d["close"].notna()                                  # complete trading days per coin
run = charged_daily.where(listed)[charged_daily.index >= pd.Timestamp("2018-01-01", tz="UTC")]
W7 = h7_weights_table(daily_panel(prices)["r"])
crash = (W7.reindex(run.index).fillna(0) > 0)
tbl = pd.DataFrame({
    "ftmo_floor_half": floor.round(2),
    "ar21_median_half": e21.reindex(run.index).median().round(2),
    "ar5_median_half": e5.reindex(run.index).median().round(2),
    "share_days_est_above_floor": (est.reindex(run.index).gt(floor, axis=1)).where(listed.reindex(run.index)).mean().round(2),
    "charged_half_median": run.median().round(2),
    "charged_half_on_H7_days": run.where(crash).median().round(2),
})
# one 24h hold, entry + exit spread at the charged rate, commission 6.5, one weekday night of swap
tbl["rt_24h_typical_bp"] = (2 * tbl["charged_half_median"] + 6.5 + 1e4 / 365 * 0.30).round(1)
tbl["rt_24h_H7_day_bp"] = (2 * tbl["charged_half_on_H7_days"] + 6.5 + 1e4 / 365 * 0.30).round(1)
print("\nhalf-spreads in bp (2018 -> 2023-03-19)\n", tbl.to_string())
tbl.to_csv("cell_03_cost_table.csv")

# hand check through the engine's own cost objects: BTC, buy 1.0 equity at a typical bar then hold through one Tue rollover
from backtest.data.panel import DataPanel  # noqa: E402
import warnings  # noqa: E402
warnings.filterwarnings("ignore")
panel = DataPanel(prices, check_outliers=False)
t_tue = roll.index[(roll.index.tz_convert("America/New_York").dayofweek == 1) & (roll.index.year == 2021)][10]
t_fri = roll.index[(roll.index.tz_convert("America/New_York").dayofweek == 4) & (roll.index.year == 2021)][10]
v = panel.as_of(t_tue)
trade = pd.Series(0.0, index=COINS)
trade["BTCUSD"] = 1.0
entry = costs.trade_cost(trade, v) * 1e4
pos = trade.copy()
sw_tue = costs.holding_cost(pos, panel.as_of(t_tue)) * 1e4
sw_fri = costs.holding_cost(pos, panel.as_of(t_fri)) * 1e4
print(f"\nengine hand check, BTC 1.0x: entry spread+commission {entry:.2f} bp (expect {half_bp.loc[t_tue,'BTCUSD']:.2f}+3.25); "
      f"swap at a Tue rollover {sw_tue:.2f} bp (expect 8.22); at a Fri rollover {sw_fri:.2f} bp (expect 24.66)")

import matplotlib.pyplot as plt  # noqa: E402
fig, axes = plt.subplots(2, 5, figsize=(14, 5.2), sharex=True)
for ax, c in zip(axes.ravel(), COINS):
    s = run[c].dropna()
    ax.plot(s.index, s.values, color=PAL[0], lw=0.8)
    ax.axhline(floor[c], color=GRAY, ls="--", lw=0.9)
    cd = run[c].where(crash[c]).dropna()
    ax.scatter(cd.index, cd.values, s=6, color=PAL[7], zorder=3)
    ax.set_yscale("log")
    ax.set_title(c, fontsize=10, loc="left")
    ax.tick_params(labelsize=7)
fig.suptitle("Cell 3 — charged half-spread (bp, log): max(FTMO floor --, Abdi-Ranaldo 21d/5d); red = H7 entry days", x=0.01,
             ha="left", fontsize=12, fontweight="bold")
fig.tight_layout()
cellplot(fig, 3, "spread_charged")

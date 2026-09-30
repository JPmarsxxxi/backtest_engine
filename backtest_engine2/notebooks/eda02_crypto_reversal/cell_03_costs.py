# Cell 3 — Costs: MEASURED spread (Binance quotes via Tardis, FTMO floor), commission, nightly FTMO swap (Friday x3)
# Stage 4. Costs · skills/04-costs.md §4, §4b · Depends on: Cell 1
# Decisions (user, 2026-09-30): spread source = MEASURED (from_bidask principle: real bid/ask, Tardis first-of-month
# samples, verified against Binance bars in finding-alphas p8b_quotes_verify.py); charged = max(FTMO x098 half, measured
# Binance half by hour of day); TWO cost settings for Cell 4: LOW = measured median, HIGH = measured 90th percentile
# (stand-in for crash-day widening, which the monthly sample cannot measure). OHLC estimators (EDGE, Abdi-Ranaldo) were
# tried and REJECTED: daily crypto volatility swamps the bid-ask bounce (BTC read 3-57 bp vs 0.09 bp measured).
# Swap: discrete, once per night at FTMO server midnight (17:00 New York), Fri x3, weekend 0.
import os
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path[:0] = [ROOT, os.path.dirname(ROOT)]
os.chdir(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
from cellplot import cellplot, setup, PAL, GRAY  # noqa: E402
from backtest.costs import Commission, CompositeCostModel, RealizedSpread  # noqa: E402
from backtest.data.panel import DataPanel  # noqa: E402
from eda02_costs import FTMOSwapNightly, measured_half_frames, rollover_stamps  # noqa: E402
from eda02_strategies import daily_panel, h7_weights_table  # noqa: E402

setup()
QDIR = os.path.join(ROOT, "data", "eda02_crypto", "quotes")
prices = pd.read_parquet("prices_close_stamped.parquet")
ftmo = pd.read_csv("ftmo_costs.csv", index_col=0)
COINS = list(prices.columns)
floor = (ftmo["spread_rt_bp"] / 2.0).reindex(COINS)
quotes = {c: pd.read_parquet(os.path.join(QDIR, f"binance_quotes_minute_{c}.parquet")) for c in COINS}
for c in COINS:
    quotes[c]["t"] = pd.to_datetime(quotes[c]["t"], utc=True)
    assert quotes[c]["t"].max() <= pd.Timestamp("2023-03-19 23:59:59", tz="UTC")

half_lo, half_hi = measured_half_frames(quotes, prices.index, floor)

# ---- look-ahead test: the charge at t must be computable from quotes up to t only ----
rng = np.random.default_rng(7)
ts = prices.index[prices.index >= pd.Timestamp("2019-06-01", tz="UTC")]
fails = 0
for t in rng.choice(ts, 60, replace=False):
    qt = {c: q[q["t"] < t] for c, q in quotes.items()}
    qt = {c: (q if len(q) else quotes[c].iloc[:0]) for c, q in qt.items()}
    ok = {c: q for c, q in qt.items() if len(q)}
    lo_t, hi_t = measured_half_frames(ok, pd.DatetimeIndex([t]), floor)
    fails += int(not np.allclose(lo_t.iloc[0].values, half_lo.loc[t, list(ok)].values)) + \
        int(not np.allclose(hi_t.iloc[0].values, half_hi.loc[t, list(ok)].values))
print(f"SPREAD TRUNCATION TEST: 60 bars x 2 settings, {fails} mismatches")
assert fails == 0, "look-ahead in the measured spread frame"

half_lo.to_parquet("half_spread_bp_hourly_LOW.parquet")
half_hi.to_parquet("half_spread_bp_hourly_HIGH.parquet")
if os.path.exists("half_spread_bp_hourly.parquet"):
    os.remove("half_spread_bp_hourly.parquet")        # the rejected Abdi-Ranaldo frame

swap = FTMOSwapNightly(prices.index)
costs_lo = CompositeCostModel([RealizedSpread(half_bps=half_lo), Commission(bps=3.25), swap])
costs_hi = CompositeCostModel([RealizedSpread(half_bps=half_hi), Commission(bps=3.25), swap])

# ---- checks ----
roll = rollover_stamps(prices.index)
wk = roll.groupby(roll.index.tz_convert("America/New_York").dayofweek).agg(["count", "first"])
print("rollover bars by New-York weekday (0=Mon): count, multiplier\n", wk.rename(columns={"first": "mult"}).T.to_string())

RUN = pd.Timestamp("2018-01-01", tz="UTC")
listed = prices.notna() & (prices.index >= RUN)[:, None]
at0 = prices.index.hour == 0
W7 = h7_weights_table(daily_panel(prices)["r"])
h7_bar = W7.reindex(prices.index).fillna(0) > 0            # the 00:00 bars where H7 enters
tbl = pd.DataFrame({
    "ftmo_floor_half": floor.round(3),
    "LOW_half_median": half_lo.where(listed).median().round(3),
    "HIGH_half_median": half_hi.where(listed).median().round(3),
    "share_bars_measured_above_floor_LOW": half_lo.gt(floor, axis=1).where(listed).mean().round(2),
    "LOW_half_at_00UTC": half_lo.where(listed)[at0].median().round(3),
    "HIGH_half_on_H7_entries": half_hi.where(h7_bar & listed).median().round(3),
})
swap_night = 1e4 * 0.30 / 365
tbl["rt_24h_LOW_bp"] = (2 * tbl["LOW_half_at_00UTC"] + 6.5 + swap_night).round(2)
tbl["rt_24h_HIGH_bp"] = (2 * tbl["HIGH_half_on_H7_entries"] + 6.5 + swap_night).round(2)
print("\nhalf-spreads in bp, 2018 -> 2023-03-19, listed bars only (2018 uses the 2019-04 sample: assumption)\n",
      tbl.to_string())
tbl.to_csv("cell_03_cost_table.csv")

panel = DataPanel(prices, check_outliers=False)
t_tue = roll.index[(roll.index.tz_convert("America/New_York").dayofweek == 1) & (roll.index.year == 2021)][10]
t_fri = roll.index[(roll.index.tz_convert("America/New_York").dayofweek == 4) & (roll.index.year == 2021)][10]
trade = pd.Series(0.0, index=COINS)
trade["DOGEUSD"] = 1.0
e_lo = costs_lo.trade_cost(trade, panel.as_of(t_tue)) * 1e4
e_hi = costs_hi.trade_cost(trade, panel.as_of(t_tue)) * 1e4
print(f"\nengine hand check, DOGE 1.0x at {t_tue}: entry LOW {e_lo:.3f} bp (expect {half_lo.loc[t_tue,'DOGEUSD']:.3f}+3.25), "
      f"HIGH {e_hi:.3f} bp (expect {half_hi.loc[t_tue,'DOGEUSD']:.3f}+3.25); "
      f"swap Tue {costs_lo.holding_cost(trade, panel.as_of(t_tue))*1e4:.2f} bp (8.22), "
      f"Fri {costs_lo.holding_cost(trade, panel.as_of(t_fri))*1e4:.2f} bp (24.66)")

import matplotlib.pyplot as plt  # noqa: E402
daily_lo = half_lo.where(listed).resample("MS").median()
daily_hi = half_hi.where(listed).resample("MS").median()
fig, axes = plt.subplots(2, 5, figsize=(14, 5.2), sharex=True)
for ax, c in zip(axes.ravel(), COINS):
    ax.plot(daily_hi.index, daily_hi[c], color=PAL[5], lw=1.2, label="HIGH (p90)")
    ax.plot(daily_lo.index, daily_lo[c], color=PAL[0], lw=1.4, label="LOW (median)")
    ax.axhline(floor[c], color=GRAY, ls="--", lw=0.9)
    ax.axvline(pd.Timestamp("2019-04-02", tz="UTC"), color=GRAY, lw=0.6, ls=":")
    ax.set_yscale("log")
    ax.set_title(c, fontsize=10, loc="left")
    ax.tick_params(labelsize=7)
axes[0, 0].legend(fontsize=7, frameon=False)
fig.suptitle("Cell 3 — charged half-spread, bp (log, monthly median): max(FTMO floor --, MEASURED Binance by hour); "
             "left of dotted line = 2019-04 sample stands in", x=0.01, ha="left", fontsize=11, fontweight="bold")
fig.tight_layout()
cellplot(fig, 3, "spread_charged")

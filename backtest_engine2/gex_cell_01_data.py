# Cell 1 — #017 GEX intraday regime: data load + align (no engine; decay-check is panel-level).
# US500 30m close-stamped panel (reused verbatim from #012/#013, label="right" = PIT-honest).
# GEX: SqueezeMetrics daily (data/sqz_dix_gex.csv, 2011-2026), keyed by ET trading day,
# LAGGED 1 NYSE day: gex[T] is knowable at close T -> informs session T+1 (no lookahead).
import numpy as np
import pandas as pd
from backtest.data import DataPanel

ENG = r"C:\Users\User\backtest_engine\backtest_engine2"

RAW = pd.read_parquet(ENG + r"\data\us500_1m.parquet")
s = RAW.set_index("open_time")["close"].sort_index()
s.index = s.index.tz_localize(None)  # tz-naive UTC
px30 = s.resample("30min", closed="left", label="right").last().dropna()
prices = px30.to_frame("USA500")
panel = DataPanel(prices, check_outliers=False)

ET = "America/New_York"
et_idx = prices.index.tz_localize("UTC").tz_convert(ET)
px_ = prices["USA500"]

# --- GEX daily, lag 1 NYSE day, keyed by plain trading-day date ---
g = pd.read_csv(ENG + r"\data\sqz_dix_gex.csv", parse_dates=["date"]).set_index("date").sort_index()
gex_lag = g["gex"].shift(1)                                   # PIT: prior-session GEX
gex_lag.index = pd.DatetimeIndex(gex_lag.index.normalize())  # date key

us_days = pd.DatetimeIndex(sorted(pd.Series(et_idx.date).unique()))
overlap = us_days.intersection(gex_lag.dropna().index)

print(f"US500 30m bars : {len(panel.dates)} | {panel.dates[0]} -> {panel.dates[-1]}")
print(f"US500 days     : {len(us_days)} | {us_days.min().date()} -> {us_days.max().date()}")
print(f"GEX days       : {len(g)} | {g.index.min().date()} -> {g.index.max().date()}")
print(f"Overlap (day has lagged GEX): {len(overlap)} | {overlap.min().date()} -> {overlap.max().date()}")
ov22 = overlap[overlap.year >= 2022]
print(f"  2022+ subset : {len(ov22)} days")

gj = gex_lag.reindex(overlap)
print("GEX(lag) quintile cuts (overlap):", {k: round(v, 3) for k, v in gj.quantile([.2, .4, .6, .8]).items()})
print("GEX(lag) pos/neg on overlap     :", int((gj > 0).sum()), "/", int((gj < 0).sum()))
print("GEX(lag) describe (1e9 units)   :", {k: round(v / 1e9, 3) for k, v in gj.describe().items()})

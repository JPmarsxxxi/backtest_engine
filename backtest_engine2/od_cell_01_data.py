# Cell 1 — Setup + data load: US500 30m panel from Dukascopy 1-min BID closes.
# 1-min bars are open-time-stamped; resampling with label="right" stamps each 30m bar
# by its CLOSE time -> PIT-honest at build (the #011 lesson, applied up front).
import pandas as pd
from backtest.data import DataPanel

RAW = pd.read_parquet(r"C:\Users\User\backtest_engine\backtest_engine2\data\us500_1m.parquet")
s = RAW.set_index("open_time")["close"].sort_index()
s.index = s.index.tz_localize(None)  # tz-naive UTC

px30 = s.resample("30min", closed="left", label="right").last().dropna()
prices = px30.to_frame("USA500")

panel = DataPanel(prices, check_outliers=True)

print(f"Bars: {len(panel.dates)}")
print(f"Range: {panel.dates[0]} -> {panel.dates[-1]}")
print("Bars per year:", prices.groupby(prices.index.year).size().to_dict())
bars_per_day = prices.groupby(prices.index.normalize()).size()
print(f"Days with data: {len(bars_per_day)}; bars/day median: {bars_per_day.median():.0f}")
print(f"NaN fraction: {prices['USA500'].isna().mean():.3%}")
prices.tail(3)

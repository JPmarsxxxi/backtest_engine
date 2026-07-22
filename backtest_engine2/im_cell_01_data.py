# Cell 1 — #012 setup + data: US500 30m close-stamped panel (reused from #013 build).
# 1-min Dukascopy BID closes -> 30m bars, label="right" stamps by CLOSE time (PIT-honest).
# check_outliers=False: COVID March-2020 moves reviewed and accepted in #013 Cell 1.
# Verify the three ET stamps #012 needs exist daily: 10:00 (signal), 15:30 (entry), 16:00 (exit).
import numpy as np
import pandas as pd
from backtest.data import DataPanel

RAW = pd.read_parquet(r"C:\Users\User\backtest_engine\backtest_engine2\data\us500_1m.parquet")
s = RAW.set_index("open_time")["close"].sort_index()
s.index = s.index.tz_localize(None)  # tz-naive UTC

px30 = s.resample("30min", closed="left", label="right").last().dropna()
prices = px30.to_frame("USA500")
panel = DataPanel(prices, check_outliers=False)

ET = "America/New_York"


def _et(dates):
    return dates.tz_localize("UTC").tz_convert(ET)


print(f"Bars: {len(panel.dates)} | range {panel.dates[0]} -> {panel.dates[-1]}")

et_idx = _et(prices.index)
px_ = prices["USA500"]
weekday = pd.Series(et_idx.dayofweek, index=prices.index)
for hh, mm, label in [(10, 0, "10:00 signal"), (15, 30, "15:30 entry"), (16, 0, "16:00 exit")]:
    mask = (et_idx.hour == hh) & (et_idx.minute == mm) & (weekday < 5).values
    days = pd.Series(et_idx[mask].date).nunique()
    print(f"  {label}: {int(mask.sum())} stamps on {days} weekdays")

# Trading-day alignment: how many days have ALL THREE stamps (a complete signal->trade day)?
df = pd.DataFrame({"d": et_idx.date, "h": et_idx.hour, "m": et_idx.minute})
have = df.groupby("d").apply(
    lambda g: ((g.h == 10) & (g.m == 0)).any()
    and ((g.h == 15) & (g.m == 30)).any()
    and ((g.h == 16) & (g.m == 0)).any()
)
print(f"Complete days (10:00 + 15:30 + 16:00 ET): {int(have.sum())} of {len(have)}")
yrs = pd.Series(pd.to_datetime(have[have].index)).dt.year.value_counts().sort_index()
print("Complete days per year:", yrs.to_dict())

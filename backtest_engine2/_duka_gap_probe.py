"""Probe the 2022-2023 gap: which days are missing, and do alternate Dukascopy
instrument names have data there? Tests one known-missing weekday across name variants."""
import lzma

import httpx
import pandas as pd

RAW = pd.read_parquet(r"C:\Users\User\backtest_engine\backtest_engine2\data\us500_1m.parquet")
have = set(pd.DatetimeIndex(RAW["open_time"]).tz_localize(None).normalize().unique())
all_weekdays = pd.date_range("2022-01-01", "2023-12-31", freq="B")
missing = [d for d in all_weekdays if d not in have]
print(f"Missing weekdays 2022-2023: {len(missing)}")
if missing:
    print("first:", missing[0].date(), "last:", missing[-1].date())
    runs = []
    start = prev = missing[0]
    for d in missing[1:]:
        if (d - prev).days > 7:
            runs.append((start.date(), prev.date()))
            start = d
        prev = d
    runs.append((start.date(), prev.date()))
    print("contiguous runs:", runs[:10])

probe_day = missing[len(missing) // 2]
print(f"\nProbing {probe_day.date()} across instrument names:")
client = httpx.Client(timeout=20)
for name in ["USA500IDXUSD", "US500IDXUSD", "USA500.IDXUSD", "SPX500IDXUSD",
             "USSPX500IDXUSD", "E_SXY", "USA30IDXUSD"]:
    url = (f"https://datafeed.dukascopy.com/datafeed/{name}/{probe_day.year}/"
           f"{probe_day.month - 1:02d}/{probe_day.day:02d}/BID_candles_min_1.bi5")
    try:
        r = client.get(url)
        size = len(r.content)
        note = ""
        if r.status_code == 200 and size > 0:
            try:
                note = f"-> {len(lzma.decompress(r.content)) // 24} records"
            except lzma.LZMAError:
                note = "-> not lzma"
        print(f"  {name:>16}: HTTP {r.status_code}, {size} bytes {note}")
    except Exception as e:
        print(f"  {name:>16}: {e}")

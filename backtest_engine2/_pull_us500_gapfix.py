"""Re-pull the 2022-07-26 -> 2023-05-22 gap for USA500IDXUSD without the vol>0 filter.
Keep minutes where vol>0 OR price changed vs previous minute (drops closed-market padding).
Merge with existing data/us500_1m.parquet (dedupe on open_time, keep new)."""
import lzma
import struct
import time

import httpx
import pandas as pd

BASE = "https://datafeed.dukascopy.com/datafeed/USA500IDXUSD"
OUT = r"C:\Users\User\backtest_engine\backtest_engine2\data\us500_1m.parquet"

frames = []
ok = empty = 0
client = httpx.Client(timeout=30)
for day in pd.date_range("2022-07-26", "2023-05-22", freq="D"):
    url = f"{BASE}/{day.year}/{day.month - 1:02d}/{day.day:02d}/BID_candles_min_1.bi5"
    for attempt in range(3):
        try:
            r = client.get(url)
            break
        except Exception:
            time.sleep(2 * (attempt + 1))
    else:
        continue
    if r.status_code != 200 or len(r.content) == 0:
        empty += 1
        continue
    try:
        raw = lzma.decompress(r.content)
    except lzma.LZMAError:
        empty += 1
        continue
    n = len(raw) // 24
    recs = [struct.unpack(">5if", raw[i * 24:(i + 1) * 24]) for i in range(n)]
    df = pd.DataFrame(recs, columns=["sec", "open", "close", "low", "high", "vol"])
    active = (df["vol"] > 0) | (df["close"].diff().fillna(0) != 0)
    df = df[active]
    if len(df) < 60:           # under an hour of activity = closed day
        empty += 1
        continue
    df["open_time"] = day.tz_localize("UTC") + pd.to_timedelta(df["sec"], unit="s")
    df["close"] = df["close"] / 1000.0
    frames.append(df[["open_time", "close", "vol"]])
    ok += 1
    time.sleep(0.05)

new = pd.concat(frames, ignore_index=True).rename(columns={"vol": "quote_volume"})
new["symbol"] = "USA500"
old = pd.read_parquet(OUT)
merged = (pd.concat([old, new], ignore_index=True)
          .drop_duplicates("open_time", keep="last")
          .sort_values("open_time"))
merged.to_parquet(OUT, index=False)
print(f"gap days recovered: {ok} (skipped {empty} closed days)")
print(f"parquet: {len(old):,} -> {len(merged):,} rows")
days = pd.DatetimeIndex(merged["open_time"]).tz_localize(None).normalize()
print("days per year:", pd.Series(days.unique()).dt.year.value_counts().sort_index().to_dict())

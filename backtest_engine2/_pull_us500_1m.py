"""Pull USA500IDXUSD 1-min BID candles from Dukascopy, 2019-01-01 -> 2026-06-10.
One bi5 file per day (1440 records, 24-byte BE structs, prices x1000). Month is 0-indexed.
Weekends/holidays: 404 or empty -> skipped. Output: data/us500_1m.parquet (long format).
"""
import lzma
import struct
import time

import httpx
import pandas as pd

BASE = "https://datafeed.dukascopy.com/datafeed/USA500IDXUSD"
OUT = r"C:\Users\User\backtest_engine\backtest_engine2\data\us500_1m.parquet"
START = pd.Timestamp("2019-01-01")
END = pd.Timestamp("2026-06-10")

frames = []
ok = empty = miss = 0
client = httpx.Client(timeout=30)
for day in pd.date_range(START, END, freq="D"):
    url = f"{BASE}/{day.year}/{day.month - 1:02d}/{day.day:02d}/BID_candles_min_1.bi5"
    for attempt in range(3):
        try:
            r = client.get(url)
            break
        except Exception:
            time.sleep(2 * (attempt + 1))
    else:
        miss += 1
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
    df = df[df["vol"] > 0]  # drop dead minutes (market closed padding)
    if df.empty:
        empty += 1
        continue
    df["open_time"] = day.tz_localize("UTC") + pd.to_timedelta(df["sec"], unit="s")
    df["close"] = df["close"] / 1000.0
    frames.append(df[["open_time", "close", "vol"]])
    ok += 1
    if ok % 200 == 0:
        print(f"{ok} days parsed (at {day.date()}), {empty} empty/closed, {miss} failed", flush=True)
    time.sleep(0.05)

out = pd.concat(frames, ignore_index=True).sort_values("open_time")
out["symbol"] = "USA500"
out = out.rename(columns={"vol": "quote_volume"})
out.to_parquet(OUT, index=False)
print(f"DONE: {len(out):,} 1-min bars from {ok} trading days ({empty} closed days, {miss} failed)")
print(out.head(2))
print(out.tail(2))

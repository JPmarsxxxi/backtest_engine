"""Pull EURUSD 1-min BID candles from Dukascopy 2019-01-01 -> 2026-06-13, resample to HOURLY.
Same bi5 format as us500 (24-byte BE >5if, month 0-indexed). FX 5-digit -> prices x100000.
Output: data/eurusd_h1.parquet (long format: open_time UTC, close, quote_volume, symbol)."""
import lzma
import struct
import time

import httpx
import pandas as pd

BASE = "https://datafeed.dukascopy.com/datafeed/EURUSD"
OUT = r"C:\Users\User\backtest_engine\backtest_engine2\data\eurusd_h1.parquet"
START = pd.Timestamp("2019-01-01")
END = pd.Timestamp("2026-06-13")

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
    df = df[df["vol"] > 0]
    if df.empty:
        empty += 1
        continue
    df["open_time"] = day.tz_localize("UTC") + pd.to_timedelta(df["sec"], unit="s")
    df["close"] = df["close"] / 100000.0
    frames.append(df[["open_time", "close", "vol"]])
    ok += 1
    if ok % 300 == 0:
        print(f"{ok} days (at {day.date()}), {empty} closed, {miss} failed", flush=True)
    time.sleep(0.03)

m1 = pd.concat(frames, ignore_index=True).set_index("open_time").sort_index()
h1 = m1.resample("1h").agg({"close": "last", "vol": "sum"}).dropna(subset=["close"])
h1 = h1.reset_index().rename(columns={"vol": "quote_volume"})
h1["symbol"] = "EURUSD"
h1.to_parquet(OUT, index=False)
print(f"DONE: {len(h1):,} hourly bars from {ok} trading days ({empty} closed, {miss} failed)")
print(h1.head(2))
print(h1.tail(2))

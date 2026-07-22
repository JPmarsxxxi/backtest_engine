"""Pull any Dukascopy FX pair 1-min BID -> HOURLY. Args: SYMBOL SCALE.
  python _pull_fx_h1.py GBPUSD 100000   (5-digit)
  python _pull_fx_h1.py USDJPY 1000     (3-digit JPY)
Output: data/<symbol_lower>_h1.parquet (open_time UTC, close, quote_volume, symbol)."""
import lzma
import struct
import sys
import time

import httpx
import pandas as pd

SYM = sys.argv[1]
SCALE = float(sys.argv[2])
BASE = f"https://datafeed.dukascopy.com/datafeed/{SYM}"
OUT = rf"C:\Users\User\backtest_engine\backtest_engine2\data\{SYM.lower()}_h1.parquet"
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
    df["close"] = df["close"] / SCALE
    frames.append(df[["open_time", "close", "vol"]])
    ok += 1
    if ok % 300 == 0:
        print(f"{SYM} {ok} days (at {day.date()})", flush=True)
    time.sleep(0.03)

m1 = pd.concat(frames, ignore_index=True).set_index("open_time").sort_index()
h1 = m1.resample("1h").agg({"close": "last", "vol": "sum"}).dropna(subset=["close"]).reset_index()
h1 = h1.rename(columns={"vol": "quote_volume"})
h1["symbol"] = SYM
h1.to_parquet(OUT, index=False)
print(f"DONE {SYM}: {len(h1):,} hourly bars from {ok} days -> {OUT}")
print(h1.tail(2))

"""Probe Dukascopy datafeed: fetch one day of USA500 1-min BID candles, parse, sanity-check.
Candle bi5 = LZMA stream of 24-byte big-endian records. Month in URL is 0-indexed.
"""
import lzma
import struct

import httpx
import pandas as pd

# 2019-01-02: S&P 500 traded ~2450-2520.
URL = "https://datafeed.dukascopy.com/datafeed/USA500IDXUSD/2019/00/02/BID_candles_min_1.bi5"

r = httpx.get(URL, timeout=30)
print("status:", r.status_code, "bytes:", len(r.content))
raw = lzma.decompress(r.content)
n = len(raw) // 24
print("records:", n)

for fmt, label in [(">5if", "int-prices"), (">i4ff", "float-mix")]:
    t0, o, c, lo, hi, v = struct.unpack(fmt, raw[:24]) if fmt == ">5if" else struct.unpack(">iffff f".replace(" ", ""), raw[:24])
    print(f"{label}: t={t0} o={o} c={c} lo={lo} hi={hi} v={v}")

# Parse with int-price interpretation (typical for indices: price*1000)
recs = [struct.unpack(">5if", raw[i*24:(i+1)*24]) for i in range(n)]
df = pd.DataFrame(recs, columns=["sec", "open", "close", "low", "high", "vol"])
df["ts"] = pd.Timestamp("2019-01-02", tz="UTC") + pd.to_timedelta(df["sec"], unit="s")
for scale in (1.0, 10.0, 100.0, 1000.0):
    print(f"scale {scale}: first open {df['open'].iloc[0]/scale:.2f}, last close {df['close'].iloc[-1]/scale:.2f}")
print(df[["ts", "sec"]].head(3))
print(df[["ts", "sec"]].tail(3))

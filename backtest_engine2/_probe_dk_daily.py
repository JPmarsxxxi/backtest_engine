"""Probe Dukascopy daily-candle format: {INSTR}/{year}/BID_candles_day_1.bi5
(one LZMA file per year, 24-byte records >5if: sec-offset-from-year-start, OCLH ints, vol)."""
import lzma
import struct

import httpx
import pandas as pd

for instr, year in [("EURUSD", 2024), ("USA500IDXUSD", 2024), ("XAUUSD", 2024),
                    ("BRENTCMDUSD", 2024), ("USDJPY", 2024)]:
    url = f"https://datafeed.dukascopy.com/datafeed/{instr}/{year}/BID_candles_day_1.bi5"
    r = httpx.get(url, timeout=30)
    if r.status_code != 200 or not r.content:
        print(f"{instr}: HTTP {r.status_code}, {len(r.content)} bytes")
        continue
    raw = lzma.decompress(r.content)
    n = len(raw) // 24
    first = struct.unpack(">5if", raw[:24])
    last = struct.unpack(">5if", raw[(n - 1) * 24:n * 24])
    t0 = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(seconds=first[0])
    t1 = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(seconds=last[0])
    print(f"{instr}: {n} recs | first {t0} O={first[1]} | last {t1} C={last[2]} vol={last[5]:.1f}")

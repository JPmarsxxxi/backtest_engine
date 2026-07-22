"""#016k — one-time HISTORIC ASK pull, hourly, 2019-2022, for the edge_health mid rebuild.

The bid h1 history exists ({sym}_h1.parquet / fx_wide_m15 — BID closes, the #016 artifact source);
ask exists only as 1m from 2023 (x030 pulls). This fills the missing ask 2019-01-01 -> 2022-12-31 so
edge_health_016 can compute MIDs over the full window. Writes {sym}_ask_h1_hist.parquet.
Resume-safe (skips if output exists). One pair via argv for parallel jobs:
    python x016k_pull_ask_h1.py NZDUSD
"""
import lzma
import os
import struct
import sys
import time

import httpx
import pandas as pd

OUT = r"C:\Users\User\backtest_engine\backtest_engine2\data"
START, END = pd.Timestamp("2019-01-01"), pd.Timestamp("2022-12-31")
SCALE = {"USDJPY": 1e3}          # default 1e5


def pull(sym):
    f = os.path.join(OUT, f"{sym.lower()}_ask_h1_hist.parquet")
    if os.path.exists(f):
        print(f"{sym}: on disk, skip", flush=True)
        return
    scale = SCALE.get(sym, 1e5)
    base = f"https://datafeed.dukascopy.com/datafeed/{sym}"
    frames, ok, miss = [], 0, 0
    client = httpx.Client(timeout=30)
    for day in pd.date_range(START, END, freq="D"):
        if day.weekday() == 5:
            continue
        url = f"{base}/{day.year}/{day.month - 1:02d}/{day.day:02d}/ASK_candles_min_1.bi5"
        content = None
        for attempt in range(4):
            try:
                r = client.get(url)
                if r.status_code == 200 and len(r.content) > 0:
                    content = r.content
                break
            except Exception:
                time.sleep(2 * (attempt + 1))
        else:
            miss += 1
            continue
        if content is None:
            continue
        try:
            raw = lzma.decompress(content)
        except lzma.LZMAError:
            continue
        n = len(raw) // 24
        recs = [struct.unpack(">5if", raw[i * 24:(i + 1) * 24]) for i in range(n)]
        df = pd.DataFrame(recs, columns=["sec", "open", "close", "low", "high", "vol"])
        df = df[df["vol"] > 0]
        if df.empty:
            continue
        df["open_time"] = day.tz_localize("UTC") + pd.to_timedelta(df["sec"], unit="s")
        df["close"] = df["close"] / scale
        frames.append(df[["open_time", "close"]])
        ok += 1
        if ok % 250 == 0:
            print(f"{sym}: {ok} days (at {day.date()})", flush=True)
        time.sleep(0.03)
    client.close()
    if not frames:
        print(f"{sym}: NO DATA", flush=True)
        return
    m1 = pd.concat(frames, ignore_index=True).set_index("open_time").sort_index()
    h1 = m1.resample("1h").agg({"close": "last"}).dropna().reset_index()
    h1["symbol"] = sym
    h1.to_parquet(f, index=False)
    print(f"{sym}: DONE — {len(h1):,} hourly ask bars ({ok} days, {miss} failed)", flush=True)


if __name__ == "__main__":
    for sym in [s.upper() for s in sys.argv[1:]] or ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY"]:
        pull(sym)

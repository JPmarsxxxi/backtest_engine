"""Backfill 2026 (Jan-Jun) into data/ftmo_daily.parquet from Dukascopy HOURLY candles
(the day_1 year-file is only published after year end). Layout: {INSTR}/2026/{m:02d}/
BID_candles_hour_1.bi5, month dir 0-indexed, offsets = sec from month start.
Resample to daily close (UTC date, last active hour); active = vol>0 OR close changed."""
import lzma
import struct
import time

import httpx
import pandas as pd

from _pull_ftmo_daily import UNIVERSE  # dk name -> (ftmo name, scale)

OUT = r"C:\Users\User\backtest_engine\backtest_engine2\data\ftmo_daily.parquet"

client = httpx.Client(timeout=30)
frames = []
for dk, (ftmo, scale) in UNIVERSE.items():
    parts = []
    for m in range(6):  # dirs 00..05 = Jan..Jun 2026
        url = f"https://datafeed.dukascopy.com/datafeed/{dk}/2026/{m:02d}/BID_candles_hour_1.bi5"
        for attempt in range(3):
            try:
                r = client.get(url)
                break
            except Exception:
                time.sleep(2 * (attempt + 1))
        else:
            continue
        if r.status_code != 200 or len(r.content) == 0:
            continue
        try:
            raw = lzma.decompress(r.content)
        except lzma.LZMAError:
            continue
        n = len(raw) // 24
        recs = [struct.unpack(">5if", raw[i * 24:(i + 1) * 24]) for i in range(n)]
        df = pd.DataFrame(recs, columns=["sec", "open", "close", "low", "high", "vol"])
        df["ts"] = pd.Timestamp(f"2026-{m + 1:02d}-01") + pd.to_timedelta(df["sec"], unit="s")
        df = df[(df["vol"] > 0) | (df["close"].diff().fillna(0) != 0)]
        parts.append(df)
        time.sleep(0.05)
    if not parts:
        print(f"{dk}: no 2026 data", flush=True)
        continue
    h = pd.concat(parts, ignore_index=True)
    daily = h.groupby(h["ts"].dt.normalize()).agg(close=("close", "last"), vol=("vol", "sum"))
    daily = daily.reset_index().rename(columns={"ts": "date"})
    daily["close"] = daily["close"] / scale
    daily["symbol"] = ftmo
    daily["dk_name"] = dk
    frames.append(daily[["date", "close", "vol", "symbol", "dk_name"]])
    print(f"{dk} -> {ftmo}: {len(daily)} days 2026", flush=True)

new = pd.concat(frames, ignore_index=True).rename(columns={"vol": "volume"})
old = pd.read_parquet(OUT)
merged = (pd.concat([old, new], ignore_index=True)
          .drop_duplicates(["symbol", "date"], keep="last")
          .sort_values(["symbol", "date"]))
merged.to_parquet(OUT, index=False)
print(f"\nparquet: {len(old):,} -> {len(merged):,} rows")
print(merged.groupby("symbol")["date"].max().value_counts().sort_index().to_string())

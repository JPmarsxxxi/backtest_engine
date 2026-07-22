"""
x025c — extend FX H1 history back to 2003 for the #025 month-end power test.
Pulls Dukascopy 1-min BID -> hourly for 2003..2019 (the existing _h1 files cover 2019..2026),
writes <sym>_h1_hist.parquet, then merges hist+existing -> <sym>_h1_full.parquet (originals untouched).
"""
import lzma, struct, time
import httpx, pandas as pd

PAIRS = {"EURUSD": 100000.0, "GBPUSD": 100000.0, "USDJPY": 1000.0}
START = pd.Timestamp("2003-01-01")
END   = pd.Timestamp("2019-01-01")     # butts against existing data
DATA  = r"C:\Users\User\backtest_engine\backtest_engine2\data"

def pull(sym, scale):
    base = f"https://datafeed.dukascopy.com/datafeed/{sym}"
    frames = []; ok = empty = miss = 0
    client = httpx.Client(timeout=30)
    for day in pd.date_range(START, END, freq="D"):
        url = f"{base}/{day.year}/{day.month - 1:02d}/{day.day:02d}/BID_candles_min_1.bi5"
        for attempt in range(3):
            try:
                r = client.get(url); break
            except Exception:
                time.sleep(2 * (attempt + 1))
        else:
            miss += 1; continue
        if r.status_code != 200 or len(r.content) == 0:
            empty += 1; continue
        try:
            raw = lzma.decompress(r.content)
        except lzma.LZMAError:
            empty += 1; continue
        n = len(raw) // 24
        recs = [struct.unpack(">5if", raw[i*24:(i+1)*24]) for i in range(n)]
        df = pd.DataFrame(recs, columns=["sec", "open", "close", "low", "high", "vol"])
        df = df[df["vol"] > 0]
        if df.empty:
            empty += 1; continue
        df["open_time"] = day.tz_localize("UTC") + pd.to_timedelta(df["sec"], unit="s")
        df["close"] = df["close"] / scale
        frames.append(df[["open_time", "close", "vol"]])
        ok += 1
        if ok % 500 == 0:
            print(f"{sym} {ok} days (at {day.date()})", flush=True)
        time.sleep(0.02)
    m1 = pd.concat(frames, ignore_index=True).set_index("open_time").sort_index()
    h1 = m1.resample("1h").agg({"close": "last", "vol": "sum"}).dropna(subset=["close"]).reset_index()
    h1 = h1.rename(columns={"vol": "quote_volume"}); h1["symbol"] = sym
    hist_path = rf"{DATA}\{sym.lower()}_h1_hist.parquet"
    h1.to_parquet(hist_path, index=False)
    print(f"HIST {sym}: {len(h1):,} bars from {ok} days (empty {empty}, miss {miss}) -> {hist_path}", flush=True)

    # merge with existing 2019+ file
    exist = pd.read_parquet(rf"{DATA}\{sym.lower()}_h1.parquet")
    full = (pd.concat([h1, exist], ignore_index=True)
              .drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True))
    full_path = rf"{DATA}\{sym.lower()}_h1_full.parquet"
    full.to_parquet(full_path, index=False)
    print(f"FULL {sym}: {len(full):,} bars {full['open_time'].min()}..{full['open_time'].max()} -> {full_path}", flush=True)

for sym, scale in PAIRS.items():
    print(f"=== pulling {sym} 2003-2019 ===", flush=True)
    pull(sym, scale)
print("ALL DONE", flush=True)

"""Pull Dukascopy daily BID candles for the FTMO macro universe (#014 TSMOM).
One LZMA .bi5 per instrument-year, 24-byte records >5if (sec offset from year start,
O C L H ints scaled by 10^digits, float vol). Keep rows where vol>0 OR close changed
(us500 lesson: vol field can be zeroed server-side). Output: data/ftmo_daily.parquet
(date, close, volume, symbol=FTMO name, dk_name)."""
import lzma
import struct
import time

import httpx
import pandas as pd

OUT = r"C:\Users\User\backtest_engine\backtest_engine2\data\ftmo_daily.parquet"
YEARS = range(2000, 2027)

FX = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD",
      "EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURCAD", "EURNZD",
      "GBPJPY", "GBPCHF", "GBPAUD", "GBPCAD", "GBPNZD",
      "AUDJPY", "AUDCAD", "AUDCHF", "AUDNZD",
      "CADJPY", "CADCHF", "CHFJPY", "NZDJPY", "NZDCAD", "NZDCHF"]
# dukascopy name -> (ftmo name, scale)
UNIVERSE = {fx: (fx, 1e3 if fx.endswith("JPY") else 1e5) for fx in FX}
UNIVERSE.update({
    "USA500IDXUSD": ("US500", 1e3), "USATECHIDXUSD": ("US100", 1e3),
    "USA30IDXUSD": ("US30", 1e3), "DEUIDXEUR": ("GER40", 1e3),
    "EUSIDXEUR": ("EU50", 1e3), "FRAIDXEUR": ("FRA40", 1e3),
    "GBRIDXGBP": ("UK100", 1e3), "JPNIDXJPY": ("JP225", 1e3),
    "AUSIDXAUD": ("AUS200", 1e3), "HKGIDXHKD": ("HK50", 1e3),
    "ESPIDXEUR": ("SPA35", 1e3), "CHEIDXCHF": ("SWI20", 1e3),
    "XAUUSD": ("XAUUSD", 1e3), "XAGUSD": ("XAGUSD", 1e3),
    "BRENTCMDUSD": ("UKOIL", 1e3), "LIGHTCMDUSD": ("USOIL", 1e3),
    "GASCMDUSD": ("NATGAS", 1e3),
})

client = httpx.Client(timeout=30)
frames = []
for dk, (ftmo, scale) in UNIVERSE.items():
    got = 0
    for year in YEARS:
        url = f"https://datafeed.dukascopy.com/datafeed/{dk}/{year}/BID_candles_day_1.bi5"
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
        df["date"] = pd.Timestamp(f"{year}-01-01") + pd.to_timedelta(df["sec"], unit="s")
        active = (df["vol"] > 0) | (df["close"].diff().fillna(0) != 0)
        df = df[active]
        if df.empty:
            continue
        df["close"] = df["close"] / scale
        df["symbol"] = ftmo
        df["dk_name"] = dk
        frames.append(df[["date", "close", "vol", "symbol", "dk_name"]])
        got += 1
        time.sleep(0.05)
    print(f"{dk} -> {ftmo}: {got} year-files", flush=True)

out = pd.concat(frames, ignore_index=True).rename(columns={"vol": "volume"})
out.to_parquet(OUT, index=False)
print(f"\n{len(out):,} rows, {out['symbol'].nunique()} instruments -> {OUT}")
rng = out.groupby("symbol")["date"].agg(["min", "max", "count"])
print(rng.to_string())

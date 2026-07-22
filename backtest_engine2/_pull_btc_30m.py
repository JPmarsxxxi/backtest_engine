"""Pull BTCUSDT 30m klines (full history) from Binance REST -> data/btc_30m.parquet.

Long format matching binance_hourly.parquet: open_time, close, quote_volume, symbol.
Public endpoint, no key. ~160 calls at limit=1000.
"""
import time

import httpx
import pandas as pd

SYMBOL = "BTCUSDT"
INTERVAL = "30m"
START_MS = 1502928000000  # 2017-08-17 00:00 UTC (BTCUSDT listing)
OUT = r"C:\Users\User\backtest_engine\backtest_engine2\data\btc_30m.parquet"

BASES = [
    "https://api.binance.com",
    "https://data-api.binance.vision",
]


def pick_base() -> str:
    for base in BASES:
        try:
            r = httpx.get(f"{base}/api/v3/ping", timeout=10)
            if r.status_code == 200:
                return base
        except Exception as e:
            print(f"{base}: {e}")
    raise SystemExit("no reachable Binance endpoint")


def main():
    base = pick_base()
    print("using", base)
    url = f"{base}/api/v3/klines"
    rows = []
    start = START_MS
    calls = 0
    while True:
        r = httpx.get(url, params={
            "symbol": SYMBOL, "interval": INTERVAL,
            "startTime": start, "limit": 1000,
        }, timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        calls += 1
        if calls % 20 == 0:
            print(f"{calls} calls, {len(rows)} bars, at {pd.Timestamp(batch[-1][0], unit='ms', tz='UTC')}")
        if len(batch) < 1000:
            break
        start = batch[-1][0] + 1
        time.sleep(0.15)  # stay far under rate limits

    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "n_trades", "taker_base", "taker_quote", "ignore",
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close"] = df["close"].astype(float)
    df["quote_volume"] = df["quote_volume"].astype(float)
    df["symbol"] = SYMBOL
    out = df[["open_time", "close", "quote_volume", "symbol"]].drop_duplicates("open_time")
    out.to_parquet(OUT, index=False)
    print(f"saved {len(out)} bars -> {OUT}")
    print(out.head(2))
    print(out.tail(2))


if __name__ == "__main__":
    main()

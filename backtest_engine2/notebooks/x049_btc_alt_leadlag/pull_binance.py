"""#049 BTC->alt lead-lag, Cell 1 data pull: Binance USD-M perp 1m klines + bookTicker 1m mids.

Data-hygiene (finding-alphas/data-hygiene.md) decides the shape of this pull:
  * klines are TRADE PRINTS; a 1m close is the minute's LAST TRADE, so a quieter coin's close can be
    stale relative to BTC's -> fake lead-lag (non-synchronous trading, Lo-MacKinlay 1990). Kept for
    depth of history, with n_trades per minute for the staleness check.
  * bookTicker = best bid/ask (both sides, rule 1). Reduced here to the LAST quote in each minute
    (mid, half-spread bp, seconds since that quote) so the signal can run on mids.
  * perp klines, not spot, so both feeds come from the same venue (USD-M).
Provenance: "klines = trade prints from Binance USD-M, stamped at bar open (UTC), close knowable at
next bar open; bookTicker = best bid/ask from Binance USD-M, stamped at event_time (UTC ms)."

Sealed holdout: nothing after 2024-07-31 is pulled (BTC/ETH seal, extended to BNB/SOL).
Resumable: one parquet per (feed, symbol, month) under DATA/x049/; existing files are skipped.
Usage: python pull_binance.py [klines|book|all]     DATA dir overridable with env X049_DATA.
"""
import io
import os
import sys
import time
import zipfile

import httpx
import pandas as pd

DATA = os.environ.get("X049_DATA", r"C:\Users\User\backtest_engine\backtest_engine2\data")
OUT = os.path.join(DATA, "x049")
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
MONTHS = pd.period_range("2020-01", "2024-07", freq="M")   # seal: last month pulled is 2024-07
BASE = "https://data.binance.vision/data/futures/um"

K_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
          "n_trades", "taker_base", "taker_quote", "ignore"]
B_COLS = ["update_id", "bid", "bid_qty", "ask", "ask_qty", "transaction_time", "event_time"]

client = httpx.Client(timeout=120, follow_redirects=True)


def fetch(url):
    for attempt in range(4):
        try:
            r = client.get(url)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content
        except httpx.HTTPError:
            time.sleep(2 ** (attempt + 1))
    raise SystemExit(f"gave up on {url}")


def read_zip_csv(blob, cols):
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        raw = z.read(z.namelist()[0])
    first = raw[:200].decode(errors="ignore").split("\n")[0]
    header = 0 if not first[:1].isdigit() else None       # newer archive files carry a header row
    df = pd.read_csv(io.BytesIO(raw), header=header)
    df.columns = cols[:df.shape[1]]
    return df


def to_utc(ms):
    ms = pd.to_numeric(ms)
    unit = "us" if ms.iloc[0] > 1e14 else "ms"             # archive switched to microseconds in 2025
    return pd.to_datetime(ms, unit=unit, utc=True)


def pull_klines(sym, m):
    path = os.path.join(OUT, f"klines_{sym}_{m}.parquet")
    if os.path.exists(path):
        return "cached"
    blob = fetch(f"{BASE}/monthly/klines/{sym}/1m/{sym}-1m-{m}.zip")
    if blob is None:
        return "missing"
    df = read_zip_csv(blob, K_COLS)
    df["open_time"] = to_utc(df["open_time"])
    df = df[["open_time", "open", "high", "low", "close", "volume", "n_trades"]]
    df.to_parquet(path, index=False)
    return len(df)


def reduce_book(df):
    """Tick-level best bid/ask -> last quote per UTC minute (mid, half-spread bp, quote age s)."""
    df = df[(df.bid > 0) & (df.ask >= df.bid)].copy()
    df["t"] = to_utc(df["event_time"])
    df = df.sort_values("t")
    df["minute"] = df["t"].dt.floor("min")
    last = df.groupby("minute").tail(1).set_index("minute")
    out = pd.DataFrame(index=last.index)
    out["bid"], out["ask"] = last["bid"], last["ask"]
    out["mid"] = (last["bid"] + last["ask"]) / 2
    out["half_bp"] = (last["ask"] - last["bid"]) / 2 / out["mid"] * 1e4
    minute_end = out.index + pd.Timedelta(minutes=1)
    out["quote_age_s"] = (minute_end - last["t"]).dt.total_seconds()
    out["n_updates"] = df.groupby("minute").size()
    return out.reset_index().rename(columns={"minute": "open_time"})


def pull_book(sym, m):
    path = os.path.join(OUT, f"book_{sym}_{m}.parquet")
    if os.path.exists(path):
        return "cached"
    days = pd.date_range(m.start_time, m.end_time, freq="D")
    parts, missing = [], 0
    for d in days:
        blob = fetch(f"{BASE}/daily/bookTicker/{sym}/{sym}-bookTicker-{d:%Y-%m-%d}.zip")
        if blob is None:
            missing += 1
            continue
        parts.append(reduce_book(read_zip_csv(blob, B_COLS)))
    if not parts:
        return "missing"
    pd.concat(parts, ignore_index=True).to_parquet(path, index=False)
    return f"{len(days) - missing}/{len(days)} days"


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    os.makedirs(OUT, exist_ok=True)
    log = []
    for sym in SYMBOLS:
        for m in MONTHS:
            if what in ("klines", "all"):
                log.append(("klines", sym, str(m), pull_klines(sym, m)))
            if what in ("book", "all"):
                log.append(("book", sym, str(m), pull_book(sym, m)))
            print(*log[-1], flush=True)
    cov = pd.DataFrame(log, columns=["feed", "symbol", "month", "result"])
    cov.to_csv(os.path.join(OUT, "pull_coverage.csv"), index=False)
    print(cov.groupby(["feed", "symbol"])["result"].apply(lambda s: (s != "missing").sum()))


if __name__ == "__main__":
    main()

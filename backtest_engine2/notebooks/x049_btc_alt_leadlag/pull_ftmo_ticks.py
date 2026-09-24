"""#049 Cell 1 (Windows box only): FTMO MT5 tick history for the four cost-viable crypto CFDs.

Why: the signal would be read on Binance but traded on FTMO's own quotes, so the lag that matters
is FTMO-alt vs Binance-BTC. Only the broker's ticks can measure it. Both sides (bid AND ask) are
kept, per data-hygiene rule 1. Times are converted from the FTMO server clock to UTC using the live
offset (the #044 timestamp bug: FTMO server is UTC+2/+3, never assume).

Read-only: no orders. Output: data/x049/ftmo_ticks_<SYM>_<YYYY-MM>.parquet (time_utc, bid, ask),
plus data/x049/ftmo_tick_coverage.csv. The broker only serves what history it holds; the coverage
file records where it starts. Seal: nothing after 2024-07-31 is kept.
"""
import os
import time as _t
from datetime import datetime, timezone

import MetaTrader5 as mt5
import pandas as pd

OUT = r"C:\Users\User\backtest_engine\backtest_engine2\data\x049"
SYMBOLS = ["BTCUSD", "BNBUSD", "ETHUSD", "SOLUSD"]
MONTHS = pd.period_range("2020-01", "2024-07", freq="M")
SEAL = pd.Timestamp("2024-08-01", tz="UTC")


def main():
    if not mt5.initialize():
        raise SystemExit(f"MT5 initialize failed: {mt5.last_error()}")
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for sym in SYMBOLS:
        if mt5.symbol_info(sym) is None or not mt5.symbol_select(sym, True):
            rows.append((sym, None, "symbol not found"))
            continue
        tick = mt5.symbol_info_tick(sym)
        off_h = round((tick.time - _t.time()) / 3600)   # server clock offset vs UTC, measured live
        for m in MONTHS:
            path = os.path.join(OUT, f"ftmo_ticks_{sym}_{m}.parquet")
            if os.path.exists(path):
                rows.append((sym, str(m), "cached"))
                continue
            # copy_ticks_range takes server-clock datetimes (naive treated as UTC by the API)
            start = (m.start_time + pd.Timedelta(hours=off_h)).to_pydatetime().replace(tzinfo=timezone.utc)
            end = (m.end_time + pd.Timedelta(hours=off_h)).to_pydatetime().replace(tzinfo=timezone.utc)
            ticks = mt5.copy_ticks_range(sym, start, end, mt5.COPY_TICKS_INFO)
            if ticks is None or len(ticks) == 0:
                rows.append((sym, str(m), "no ticks"))
                continue
            df = pd.DataFrame(ticks)
            df = df[(df.bid > 0) & (df.ask >= df.bid)]
            df["time_utc"] = pd.to_datetime(df["time_msc"] - off_h * 3_600_000, unit="ms", utc=True)
            df = df[df.time_utc < SEAL][["time_utc", "bid", "ask"]]
            df.to_parquet(path, index=False)
            rows.append((sym, str(m), len(df)))
            print(sym, m, len(df), flush=True)
        print(f"{sym}: server offset {off_h:+d}h")
    mt5.shutdown()
    cov = pd.DataFrame(rows, columns=["symbol", "month", "result"])
    cov.to_csv(os.path.join(OUT, "ftmo_tick_coverage.csv"), index=False)
    print(cov[cov.result.apply(lambda r: isinstance(r, int))].groupby("symbol")["month"].agg(["min", "max", "count"]))


if __name__ == "__main__":
    main()

"""Measure 21:00/23:00 London half-spread for EURUSD, GBPUSD, USDJPY from the FTMO MT5 demo (ticks)."""
import sys
import time as _t
from datetime import datetime, timedelta

import MetaTrader5 as mt5
import pandas as pd

if not mt5.initialize():
    print(f"initialize failed: {mt5.last_error()} -> open & log into FTMO MT5"); sys.exit(1)
print(f"Connected: {mt5.account_info().server}\n")

def measure(want):
    sym = want
    if mt5.symbol_info(sym) is None:
        cand = [s.name for s in mt5.symbols_get() if s.name.upper().startswith(want)]
        if not cand:
            return None
        sym = cand[0]
    mt5.symbol_select(sym, True)
    srv = mt5.symbol_info_tick(sym).time
    off_h = round((srv - _t.time()) / 3600)
    server_now = datetime.utcfromtimestamp(srv)
    ticks = mt5.copy_ticks_range(sym, server_now - timedelta(days=12), server_now + timedelta(hours=2),
                                 mt5.COPY_TICKS_INFO)
    if ticks is None or len(ticks) == 0:
        return None
    df = pd.DataFrame(ticks)
    df = df[(df.bid > 0) & (df.ask > 0) & (df.ask >= df.bid)].copy()
    utc = pd.to_datetime(df["time"] - off_h * 3600, unit="s", utc=True)
    ldn = utc.dt.tz_convert("Europe/London")
    df["hr"], df["dow"] = ldn.dt.hour, ldn.dt.dayofweek
    df = df[df.dow < 5]
    df["half_bp"] = (df.ask - df.bid) / ((df.ask + df.bid) / 2) * 1e4 / 2
    out = {}
    for hh in (21, 22, 23):
        x = df[df.hr == hh]["half_bp"]
        out[hh] = (x.median(), len(x))
    ref = df[df.hr.isin([13, 14, 15])]["half_bp"].median()
    return sym, out, ref

print(f"{'pair':>8} {'21:00 half':>11} {'22:00(roll)':>12} {'23:00 half':>11} {'13-15 ref':>10}")
for want in ("EURUSD", "GBPUSD", "USDJPY"):
    res = measure(want)
    if res is None:
        print(f"{want:>8}  (no data)"); continue
    sym, o, ref = res
    print(f"{sym:>8} {o[21][0]:>9.3f}bp {o[22][0]:>10.3f}bp {o[23][0]:>9.3f}bp {ref:>8.3f}bp  "
          f"(n21={o[21][1]}, n23={o[23][1]})")
mt5.shutdown()

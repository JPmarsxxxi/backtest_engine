"""Measure EURUSD half-spread BY LONDON HOUR from the running FTMO MT5 terminal (tick data).
Resolves the #016 gate: is the 21:00-23:00 London spread <=~0.25bp half (lead alive) or >=0.3bp (dead)?
Attaches to a RUNNING, LOGGED-IN MT5 terminal (same as _pull_ftmo_specs.py)."""
import sys
import time as _t
from datetime import datetime, timedelta, timezone

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

if not mt5.initialize():
    print(f"initialize() failed: {mt5.last_error()}")
    print("-> Open the FTMO MT5 terminal and log into the demo, then re-run.")
    sys.exit(1)

acct = mt5.account_info()
print(f"Connected: {acct.server} | login {acct.login} | {acct.company}\n")

# resolve the EURUSD symbol name (handle broker suffixes)
SYM = "EURUSD"
if mt5.symbol_info(SYM) is None:
    cand = [s.name for s in mt5.symbols_get() if s.name.upper().startswith("EURUSD")]
    if not cand:
        print("No EURUSD-like symbol found."); mt5.shutdown(); sys.exit(1)
    SYM = cand[0]
mt5.symbol_select(SYM, True)
print(f"symbol: {SYM}")

# infer server->UTC offset (MT5 tick.time is server wall-clock as epoch)
srv = mt5.symbol_info_tick(SYM).time
off_h = round((srv - _t.time()) / 3600)
print(f"server UTC offset: {off_h:+d}h")

# MT5 wants NAIVE datetimes in SERVER time for the range; tick.time comes back as server epoch.
server_now = datetime.utcfromtimestamp(srv)              # server wall-clock as naive dt
date_from = server_now - timedelta(days=12)
date_to = server_now + timedelta(hours=2)
ticks = mt5.copy_ticks_range(SYM, date_from, date_to, mt5.COPY_TICKS_INFO)
mt5.shutdown()
if ticks is None or len(ticks) == 0:
    print(f"no ticks returned (last_error {mt5.last_error()})"); sys.exit(1)

df = pd.DataFrame(ticks)
print(f"raw ticks {len(df)} | cols {list(df.columns)} | time[0]={df['time'].iloc[0]} -> "
      f"{pd.to_datetime(df['time'].iloc[0], unit='s')}")
df = df[(df["bid"] > 0) & (df["ask"] > 0) & (df["ask"] >= df["bid"])].copy()
df["utc"] = pd.to_datetime(df["time"] - off_h * 3600, unit="s", utc=True)
ldn = df["utc"].dt.tz_convert("Europe/London")
df["hr"] = ldn.dt.hour
df["dow"] = ldn.dt.dayofweek
df = df[df["dow"] < 5]                                    # weekdays only
df["half_bp"] = (df["ask"] - df["bid"]) / ((df["ask"] + df["bid"]) / 2) * 1e4 / 2

print(f"\n{len(df):,} weekday ticks | {df.utc.min().date()} -> {df.utc.max().date()}")
g = df.groupby("hr")["half_bp"].agg(["median", "mean", "count"])
print("\nLondon hr | median half_bp | mean | nticks")
for hh, r in g.iterrows():
    star = " <- TRADE HOUR" if hh in (21, 22, 23) else (" (liquid ref)" if hh in (13, 14, 15) else "")
    print(f"   {hh:02d}:00   {r['median']:.3f}        {r['mean']:.3f}   {int(r['count']):>8,}{star}")

trade = df[df.hr.isin([21, 22, 23])]["half_bp"]
ref = df[df.hr.isin([13, 14, 15])]["half_bp"]
print(f"\n=== #016 GATE ===")
print(f"21-23:00 London half-spread: median {trade.median():.3f}bp | mean {trade.mean():.3f}bp | "
      f"p75 {trade.quantile(.75):.3f} | p90 {trade.quantile(.90):.3f}")
print(f"13-15:00 (liquid ref)      : median {ref.median():.3f}bp")
bf = trade.median()
print(f"\nstrategy break-even ~0.30bp half | trade-hour median {bf:.3f}bp -> "
      f"{'ALIVE (net Sh ~1.2 at this spread)' if bf <= 0.25 else 'MARGINAL' if bf <= 0.32 else 'DEAD (spread eats it)'}")

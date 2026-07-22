"""#034 (LEAD 2026-07-12) — FTMO single-stock CFD spread profile, the pre-committed DECISIVE FIRST
STEP for the swap-dodged cross-sectional equity book (alpha log LEAD entry + data-hygiene rule 1:
measure the toll per-instrument per-time-of-day from the broker's own feed BEFORE any backtest).

Read-only (ticks only, no orders). For each Equities-I (US) CFD symbol: pull ~12 days of INFO ticks,
compute half-spread (bp of mid) by NEW YORK hour, plus quote coverage per hour. The two numbers that
decide the venue:
  * cash-session spread (NY 09:30-16:00)  -> the intraday version's toll
  * 18:00-NY spread + does it QUOTE at all then -> the overnight/swap-dodge version's toll (if the
    instrument doesn't quote after the 17:00 snapshot, the overnight version is dead on arrival)
Output: data/ftmo_equity_spread_profile.parquet (long: symbol, ny_hour, median/p75 half bp, n_ticks,
days_seen, run_date) + console summary. Re-runnable (overwrites; it is a dated snapshot).
"""
import os
import sys
import time as _t
from datetime import datetime, timedelta, timezone

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

D = r"C:\Users\User\backtest_engine\backtest_engine2\data"
OUT = os.path.join(D, "ftmo_equity_spread_profile.parquet")
LOGFILE = os.path.join(D, "x034_profile.out")
if sys.stdout is None:
    sys.stdout = sys.stderr = open(LOGFILE, "a", encoding="utf-8")

WINDOW_DAYS = 12
CASH_HOURS = list(range(10, 16))          # full cash hours NY (skip the 9:30 partial for the summary)
KEY_HOURS = [9, 16, 17, 18, 19]           # open partial, close, snapshot, dodge-entry, late


def us_equity_symbols():
    spec = pd.read_parquet(os.path.join(D, "ftmo_specs.parquet"))
    return spec.loc[spec.path.str.startswith("Equities I CFD"), "symbol"].tolist()


def server_offset_h():
    """Broker-clock offset vs UTC, measured ONCE from a 24/5 instrument whose last tick is always
    fresh (EURUSD). Do NOT infer it per-stock: a closed stock's last tick is hours old and the
    offset comes out garbage (the bug that emptied the first run of this script)."""
    mt5.symbol_select("EURUSD", True)
    t = mt5.symbol_info_tick("EURUSD")
    return round((t.time - _t.time()) / 3600)


def profile(sym, off_h):
    if mt5.symbol_info(sym) is None:
        return None
    mt5.symbol_select(sym, True)
    tick = None
    for _ in range(4):                     # fresh symbols need a beat before ticks stream
        tick = mt5.symbol_info_tick(sym)
        if tick is not None and tick.time > 0:
            break
        _t.sleep(0.4)
    if tick is not None and tick.time > 0:
        server_now = datetime.fromtimestamp(tick.time, timezone.utc).replace(tzinfo=None)
    else:                                  # closed/quiet symbol: derive server time from wallclock
        server_now = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=off_h)
    ticks = mt5.copy_ticks_range(sym, server_now - timedelta(days=WINDOW_DAYS),
                                 server_now + timedelta(hours=2), mt5.COPY_TICKS_INFO)
    if ticks is None or len(ticks) == 0:
        return None
    df = pd.DataFrame(ticks)
    df = df[(df.bid > 0) & (df.ask > 0) & (df.ask >= df.bid)].copy()
    if df.empty:
        return None
    utc = pd.to_datetime(df["time"] - off_h * 3600, unit="s", utc=True)
    ny = utc.dt.tz_convert("America/New_York")
    df["hr"], df["dow"], df["day"] = ny.dt.hour, ny.dt.dayofweek, ny.dt.date
    df = df[df.dow < 5]
    mid = (df.ask + df.bid) / 2
    df["half_bp"] = (df.ask - df.bid) / mid / 2 * 1e4
    g = df.groupby("hr")["half_bp"]
    out = pd.DataFrame({"median_half_bp": g.median(), "p75_half_bp": g.quantile(0.75),
                        "n_ticks": g.size()})
    out["days_seen"] = df.groupby("hr")["day"].nunique()
    out = out.reset_index().rename(columns={"hr": "ny_hour"})
    out["symbol"] = sym
    return out


def run():
    if not mt5.initialize():
        print(f"x034: MT5 initialize failed {mt5.last_error()}")
        return 1
    acct = mt5.account_info()
    print(f"\n{datetime.now():%Y-%m-%d %H:%M} x034 equity spread profile on {acct.server} "
          f"(trailing {WINDOW_DAYS}d of ticks, hours in NEW YORK time)")
    off_h = server_offset_h()
    print(f"broker clock offset vs UTC: {off_h:+d}h (from EURUSD)")
    frames, missing = [], []
    for sym in us_equity_symbols():
        r = profile(sym, off_h)
        if r is None:
            missing.append(sym)
            continue
        frames.append(r)
    mt5.shutdown()
    if not frames:
        print("no tick data for any symbol — is the terminal logged in?")
        return 1
    prof = pd.concat(frames, ignore_index=True)
    prof["run_date"] = f"{datetime.now():%Y-%m-%d}"
    prof.to_parquet(OUT, index=False)

    # summary: per symbol — cash-session median vs the swap-dodge window
    rows = []
    for sym, g in prof.groupby("symbol"):
        g = g.set_index("ny_hour")
        cash = g.loc[g.index.isin(CASH_HOURS), "median_half_bp"].median()
        h18 = g.loc[18, "median_half_bp"] if 18 in g.index else np.nan
        n18 = int(g.loc[18, "n_ticks"]) if 18 in g.index else 0
        d18 = int(g.loc[18, "days_seen"]) if 18 in g.index else 0
        rows.append({"symbol": sym, "cash_half_bp": cash, "h18_half_bp": h18,
                     "h18_ticks": n18, "h18_days": d18})
    s = pd.DataFrame(rows).sort_values("cash_half_bp")
    print(f"\n{'symbol':>7} {'cash half bp':>13} {'18:00NY half':>13} {'18h ticks':>10} {'18h days':>9}")
    for _, r in s.iterrows():
        print(f"{r.symbol:>7} {r.cash_half_bp:>13.2f} "
              f"{(f'{r.h18_half_bp:.2f}' if pd.notna(r.h18_half_bp) else '—'):>13} "
              f"{r.h18_ticks:>10,} {r.h18_days:>9}")
    quoted18 = int((s.h18_days >= 3).sum())
    print(f"\nsymbols quoting at 18:00 NY on >=3 of last {WINDOW_DAYS} days: {quoted18}/{len(s)}")
    print(f"cash-session median half-spread across book: {s.cash_half_bp.median():.2f}bp "
          f"(P25 {s.cash_half_bp.quantile(0.25):.2f} / P75 {s.cash_half_bp.quantile(0.75):.2f})")
    if missing:
        print(f"no data: {missing}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(run())

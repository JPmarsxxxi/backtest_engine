"""#016 v3 — monthly ENTRY-CAP recalibration from the broker's own tick feed (read-only, no orders).

The live spread gate compares the current half-spread to a per-pair cap. Those caps describe a FACT
about the broker (its normal spread at our two entry hours), so they must track the broker: this
script re-measures them from recent MT5 ticks and writes data/fx_entry_caps_016.json, which
fx_seasonal_live_v3.py loads at each tick (falling back to its hardcoded defaults if absent/insane).

Rule (same one used for the original hand calibration, 2026-07-12): cap = 2.2 x the median half-spread
observed during the 21:00 and 23:00 London hours over the trailing window, rounded UP to 0.05bp.
Guards: enough ticks, cap clamped to [0.10, 1.50]bp, and never moves more than 2x from the current cap
in one recalibration (a bigger jump means something changed at the broker -> human should look).

Run monthly (scheduled) or manually:  python recal_caps_016.py
"""
import json
import math
import os
import sys
import time as _t
from datetime import datetime, timedelta, timezone

D = r"C:\Users\User\backtest_engine\backtest_engine2\data"
OUT = os.path.join(D, "fx_entry_caps_016.json")
LOGFILE = os.path.join(D, "recal_caps_016.out")

# pythonw has no console; print() would crash on stdout=None (see ops runbook)
if sys.stdout is None:
    sys.stdout = sys.stderr = open(LOGFILE, "a")

import MetaTrader5 as mt5
import pandas as pd

PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY"]
# current live defaults (also the change-guard baseline when no previous JSON exists)
FALLBACK = {"EURUSD": 0.20, "GBPUSD": 0.45, "AUDUSD": 0.50, "NZDUSD": 0.90, "USDJPY": 0.35}
ENTRY_HOURS = (21, 23)          # London
WINDOW_DAYS = 21                # trailing tick window
MULT = 2.2                      # cap = MULT x median entry-hour half-spread
CLAMP = (0.10, 1.50)            # bp, absolute sanity bounds
MIN_TICKS = 2000                # per (pair, hour) or the measurement is refused
MAX_STEP = 2.0                  # new cap within [old/2, old*2] or it is clamped + flagged


def measure(sym):
    if mt5.symbol_info(sym) is None:
        return None
    mt5.symbol_select(sym, True)
    tick = mt5.symbol_info_tick(sym)
    if tick is None:
        return None
    off_h = round((tick.time - _t.time()) / 3600)        # server-clock offset (FTMO server is UTC+3)
    server_now = datetime.fromtimestamp(tick.time, timezone.utc).replace(tzinfo=None)
    ticks = mt5.copy_ticks_range(sym, server_now - timedelta(days=WINDOW_DAYS),
                                 server_now + timedelta(hours=2), mt5.COPY_TICKS_INFO)
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
    for hh in ENTRY_HOURS:
        x = df[df.hr == hh]["half_bp"]
        out[hh] = {"median": float(x.median()) if len(x) else float("nan"), "n": int(len(x))}
    return out


def run():
    if not mt5.initialize():
        print(f"{datetime.now():%Y-%m-%d %H:%M} recal_caps: MT5 initialize failed {mt5.last_error()}")
        return 1
    acct = mt5.account_info()
    print(f"\n{datetime.now():%Y-%m-%d %H:%M} recal_caps_016 on {acct.server} login {acct.login} "
          f"(window {WINDOW_DAYS}d, rule {MULT}x median, hours {ENTRY_HOURS} London)")

    prev = dict(FALLBACK)
    if os.path.exists(OUT):
        try:
            with open(OUT) as f:
                prev.update({k: float(v) for k, v in json.load(f)["caps"].items()})
        except Exception:
            pass

    caps, detail, flags = {}, {}, []
    for sym in PAIRS:
        m = measure(sym)
        old = prev[sym]
        if m is None or any(m[h]["n"] < MIN_TICKS or math.isnan(m[h]["median"]) for h in ENTRY_HOURS):
            caps[sym] = old
            flags.append(f"{sym}: measurement refused (no/too few ticks) -> kept {old:.2f}")
            detail[sym] = {"kept_previous": True, "raw": m}
            continue
        med = max(m[h]["median"] for h in ENTRY_HOURS)      # cap must admit BOTH entry hours
        raw = MULT * med
        cap = math.ceil(raw / 0.05) * 0.05                  # round UP to 0.05bp
        cap = min(max(cap, CLAMP[0]), CLAMP[1])
        if cap > old * MAX_STEP or cap < old / MAX_STEP:
            flags.append(f"{sym}: measured cap {cap:.2f} vs current {old:.2f} moved >{MAX_STEP}x "
                         f"-> clamped; broker spread regime changed, LOOK AT THIS")
            cap = min(max(cap, old / MAX_STEP), old * MAX_STEP)
            cap = round(cap, 2)
        caps[sym] = round(cap, 2)
        detail[sym] = {"median_by_hour": {str(h): round(m[h]["median"], 4) for h in ENTRY_HOURS},
                       "n_by_hour": {str(h): m[h]["n"] for h in ENTRY_HOURS},
                       "old_cap": old, "new_cap": caps[sym]}
        arrow = "=" if abs(caps[sym] - old) < 1e-9 else ("^" if caps[sym] > old else "v")
        print(f"   {sym}: median21 {m[21]['median']:.3f} / median23 {m[23]['median']:.3f} bp "
              f"(n {m[21]['n']:,}/{m[23]['n']:,}) -> cap {old:.2f} {arrow} {caps[sym]:.2f}")
    mt5.shutdown()

    payload = {"generated": datetime.now(timezone.utc).isoformat(),
               "window_days": WINDOW_DAYS, "rule": f"{MULT}x median entry-hour half, ceil 0.05",
               "caps": caps, "detail": detail, "flags": flags}
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"   wrote {OUT}")
    for fl in flags:
        print(f"   FLAG: {fl}")
    return 0


if __name__ == "__main__":
    sys.exit(run())

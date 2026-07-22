"""One-shot 'tick' runner for the live FTMO forward tests — the crash-proof replacement for the
persistent daemons (fx_seasonal_live / fx_seasonal_live_v2 / overnight_live).

WHY: the daemons were `while True: step(); sleep(15)` processes started ONCE at logon. Any crash,
reboot, or (the real killer) the PC going to sleep left them dead until the next interactive logon —
they died silently three times. This script runs the SAME, already-hardened `step()` exactly once,
persisting the tiny bit of cross-call state (which session is open, entry fills) to a JSON sidecar.
Windows Task Scheduler then fires it every minute, so liveness is owned by the OS, not a fragile
long-lived process: a missed/crashed tick just means the next minute's tick runs. No trading logic
changes — it imports the strategy module and calls its `resolve()` / `step()` / `write_status()`.

  python strat_tick.py --kind fx        --module fx_seasonal_live     --selftest   # validate, NO orders
  python strat_tick.py --kind fx        --module fx_seasonal_live_v2  --selftest
  python strat_tick.py --kind overnight --module overnight_live       --selftest
  python strat_tick.py --kind fx        --module fx_seasonal_live                   # one LIVE tick

State lives beside the logs: data/<module>_tick_state.json. Delete it to reset a strat's memory.
"""
import argparse
import importlib
import json
import os
import sys
import time
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import MetaTrader5 as mt5

DATA = r"C:\Users\User\backtest_engine\backtest_engine2\data"
LONDON = ZoneInfo("Europe/London")
NY = ZoneInfo("America/New_York")


def _default(o):
    """JSON encoder for the few non-native types that leak into state (numpy scalars, dates)."""
    import numpy as np
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    raise TypeError(f"not JSON-serializable: {type(o)}")


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, default=_default)
    os.replace(tmp, path)   # atomic — a crash mid-write never corrupts the state file


def bounded_connect(attempts=3, sleep_s=4):
    """Connect to the running MT5 terminal, but give up after a few tries (a one-shot must EXIT, not
    retry forever like the daemon did). If the terminal is down this minute, the next tick retries."""
    for _ in range(attempts):
        try:
            if mt5.initialize() and mt5.account_info() is not None:
                return mt5.account_info()
        except Exception:
            pass
        time.sleep(sleep_s)
    return None


def run_fx(mod, state_path):
    """FX v1/v2: step(syms, state, openpos). state keyed by (date, session); openpos by (session, pair)."""
    raw = load_json(state_path) or {}
    state = {}
    for k, v in raw.get("state", {}).items():
        ds, sid = k.split("|", 1)
        state[(date.fromisoformat(ds), sid)] = v
    openpos = {}
    for k, v in raw.get("openpos", {}).items():
        sid, w = k.split("|", 1)
        openpos[(sid, w)] = v

    syms = mod.resolve()
    if any(v is None for v in syms.values()):
        print("missing symbol — skipping tick"); return
    mod.heartbeat()
    mod.step(syms, state, openpos)

    today = datetime.now(timezone.utc).date()
    state = {k: v for k, v in state.items() if (today - k[0]).days <= 3}   # prune old sessions
    save_json(state_path, {
        "state": {f"{k[0].isoformat()}|{k[1]}": v for k, v in state.items()},
        "openpos": {f"{k[0]}|{k[1]}": v for k, v in openpos.items()},
    })


def run_overnight(mod, state_path):
    """#023 overnight: step(sym, st) where st = {'open': dict|None, 'entered_on': date|None}."""
    raw = load_json(state_path) or {}
    st = {
        "open": raw.get("open"),
        "entered_on": date.fromisoformat(raw["entered_on"]) if raw.get("entered_on") else None,
    }
    sym = mod.resolve()
    if sym is None:
        print("USA500 symbol not found — skipping tick"); return
    mod.heartbeat()
    mod.step(sym, st)
    save_json(state_path, {
        "open": st["open"],
        "entered_on": st["entered_on"].isoformat() if st.get("entered_on") else None,
    })


def selftest(mod, kind):
    """Prove the plumbing (connect, symbols, clock, quotes) without calling step() — no orders, no state."""
    tz = LONDON if kind == "fx" else NY
    now = datetime.now(timezone.utc).astimezone(tz)
    print(f"now: {now:%Y-%m-%d %H:%M:%S %Z} (weekday {now.weekday()})")
    if kind == "fx":
        syms = mod.resolve()
        print("symbols:", syms)
        for w, sym in syms.items():
            if sym:
                bid, ask, mid, half = mod.quote(sym)
                print(f"   {sym}: half-spread {half:.3f}bp")
        print("sessions:", {s: (c["enter"], c["exit"], c["side"]) for s, c in mod.SESSIONS.items()})
    else:
        sym = mod.resolve()
        print("symbol:", sym)
        if sym:
            on, price, sma = mod.regime_on(sym)
            print(f"   regime: {on} (px {price} vs 200d {sma})")
    print("plumbing OK — connection, symbols, clock, quotes all working. No orders placed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["fx", "overnight"])
    ap.add_argument("--module", required=True)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    # Run under pythonw.exe (no console window -> no per-minute terminal flash). pythonw has no stdout,
    # so redirect the strategy's print()s to a file BEFORE calling step() (else print() crashes on None).
    if not args.selftest:
        try:
            sys.stdout = sys.stderr = open(os.path.join(DATA, f"{args.module}_tick.out"), "a", buffering=1)
        except Exception:
            pass

    mod = importlib.import_module(args.module)
    if args.dry:
        mod.DRY = True

    acct = bounded_connect()
    if acct is None:
        print("could not connect to MT5 this tick — terminal down / logged out. Next tick will retry.")
        try:
            mod.write_status("disconnected", err=mt5.last_error())
        except Exception:
            pass
        return
    mod.write_status("connected", acct)

    state_path = os.path.join(DATA, f"{args.module}_tick_state.json")
    try:
        if args.selftest:
            selftest(mod, args.kind)
        elif args.kind == "fx":
            run_fx(mod, state_path)
        else:
            run_overnight(mod, state_path)
    except Exception as e:
        print(f"tick error: {e}", flush=True)
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()

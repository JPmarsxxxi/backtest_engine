"""#016 live forward-test — VARIANT v3: v2 spread-gate execution + ENHANCED 5-pair book {EUR/GBP/AUD/NZD/USDJPY}. Parallel w/ v1,v2.

Hypothesis under test: v1's loss is an entry-timing artifact — it fires at HH:00:05, inside the
top-of-hour / post-rollover spread flare, paying 3-5x the spread the backtest assumed. v2 changes
ONLY the entry execution (exit is identical to v1, so any difference is attributable to entry):

  * At each entry time it POLLS the live spread instead of firing instantly.
  * It enters each pair the moment its half-spread is favourable (<= MAX_ENTRY_HALF[pair]) — i.e. it
    waits out the flare.
  * If a pair never gets a favourable spread within ENTRY_POLL_SECONDS, it SKIPS that leg and logs a
    skip row (so we measure: does 23:00 settle into a good spread = fixable, or stay wide = structural?).

Fully isolated from v1: own MAGIC, own log/lock/status files — positions, flatten, and single-instance
locks never collide. Both place real demo orders on the same account.

  python fx_seasonal_live_v2.py --dry --once   # validation: connect, read ticks, print intent, NO orders
  python fx_seasonal_live_v2.py                 # LIVE daemon on the demo (needs AutoTrading enabled)
"""
import os
import sys
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import MetaTrader5 as mt5
import pandas as pd

LONDON = ZoneInfo("Europe/London")
VARIANT = "v3"
LOCK = r"C:\Users\User\backtest_engine\backtest_engine2\data\.fx_daemon_v3.lock"
ALERTLOG = r"C:\Users\User\backtest_engine\backtest_engine2\data\daemon_alert.log"   # shared, v2-tagged
STATUS = r"C:\Users\User\backtest_engine\backtest_engine2\data\daemon_status_v3.txt"
LOG = r"C:\Users\User\backtest_engine\backtest_engine2\data\fx_seasonal_live_log_v3.parquet"

MAGIC = 160163                                     # v1 160160/v2 160161 — keep books separate
LOT = 0.10
# Book cut to the 3 legs where TRUE-mid edge clears the FTMO toll (2026-07-16, user-approved;
# alpha log that date): GBPUSD dropped (real +0.26bp/night edge < ~0.38bp toll), USDJPY dropped
# (23:00 edge was ~87% bid-artifact, trailing-1y mid NEGATIVE — edge_health v2 FLIPPED verdict).
# v2 (EUR/GBP/JPY) intentionally KEPT RUNNING as the control group on the dropped legs.
WANT = ["EURUSD", "AUDUSD", "NZDUSD"]
SESSIONS = {
    # Session A (21:00 SELL -> 21:50) REMOVED 2026-07-15 (user-approved): the hour-21 "fall" was a
    # bid-only-data artifact — in MID space the price RISES 21:00->21:50 (x016j; alpha log 2026-07-15
    # cont.). The short leg was a structural leak: backwards bet + spread toll every night.
    #   "A": {"enter": (21, 0), "exit": (21, 50), "side": "SELL"},
    "B": {"enter": (23, 0), "exit": (23, 58), "side": "BUY"},
}

# --- v2 entry gate (the whole experiment lives here; tune these, they're logged so we can recalibrate) ---
ENTRY_POLL_SECONDS = 90        # grace window to wait out the flare before giving up on a leg
POLL_EVERY = 3                 # seconds between spread checks
# Per-pair max acceptable ENTRY half-spread (bp). Set between v1's calm 21:00 entries (~0.10/0.26/0.24,
# keep) and its 23:00 flares (~0.65/1.73/0.85, wait or skip). Backtest-assumed half: EUR .087 / GBP .187
# / JPY .156-.125 — these caps allow ~2x that, refusing the multi-bp flares that erase a ~1.3bp edge.
MAX_ENTRY_HALF = {"EURUSD": 0.20, "GBPUSD": 0.45, "AUDUSD": 0.50, "NZDUSD": 0.90, "USDJPY": 0.35}

# Caps above are the FALLBACK. Live values come from data/fx_entry_caps_016.json, re-measured monthly
# from the broker's ticks by recal_caps_016.py (caps describe the broker's normal spread, a fact that
# drifts — the SIGNAL constants stay frozen, see alpha log 2026-07-15). Sanity-gated: known pairs,
# 0.05-1.50bp, all 5 present; a stale file (>40d = recal task dead) still loads but warns.
CAPS_JSON = r"C:\Users\User\backtest_engine\backtest_engine2\data\fx_entry_caps_016.json"


def _load_caps():
    import json
    try:
        with open(CAPS_JSON) as f:
            j = json.load(f)
        caps = {k: float(v) for k, v in j["caps"].items()}
    except FileNotFoundError:
        return
    except Exception as e:
        print(f"caps json unreadable ({e}) — using hardcoded fallback caps")
        return
    sane = {k: v for k, v in caps.items() if k in MAX_ENTRY_HALF and 0.05 <= v <= 1.50}
    if len(sane) != len(MAX_ENTRY_HALF):
        print("caps json failed sanity check — using hardcoded fallback caps")
        return
    MAX_ENTRY_HALF.update(sane)
    age_d = (datetime.now(timezone.utc) - datetime.fromisoformat(j["generated"])).days
    if age_d > 40:
        print(f"WARNING: entry caps are {age_d}d old — recal_caps_016 task looks dead")


_load_caps()

DRY = "--dry" in sys.argv
ONCE = "--once" in sys.argv


def alert(msg):
    line = f"{datetime.now(timezone.utc).astimezone(LONDON):%Y-%m-%d %H:%M:%S} ALERT [{VARIANT}]: {msg}"
    print(">>> " + line, flush=True)
    try:
        with open(ALERTLOG, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def write_status(state, acct=None, err=None):
    try:
        now = datetime.now(timezone.utc).astimezone(LONDON)
        lines = [f"variant={VARIANT}", f"time={now:%Y-%m-%d %H:%M:%S %Z}", f"state={state}", f"pid={os.getpid()}"]
        if acct is not None:
            lines += [f"login={acct.login}", f"server={acct.server}", f"balance={acct.balance:.2f}"]
        if err is not None:
            lines.append(f"last_error={err}")
        with open(STATUS, "w") as f:
            f.write("\n".join(lines) + "\n")
    except Exception:
        pass


def lock_is_fresh():
    try:
        return (time.time() - float(open(LOCK).read().strip())) < 60
    except Exception:
        return False


def heartbeat():
    try:
        with open(LOCK, "w") as f:
            f.write(str(time.time()))
    except Exception:
        pass


def connect(max_attempts=None):
    attempt = 0
    while max_attempts is None or attempt < max_attempts:
        if mt5.initialize() and mt5.account_info() is not None:
            a = mt5.account_info()
            print(f"{datetime.now(timezone.utc).astimezone(LONDON):%Y-%m-%d %H:%M} [{VARIANT}] Connected: "
                  f"{a.server} | login {a.login} | balance {a.balance:,.0f} {a.currency} | "
                  f"{'DRY-RUN (no orders)' if DRY else 'LIVE ORDERS'}", flush=True)
            write_status("connected", a)
            return
        err = mt5.last_error()
        if attempt % 6 == 0:
            print(f"[{VARIANT}] waiting for MT5 terminal... ({err})", flush=True)
            write_status("disconnected", err=err)
        if attempt > 0 and attempt % 180 == 0:
            alert(f"MT5 not connected for ~{attempt * 10 // 60} min ({err}). "
                  f"Log the FTMO demo back in / renew the Free Trial / enable AutoTrading.")
        attempt += 1
        time.sleep(10)
    print(f"[{VARIANT}] could not connect — is the FTMO MT5 terminal running & logged in?")
    write_status("disconnected", err=mt5.last_error())
    sys.exit(1)


def resolve():
    out = {}
    for w in WANT:
        if mt5.symbol_info(w) is not None:
            out[w] = w
        else:
            c = [s.name for s in mt5.symbols_get() if s.name.upper().startswith(w)]
            out[w] = c[0] if c else None
        if out[w]:
            mt5.symbol_select(out[w], True)
    return out


def filling(sym):
    fm = mt5.symbol_info(sym).filling_mode
    if fm & 2:
        return mt5.ORDER_FILLING_IOC
    if fm & 1:
        return mt5.ORDER_FILLING_FOK
    return mt5.ORDER_FILLING_RETURN


def quote(sym):
    t = mt5.symbol_info_tick(sym)
    return t.bid, t.ask, (t.ask + t.bid) / 2, (t.ask - t.bid) / ((t.ask + t.bid) / 2) * 1e4 / 2  # half-bp


def send(sym, side, volume, comment):
    bid, ask, mid, half = quote(sym)
    is_buy = side == "BUY"
    price = ask if is_buy else bid
    if DRY:
        return True, price, mid, half, "DRY"
    req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": sym, "volume": volume,
           "type": mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL, "price": price,
           "deviation": 20, "magic": MAGIC, "comment": comment,
           "type_time": mt5.ORDER_TIME_GTC, "type_filling": filling(sym)}
    r = mt5.order_send(req)
    ok = r is not None and r.retcode == mt5.TRADE_RETCODE_DONE
    # This broker returns result.price == 0 even on a filled market order. Read the TRUE fill from the
    # resulting deal; fall back to the requested price. Never return 0 — a 0 fill divides-by-zero downstream.
    fill = price
    if ok:
        if getattr(r, "price", 0):
            fill = r.price
        elif getattr(r, "deal", 0):
            d = mt5.history_deals_get(ticket=r.deal)
            if d:
                fill = d[0].price
    return ok, (fill or price), mid, half, (str(r.retcode) if r else "None")


def our_positions():
    ps = mt5.positions_get() or []
    return [p for p in ps if p.magic == MAGIC]


def wait_new_position(sym, before, timeout=6.0):
    """order_send returns DONE before positions_get reflects the new position (broker async lag), and
    this broker also returns 0 for result.order/deal/price. So poll until our new position appears."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        new = [p for p in our_positions() if p.symbol == sym and p.ticket not in before]
        if new:
            return max(new, key=lambda x: x.ticket)
        time.sleep(0.3)
    return None


def flatten(reason):
    for p in our_positions():
        side = "SELL" if p.type == mt5.ORDER_TYPE_BUY else "BUY"
        bid, ask, *_ = quote(p.symbol)
        if DRY:
            continue
        req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": p.symbol, "volume": p.volume,
               "type": mt5.ORDER_TYPE_SELL if side == "SELL" else mt5.ORDER_TYPE_BUY,
               "price": bid if side == "SELL" else ask, "position": p.ticket,
               "deviation": 20, "magic": MAGIC, "comment": f"flat:{reason}",
               "type_time": mt5.ORDER_TIME_GTC, "type_filling": filling(p.symbol)}
        mt5.order_send(req)
    if our_positions():
        print(f"  ! flatten({reason}) left {len(our_positions())} open — will retry next tick")


def log_rows(rows):
    df = pd.DataFrame(rows)
    try:
        old = pd.read_parquet(LOG)
        df = pd.concat([old, df], ignore_index=True)
    except FileNotFoundError:
        pass
    df.to_parquet(LOG, index=False)


def try_enter(syms, openpos, sid, side, now):
    """Poll the spread within a grace window; enter each pair the instant its half-spread is favourable
    (<= MAX_ENTRY_HALF). Pairs that never get a good spread before the window expires are SKIPPED and
    logged (skipped=True). This both waits out the top-of-hour flare and refuses bad-spread fills.
    Blocks up to ENTRY_POLL_SECONDS — fine, the matching exit is ~57 min away."""
    deadline = time.time() + ENTRY_POLL_SECONDS
    pending = dict(syms)
    while pending and time.time() < deadline:
        heartbeat()
        for w, sym in list(pending.items()):
            bid, ask, mid, half = quote(sym)
            if half <= MAX_ENTRY_HALF[w]:
                before = {p.ticket for p in our_positions() if p.symbol == sym}
                ok, fill, fmid, fhalf, rc = send(sym, side, LOT, f"016v3-{sid}-in")
                tk = None
                if ok:                                   # hedging acct: wait for the new position, capture ticket
                    pp = wait_new_position(sym, before)
                    if pp:
                        tk, fill = pp.ticket, pp.price_open   # TRUE fill from the position record
                tnow = datetime.now(timezone.utc).astimezone(LONDON)
                openpos[(sid, w)] = {"side": side, "entry_fill": fill, "entry_mid": fmid, "ticket": tk,
                                     "entry_half_bp": fhalf, "entry_time": str(tnow), "sym": sym,
                                     "ok": ok, "rc": rc}
                waited = ENTRY_POLL_SECONDS - max(0.0, deadline - time.time())
                print(f"    {sym} {side} fill {fill:.5f} half {fhalf:.3f}bp (waited {waited:.0f}s, "
                      f"cap {MAX_ENTRY_HALF[w]:.2f}) rc={rc}")
                del pending[w]
        if pending:
            time.sleep(POLL_EVERY)
    # window expired — anything still pending is skipped on spread
    if pending:
        acct = mt5.account_info()
        skip_rows = []
        for w, sym in pending.items():
            bid, ask, mid, half = quote(sym)
            print(f"    {sym} SKIPPED — spread {half:.3f}bp still > cap {MAX_ENTRY_HALF[w]:.2f}bp "
                  f"after {ENTRY_POLL_SECONDS}s")
            skip_rows.append({"date": str(now.date()), "session": sid, "symbol": sym, "side": side,
                              "entry_time": str(datetime.now(timezone.utc).astimezone(LONDON)),
                              "exit_time": None, "entry_fill": None, "exit_fill": None,
                              "entry_half_bp": half, "exit_half_bp": None,
                              "gross_bp": None, "net_bp": None, "dry": DRY,
                              "account": (acct.login if acct else None), "skipped": True})
        log_rows(skip_rows)


def step(syms, state, openpos):
    now = datetime.now(timezone.utc).astimezone(LONDON)
    tag = f"{now:%Y-%m-%d %H:%M:%S} London (wd {now.weekday()})"
    if now.weekday() >= 5:
        print(f"{tag}: weekend, idle"); return
    for sid, cfg in SESSIONS.items():
        eh, em = cfg["enter"]; xh, xm = cfg["exit"]; side = cfg["side"]
        ekey = (now.date(), sid)
        # FRIDAY SKIP — FX closes ~22:00 London Fri: session A can't exit before the weekend and
        # session B is after the close. Don't enter Fridays at all (same as v1).
        if now.weekday() == 4 and now.hour == eh and em <= now.minute < em + 2 and state.get(ekey) is None:
            print(f"{tag}: session {sid} SKIPPED — Friday (no safe exit before FX weekend close)")
            state[ekey] = "skipped-fri"
        # ENTER — spread-gated, flare-aware (the only difference vs v1)
        if now.hour == eh and em <= now.minute < em + 2 and state.get(ekey) is None:
            print(f"{tag}: ENTER session {sid} ({side}) — polling up to {ENTRY_POLL_SECONDS}s for a "
                  f"favourable spread")
            try_enter(syms, openpos, sid, side, now)
            state[ekey] = "entered"
        # EXIT — close BY TICKET (hedging account); only LOG a leg when its close actually succeeds
        # (rc DONE). A rejected close (e.g. rc 10018 market-closed) must NOT be logged as a filled
        # round-trip — keep the position and retry; never fake a net.
        if now.hour == xh and now.minute >= xm and state.get(ekey) == "entered":
            print(f"{tag}: EXIT session {sid}")
            rows = []
            failed_any = False
            for w, sym in syms.items():
                rec = openpos.pop((sid, w), None)
                if rec is None:                       # was skipped on spread — nothing to close
                    continue
                tk = rec.get("ticket")
                positions = [p for p in our_positions() if (tk and p.ticket == tk) or (not tk and p.symbol == sym)]
                if not positions:
                    print(f"    {sym}: no open position (already flat)"); continue
                closed_ok = True
                for p in positions:
                    bid, ask, xmid, xhalf = quote(sym)
                    is_buy_close = p.type == mt5.ORDER_TYPE_SELL    # close a SELL by BUYing back
                    xfill = ask if is_buy_close else bid
                    retcode, rc = None, "DRY"
                    if not DRY:
                        cr = mt5.order_send({"action": mt5.TRADE_ACTION_DEAL, "symbol": sym, "volume": p.volume,
                                             "type": mt5.ORDER_TYPE_BUY if is_buy_close else mt5.ORDER_TYPE_SELL,
                                             "price": xfill, "position": p.ticket, "deviation": 20, "magic": MAGIC,
                                             "comment": f"016v3-{sid}-out", "type_time": mt5.ORDER_TIME_GTC,
                                             "type_filling": filling(sym)})
                        retcode = cr.retcode if cr else None
                        rc = str(retcode)
                    if not (DRY or retcode == mt5.TRADE_RETCODE_DONE):
                        closed_ok = False
                        alert(f"EXIT close REJECTED {sym} rc={rc} — position {p.ticket} STILL OPEN, will retry")
                        continue                       # do NOT log a fake fill
                    try:
                        s = -1.0 if rec["side"] == "SELL" else 1.0
                        ef, em_ = rec["entry_fill"], rec["entry_mid"]
                        net_bp = s * (xfill - ef) / ef * 1e4 if ef else float("nan")
                        gross_bp = s * (xmid - em_) / em_ * 1e4 if em_ else float("nan")
                        acct = mt5.account_info()
                        rows.append({"date": str(now.date()), "session": sid, "symbol": sym, "side": rec["side"],
                                     "entry_time": rec["entry_time"], "exit_time": str(now),
                                     "entry_fill": ef, "exit_fill": xfill,
                                     "entry_half_bp": rec["entry_half_bp"], "exit_half_bp": xhalf,
                                     "gross_bp": gross_bp, "net_bp": net_bp, "dry": DRY,
                                     "account": (acct.login if acct else None), "skipped": False})
                        print(f"    {sym} close {xfill:.5f} | gross {gross_bp:+.2f}bp net {net_bp:+.2f}bp rc={rc}")
                    except Exception as e:
                        print(f"    {sym} closed (rc={rc}) but metric/log failed: {e}")
                if not closed_ok:
                    openpos[(sid, w)] = rec            # keep for retry next loop (do not mark exited)
                    failed_any = True
            if rows:
                log_rows(rows)
                tot = sum(r["net_bp"] for r in rows)
                print(f"    session {sid} net total {tot:+.2f}bp over {len(rows)} closed leg(s)")
            state[ekey] = "entered" if failed_any else "exited"
    in_window = any(now.hour == c["enter"][0] and now.minute >= c["enter"][1] or
                    (now.hour == c["exit"][0] and now.minute < c["exit"][1]) for c in SESSIONS.values())
    if not in_window and our_positions():
        flatten("watchdog")


def main():
    if not DRY and not ONCE and lock_is_fresh():
        print(f"[{VARIANT}] another v2 daemon is already running (fresh lock) — exiting", flush=True)
        sys.exit(0)
    heartbeat()
    connect()
    syms = resolve()
    print("symbols:", syms, "| entry caps:", MAX_ENTRY_HALF)
    if any(v is None for v in syms.values()):
        print("missing symbol — abort"); mt5.shutdown(); sys.exit(1)
    flatten("startup")
    state, openpos = {}, {}
    if ONCE:
        write_status("connected", mt5.account_info())
        now = datetime.now(timezone.utc).astimezone(LONDON)
        print(f"\nnow: {now:%Y-%m-%d %H:%M:%S} London (weekday {now.weekday()})")
        print("live quotes vs entry caps:")
        for w, sym in syms.items():
            bid, ask, mid, half = quote(sym)
            ok = "OK (would enter)" if half <= MAX_ENTRY_HALF[w] else "WIDE (would wait/skip)"
            print(f"   {sym}: half {half:.3f}bp vs cap {MAX_ENTRY_HALF[w]:.2f}bp -> {ok}")
        step(syms, state, openpos)
        print("\nplumbing OK — v2 spread-gate logic exercised, no orders (dry).")
        mt5.shutdown(); return
    print(f"[{VARIANT}] daemon running — Ctrl-C to stop. Spread-gated entry, exits 21:58/23:58.")
    connected = True
    last_login = mt5.account_info().login
    try:
        while True:
            try:
                heartbeat()
                acct = mt5.account_info()
                if acct is None:
                    if connected:
                        alert(f"connection/auth lost mid-run ({mt5.last_error()}) — retrying until the "
                              f"FTMO demo is logged back in.")
                        connected = False
                    write_status("disconnected", err=mt5.last_error())
                    mt5.shutdown(); mt5.initialize(); time.sleep(10); continue
                if not connected:
                    print(f"[{VARIANT}] reconnected: {acct.server} login {acct.login}", flush=True)
                    if acct.login != last_login:
                        alert(f"reconnected on a DIFFERENT account: {acct.login} (was {last_login}).")
                    flatten("reconnect")
                    connected, last_login = True, acct.login
                write_status("connected", acct)
                step(syms, state, openpos)
            except Exception as e:
                print(f"step error: {e}", flush=True)
            time.sleep(15)
    except KeyboardInterrupt:
        print("stopping — flattening")
        flatten("shutdown")
        mt5.shutdown()


if __name__ == "__main__":
    main()

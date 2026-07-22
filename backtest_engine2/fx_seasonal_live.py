"""#016 live forward-test on the FTMO MT5 demo. Trades the 21:00/23:00 London FX seasonal.
  Session A 21:00 London -> SELL EURUSD/GBPUSD/USDJPY, exit 21:58 (BEFORE the 22:00 rollover).
  Session B 23:00 London -> BUY  same, exit 23:58.
Robust by design: startup flatten (no orphan positions into rollover/overnight), time-driven idempotent
exits, watchdog. Logs every round-trip (entry/exit fills, real spread, realized net bp) to parquet.

  python fx_seasonal_live.py --dry --once   # validation: connect, read ticks, print intent, NO orders
  python fx_seasonal_live.py                 # LIVE daemon on the demo (needs AutoTrading enabled)
"""
import os
import sys
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import MetaTrader5 as mt5
import pandas as pd

LONDON = ZoneInfo("Europe/London")
LOCK = r"C:\Users\User\backtest_engine\backtest_engine2\data\.fx_daemon.lock"
ALERTLOG = r"C:\Users\User\backtest_engine\backtest_engine2\data\daemon_alert.log"
STATUS = r"C:\Users\User\backtest_engine\backtest_engine2\data\daemon_status.txt"


def alert(msg):
    """Loud, persistent notice for conditions a human must fix (auth lost, demo expired, AutoTrading off)."""
    line = f"{datetime.now(timezone.utc).astimezone(LONDON):%Y-%m-%d %H:%M:%S} ALERT: {msg}"
    print(">>> " + line, flush=True)
    try:
        with open(ALERTLOG, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def write_status(state, acct=None, err=None):
    """One-file health snapshot so 'check the daemon' is a single read, not a process hunt."""
    try:
        now = datetime.now(timezone.utc).astimezone(LONDON)
        lines = [f"time={now:%Y-%m-%d %H:%M:%S %Z}", f"state={state}", f"pid={os.getpid()}"]
        if acct is not None:
            lines += [f"login={acct.login}", f"server={acct.server}", f"balance={acct.balance:.2f}"]
        if err is not None:
            lines.append(f"last_error={err}")
        with open(STATUS, "w") as f:
            f.write("\n".join(lines) + "\n")
    except Exception:
        pass


def lock_is_fresh():
    """True if another live daemon wrote a heartbeat in the last 60s."""
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
MAGIC = 160160
LOT = 0.10
WANT = ["EURUSD", "GBPUSD", "USDJPY"]
SESSIONS = {
    # Session A exits 21:50, NOT 21:58: the broker has a nightly FX rollover HALT ~21:54:50-22:05 London
    # (measured from the tick feed; orders return rc 10018 inside it). 21:58 fell inside the halt, so the
    # close was always rejected and the position closed late by the watchdog. 21:50 = ~5 min safety buffer.
    "A": {"enter": (21, 0), "exit": (21, 50), "side": "SELL"},
    "B": {"enter": (23, 0), "exit": (23, 58), "side": "BUY"},
}
LOG = r"C:\Users\User\backtest_engine\backtest_engine2\data\fx_seasonal_live_log.parquet"
DRY = "--dry" in sys.argv
ONCE = "--once" in sys.argv


def connect(max_attempts=None):
    # retry — when auto-launched at logon, MT5 terminal may not be up yet. For the live daemon we
    # retry FOREVER (max_attempts=None): a demo expiry / reboot / logged-out terminal must NOT kill
    # the run — it parks here, alerts, and recovers the moment the terminal is logged back in.
    attempt = 0
    while max_attempts is None or attempt < max_attempts:
        if mt5.initialize() and mt5.account_info() is not None:
            a = mt5.account_info()
            print(f"{datetime.now(timezone.utc).astimezone(LONDON):%Y-%m-%d %H:%M} Connected: {a.server} | "
                  f"login {a.login} | balance {a.balance:,.0f} {a.currency} | "
                  f"{'DRY-RUN (no orders)' if DRY else 'LIVE ORDERS'}", flush=True)
            write_status("connected", a)
            return
        err = mt5.last_error()
        if attempt % 6 == 0:                          # ~once/min
            print(f"waiting for MT5 terminal... ({err})", flush=True)
            write_status("disconnected", err=err)
        if attempt > 0 and attempt % 180 == 0:        # ~every 30 min while still down
            alert(f"MT5 not connected for ~{attempt * 10 // 60} min ({err}). "
                  f"Log the FTMO demo back in / renew the Free Trial / enable AutoTrading.")
        attempt += 1
        time.sleep(10)
    print("could not connect — is the FTMO MT5 terminal running & logged in?")
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
        side = "SELL" if p.type == mt5.ORDER_TYPE_BUY else "BUY"  # opposite
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


def step(syms, state, openpos):
    now = datetime.now(timezone.utc).astimezone(LONDON)
    tag = f"{now:%Y-%m-%d %H:%M:%S} London (wd {now.weekday()})"
    if now.weekday() >= 5:
        print(f"{tag}: weekend, idle"); return
    for sid, cfg in SESSIONS.items():
        eh, em = cfg["enter"]; xh, xm = cfg["exit"]; side = cfg["side"]
        ekey = (now.date(), sid)
        # FRIDAY SKIP — FX closes ~22:00 London Fri: session A can't exit before the weekend (close
        # rejected, position stranded) and session B is after the close. Don't enter Fridays at all.
        if now.weekday() == 4 and now.hour == eh and em <= now.minute < em + 2 and state.get(ekey) is None:
            print(f"{tag}: session {sid} SKIPPED — Friday (no safe exit before FX weekend close)")
            state[ekey] = "skipped-fri"
        # ENTER
        if now.hour == eh and em <= now.minute < em + 2 and state.get(ekey) is None:
            print(f"{tag}: ENTER session {sid} ({side})")
            for w, sym in syms.items():
                before = {p.ticket for p in our_positions() if p.symbol == sym}
                ok, fill, mid, half, rc = send(sym, side, LOT, f"016-{sid}-in")
                tk = None
                if ok:                                   # hedging acct: wait for the new position, capture ticket
                    p = wait_new_position(sym, before)
                    if p:
                        tk, fill = p.ticket, p.price_open   # TRUE fill from the position record
                openpos[(sid, w)] = {"side": side, "entry_fill": fill, "entry_mid": mid, "ticket": tk,
                                     "entry_half_bp": half, "entry_time": str(now), "sym": sym, "ok": ok, "rc": rc}
                print(f"    {sym} {side} fill {fill:.5f} half-spread {half:.3f}bp ticket {tk} rc={rc}")
            state[ekey] = "entered"
        # EXIT — close BY TICKET (hedging account); only LOG a leg when its close actually succeeds
        # (rc DONE). A rejected close (e.g. rc 10018 market-closed near the Fri weekend) must NOT be
        # logged as a filled round-trip — keep the position and retry; never fake a net.
        if now.hour == xh and now.minute >= xm and state.get(ekey) == "entered":
            print(f"{tag}: EXIT session {sid}")
            rows = []
            failed_any = False
            for w, sym in syms.items():
                rec = openpos.pop((sid, w), None)
                if rec is None:
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
                                             "comment": f"016-{sid}-out", "type_time": mt5.ORDER_TIME_GTC,
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
                                     "account": (acct.login if acct else None)})
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
    # watchdog: outside any [enter, exit] hold window, we should be flat
    in_window = any(now.hour == c["enter"][0] and now.minute >= c["enter"][1] or
                    (now.hour == c["exit"][0] and now.minute < c["exit"][1]) for c in SESSIONS.values())
    if not in_window and our_positions():
        flatten("watchdog")


def main():
    if not DRY and not ONCE and lock_is_fresh():
        print("another live daemon is already running (fresh lock) — exiting to avoid duplicate orders",
              flush=True)
        sys.exit(0)
    heartbeat()
    connect()
    syms = resolve()
    print("symbols:", syms)
    if any(v is None for v in syms.values()):
        print("missing symbol — abort"); mt5.shutdown(); sys.exit(1)
    flatten("startup")  # no orphans into rollover/overnight
    state, openpos = {}, {}
    if ONCE:
        write_status("connected", mt5.account_info())
        now = datetime.now(timezone.utc).astimezone(LONDON)
        srv = datetime.utcfromtimestamp(mt5.symbol_info_tick(list(syms.values())[0]).time)
        print(f"\nnow: {now:%Y-%m-%d %H:%M:%S} London (weekday {now.weekday()}) | server clock {srv:%H:%M}")
        print("live quotes:")
        for w, sym in syms.items():
            bid, ask, mid, half = quote(sym)
            print(f"   {sym}: bid {bid:.5f} ask {ask:.5f} | half-spread {half:.3f}bp (r/t {2*half:.3f}bp)")
        nxt = [f"{s} {c['enter'][0]:02d}:00->{c['exit'][0]:02d}:{c['exit'][1]} ({c['side']})"
               for s, c in SESSIONS.items()]
        print("sessions:", " | ".join(nxt))
        step(syms, state, openpos)
        print("\nplumbing OK — connection, symbols, time conversion, tick reads all working.")
        mt5.shutdown(); return
    print("daemon running — Ctrl-C to stop. Trades 21:00/23:00 London, exits 21:58/23:58.")
    connected = True
    last_login = mt5.account_info().login
    try:
        while True:
            try:
                heartbeat()                           # single-instance lock: a 2nd daemon sees this & exits
                acct = mt5.account_info()
                if acct is None:                      # terminal restarted / demo expired / logged out
                    if connected:                     # log the transition once, not every 15s
                        alert(f"connection/auth lost mid-run ({mt5.last_error()}) — retrying until the "
                              f"FTMO demo is logged back in. No trades will fire while down.")
                        connected = False
                    write_status("disconnected", err=mt5.last_error())
                    mt5.shutdown()                    # force a clean re-init against the terminal
                    mt5.initialize()
                    time.sleep(10)
                    continue
                if not connected:                     # recovered
                    print(f"reconnected: {acct.server} login {acct.login}", flush=True)
                    if acct.login != last_login:
                        alert(f"reconnected on a DIFFERENT account: {acct.login} (was {last_login}). "
                              f"New demo — forward-test continuity resets; eval pools per-leg across both.")
                    flatten("reconnect")              # clear any orphans before resuming
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

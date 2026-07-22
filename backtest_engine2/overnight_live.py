"""#023 live forward-test on the FTMO MT5 demo. Overnight S&P (USA500) + 200d regime filter.
  Enter LONG ~18:00 New York (AFTER the ~16-17:00 ET daily rollover -> dodges overnight swap),
  exit ~09:30 NY (cash open) the NEXT morning. Only enters if USA500 > its 200-day average (regime ON).
  Weekend-safe: enters Mon-Thu evenings only, so a position never crosses a weekend rollover.

Reuses the hardened #016 skeleton: retry-forever connect, startup/watchdog flatten, crash-proof
close-by-ticket, broker price==0 + async-position-lag workarounds. Logs every round-trip + regime + skips.

  python overnight_live.py --dry --once   # validate: connect, read bars/quote, print regime+intent, NO orders
  python overnight_live.py                 # LIVE daemon on the demo (AutoTrading enabled)

HONEST NOTE: a short forward test cannot validate the REGIME FILTER (no crash in the window -> it just
stays ON); it validates whether the OVERNIGHT EDGE is real live after real fills/spread. The filter was
already validated on history (#023: dodged 2008, maxDD -20%->-8%).
"""
import os
import sys
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import MetaTrader5 as mt5
import pandas as pd

NY = ZoneInfo("America/New_York")
DATA = r"C:\Users\User\backtest_engine\backtest_engine2\data"
LOCK = DATA + r"\.overnight_daemon.lock"
ALERTLOG = DATA + r"\overnight_alert.log"
STATUS = DATA + r"\overnight_status.txt"
LOG = DATA + r"\overnight_live_log.parquet"

MAGIC = 160162
LOT = 0.10
WANT = ["USA500", "US500", "SP500", "SPX500", "US500.cash"]   # broker naming varies; first match wins
SMA_LEN = 200
ENTER = (18, 0)     # 18:00 NY (post-rollover, swap-dodge)
EXIT = (9, 30)      # 09:30 NY (cash open) next morning
DRY = "--dry" in sys.argv
ONCE = "--once" in sys.argv


def alert(msg):
    line = f"{datetime.now(timezone.utc).astimezone(NY):%Y-%m-%d %H:%M:%S} ALERT: {msg}"
    print(">>> " + line, flush=True)
    try:
        with open(ALERTLOG, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def write_status(state, acct=None, extra=None, err=None):
    try:
        now = datetime.now(timezone.utc).astimezone(NY)
        lines = [f"variant=023_overnight", f"time={now:%Y-%m-%d %H:%M:%S %Z}", f"state={state}", f"pid={os.getpid()}"]
        if acct is not None:
            lines += [f"login={acct.login}", f"server={acct.server}", f"balance={acct.balance:.2f}"]
        if extra:
            lines.append(extra)
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
            print(f"{datetime.now(timezone.utc).astimezone(NY):%Y-%m-%d %H:%M} Connected: {a.server} | "
                  f"login {a.login} | balance {a.balance:,.0f} {a.currency} | "
                  f"{'DRY-RUN (no orders)' if DRY else 'LIVE ORDERS'}", flush=True)
            write_status("connected", a)
            return
        err = mt5.last_error()
        if attempt % 6 == 0:
            print(f"waiting for MT5 terminal... ({err})", flush=True)
            write_status("disconnected", err=err)
        if attempt > 0 and attempt % 180 == 0:
            alert(f"MT5 not connected for ~{attempt * 10 // 60} min ({err}). Log the FTMO demo back in.")
        attempt += 1
        time.sleep(10)
    print("could not connect — is the FTMO MT5 terminal running & logged in?")
    write_status("disconnected", err=mt5.last_error())
    sys.exit(1)


def resolve():
    for w in WANT:
        if mt5.symbol_info(w) is not None:
            mt5.symbol_select(w, True)
            return w
    for w in WANT:
        c = [s.name for s in mt5.symbols_get() if s.name.upper().startswith(w.upper().split(".")[0])]
        if c:
            mt5.symbol_select(c[0], True)
            return c[0]
    return None


def filling(sym):
    fm = mt5.symbol_info(sym).filling_mode
    if fm & 2:
        return mt5.ORDER_FILLING_IOC
    if fm & 1:
        return mt5.ORDER_FILLING_FOK
    return mt5.ORDER_FILLING_RETURN


def quote(sym):
    t = mt5.symbol_info_tick(sym)
    mid = (t.ask + t.bid) / 2
    return t.bid, t.ask, mid, (t.ask - t.bid) / mid * 1e4 / 2     # half-spread bp


def regime_on(sym):
    """200d trend filter from the broker's COMPLETED daily bars (PIT). Returns (on, price, sma) or (None,...)
    if not enough history -> caller SKIPS (never trade without the filter)."""
    bars = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_D1, 1, SMA_LEN)   # skip today's forming bar
    if bars is None or len(bars) < SMA_LEN:
        return None, None, None
    closes = pd.Series([b["close"] for b in bars])
    sma = closes.mean()
    _, _, mid, _ = quote(sym)
    return (mid > sma), mid, sma


def our_positions():
    return [p for p in (mt5.positions_get() or []) if p.magic == MAGIC]


def wait_new_position(sym, before, timeout=6.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        new = [p for p in our_positions() if p.symbol == sym and p.ticket not in before]
        if new:
            return max(new, key=lambda x: x.ticket)
        time.sleep(0.3)
    return None


def buy(sym, comment):
    bid, ask, mid, half = quote(sym)
    if DRY:
        return True, ask, mid, half, "DRY"
    r = mt5.order_send({"action": mt5.TRADE_ACTION_DEAL, "symbol": sym, "volume": LOT,
                        "type": mt5.ORDER_TYPE_BUY, "price": ask, "deviation": 20, "magic": MAGIC,
                        "comment": comment, "type_time": mt5.ORDER_TIME_GTC, "type_filling": filling(sym)})
    ok = r is not None and r.retcode == mt5.TRADE_RETCODE_DONE
    fill = ask
    if ok and getattr(r, "price", 0):
        fill = r.price
    return ok, fill, mid, half, (str(r.retcode) if r else "None")


def close_position(p):
    """Close a position BY TICKET (hedging account: an opposing order would open a 2nd position)."""
    bid, ask, xmid, xhalf = quote(p.symbol)
    is_buy_close = p.type == mt5.ORDER_TYPE_SELL
    xfill = ask if is_buy_close else bid
    rc = "DRY"
    if not DRY:
        cr = mt5.order_send({"action": mt5.TRADE_ACTION_DEAL, "symbol": p.symbol, "volume": p.volume,
                             "type": mt5.ORDER_TYPE_BUY if is_buy_close else mt5.ORDER_TYPE_SELL,
                             "price": xfill, "position": p.ticket, "deviation": 20, "magic": MAGIC,
                             "comment": "023-out", "type_time": mt5.ORDER_TIME_GTC, "type_filling": filling(p.symbol)})
        rc = str(cr.retcode) if cr else "None"
    return xfill, xmid, xhalf, rc


def flatten(reason):
    for p in our_positions():
        if DRY:
            continue
        close_position(p)
    if our_positions():
        print(f"  ! flatten({reason}) left {len(our_positions())} open — retry next tick")


def log_rows(rows):
    df = pd.DataFrame(rows)
    try:
        df = pd.concat([pd.read_parquet(LOG), df], ignore_index=True)
    except FileNotFoundError:
        pass
    df.to_parquet(LOG, index=False)


def step(sym, st):
    now = datetime.now(timezone.utc).astimezone(NY)
    tag = f"{now:%Y-%m-%d %H:%M:%S} NY (wd {now.weekday()})"
    eh, em = ENTER
    xh, xm = EXIT

    # ENTER: Mon-Thu (0-3) evening. Window 18:00-18:11 NY = the CME REOPEN after the 17:00-18:00 maintenance
    # break. 18:00 is LOAD-BEARING (per #023 notes): it is the EARLIEST tradeable time AFTER the ~17:00 NY swap
    # snapshot, so the hold is swap-free (entering before 17:00 would pay swap; #023 only works swap-free, and
    # waiting later bleeds overnight move). We RETRY across the window because an order at the exact 18:00 reopen
    # edge gets rejected — but we ONLY mark LONG on a CONFIRMED real position (never a phantom).
    ENTER_RETRY_MIN = 12
    if (now.weekday() <= 3 and now.hour == eh and em <= now.minute < em + ENTER_RETRY_MIN
            and st["open"] is None and st["entered_on"] != now.date()):
        on, price, sma = regime_on(sym)
        if on is None:
            print(f"{tag}: SKIP — <{SMA_LEN} daily bars, can't compute filter (won't trade blind)")
            st["entered_on"] = now.date()
        elif not on:
            print(f"{tag}: regime OFF (px {price:.1f} < 200d {sma:.1f}) — SKIP tonight")
            log_rows([{"date": str(now.date()), "action": "skip_regime_off", "price": price, "sma200": sma,
                       "dry": DRY, "account": (mt5.account_info().login if mt5.account_info() else None)}])
            st["entered_on"] = now.date()
        else:
            before = {p.ticket for p in our_positions()}
            ok, fill, mid, half, rc = buy(sym, "023-in")
            p = wait_new_position(sym, before) if ok else None
            if p:                                          # CONFIRMED real position -> genuine LONG
                st["open"] = {"entry_fill": p.price_open, "entry_mid": mid, "entry_half_bp": half,
                              "ticket": p.ticket, "entry_time": str(now), "sym": sym, "sma200": sma, "rc": rc}
                st["entered_on"] = now.date()
                print(f"{tag}: ENTER LONG (regime ON, px {price:.1f} > 200d {sma:.1f}) "
                      f"fill {p.price_open:.1f} half {half:.3f}bp ticket {p.ticket} rc={rc}")
            else:                                          # reopen edge rejected / no fill -> retry next loop
                print(f"{tag}: entry not filled (rc={rc}) — market reopening, retrying...", flush=True)
                if now.minute >= em + ENTER_RETRY_MIN - 1:  # window nearly over -> give up tonight (no phantom)
                    alert(f"#023 entry FAILED across the whole 18:00 reopen window (last rc={rc}) — no position tonight.")
                    st["entered_on"] = now.date()

    # EXIT: next morning ~09:30 NY, position open
    if st["open"] is not None and now.hour == xh and now.minute >= xm and now.minute < xm + 20:
        rec = st["open"]
        positions = [p for p in our_positions() if (rec.get("ticket") and p.ticket == rec["ticket"])
                     or (not rec.get("ticket") and p.symbol == sym)]
        if not positions:
            print(f"{tag}: EXIT — no open position found (already flat)")
            st["open"] = None
        else:
            rows = []
            all_closed = True
            for p in positions:
                xfill, xmid, xhalf, rc = close_position(p)
                if not (DRY or rc == str(mt5.TRADE_RETCODE_DONE)):   # honest: don't log/clear a rejected close
                    all_closed = False
                    alert(f"#023 EXIT close REJECTED rc={rc} — position {p.ticket} STILL OPEN, will retry")
                    continue
                try:
                    ef, emid = rec["entry_fill"], rec["entry_mid"]
                    net_bp = (xfill - ef) / ef * 1e4 if ef else float("nan")       # LONG
                    gross_bp = (xmid - emid) / emid * 1e4 if emid else float("nan")
                    acct = mt5.account_info()
                    rows.append({"date": str(now.date()), "action": "trade", "symbol": sym, "side": "BUY",
                                 "entry_time": rec["entry_time"], "exit_time": str(now),
                                 "entry_fill": ef, "exit_fill": xfill, "sma200": rec.get("sma200"),
                                 "entry_half_bp": rec["entry_half_bp"], "exit_half_bp": xhalf,
                                 "gross_bp": gross_bp, "net_bp": net_bp, "dry": DRY,
                                 "account": (acct.login if acct else None)})
                    print(f"{tag}: EXIT close {xfill:.1f} | gross {gross_bp:+.2f}bp net {net_bp:+.2f}bp rc={rc}")
                except Exception as e:
                    print(f"{tag}: EXIT closed (rc={rc}) but metric/log failed: {e}")
            if rows:
                log_rows(rows)
            if all_closed:               # only mark flat once every leg truly closed; else retry next loop + watchdog
                st["open"] = None

    # watchdog: between 10:00 and 17:59 NY we must be flat (missed exit / orphan into rollover)
    if st["open"] is not None and 10 <= now.hour < 18:
        print(f"{tag}: watchdog — position open in daytime, flattening (missed exit?)")
        flatten("watchdog")
        st["open"] = None


def main():
    if not DRY and not ONCE and lock_is_fresh():
        print("another #023 daemon already running (fresh lock) — exiting to avoid duplicate orders", flush=True)
        sys.exit(0)
    heartbeat()
    connect()
    sym = resolve()
    print("symbol:", sym)
    if sym is None:
        print("USA500 symbol not found — abort"); mt5.shutdown(); sys.exit(1)
    flatten("startup")
    st = {"open": None, "entered_on": None}

    if ONCE:
        now = datetime.now(timezone.utc).astimezone(NY)
        on, price, sma = regime_on(sym)
        bid, ask, mid, half = quote(sym)
        print(f"\nnow: {now:%Y-%m-%d %H:%M:%S} NY (weekday {now.weekday()})")
        print(f"{sym}: bid {bid:.1f} ask {ask:.1f} | half-spread {half:.3f}bp (r/t {2*half:.3f}bp)")
        if on is None:
            print(f"regime: UNKNOWN — only {0 if mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_D1, 1, SMA_LEN) is None else len(mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_D1, 1, SMA_LEN))} daily bars (<{SMA_LEN})")
        else:
            print(f"regime: {'ON (would trade)' if on else 'OFF (would skip)'} — price {price:.1f} vs 200d {sma:.1f}")
        print(f"schedule: enter Mon-Thu {eh_str()} NY (post-rollover), exit next day 09:30 NY")
        print("\nplumbing OK — connection, symbol, daily bars, regime calc, tick reads all working.")
        mt5.shutdown(); return

    print("daemon running — Ctrl-C to stop. Enter 18:00 NY (regime-gated), exit 09:30 NY next day.")
    connected = True
    last_login = mt5.account_info().login
    try:
        while True:
            try:
                heartbeat()
                acct = mt5.account_info()
                if acct is None:
                    if connected:
                        alert(f"connection/auth lost mid-run ({mt5.last_error()}) — retrying. No trades while down.")
                        connected = False
                    write_status("disconnected", err=mt5.last_error())
                    mt5.shutdown(); mt5.initialize(); time.sleep(10); continue
                if not connected:
                    print(f"reconnected: {acct.server} login {acct.login}", flush=True)
                    if acct.login != last_login:
                        alert(f"reconnected on DIFFERENT account {acct.login} (was {last_login}) — continuity resets.")
                    flatten("reconnect"); st["open"] = None
                    connected, last_login = True, acct.login
                openinfo = "flat" if st["open"] is None else f"LONG@{st['open']['entry_fill']:.1f}"
                write_status("connected", acct, extra=f"position={openinfo}")
                step(sym, st)
            except Exception as e:
                print(f"step error: {e}", flush=True)
            time.sleep(15)
    except KeyboardInterrupt:
        print("stopping — flattening")
        flatten("shutdown"); mt5.shutdown()


def eh_str():
    return f"{ENTER[0]:02d}:{ENTER[1]:02d}"


if __name__ == "__main__":
    main()

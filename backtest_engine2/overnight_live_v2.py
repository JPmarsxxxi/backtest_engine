"""#023-v2 live forward-test — overnight S&P gated by the FROZEN RF model (overnight_rf.ipynb).
Runs PARALLEL to v1 (200d filter, magic 160162) on the FTMO demo: same 18:00-NY swap-dodged entry,
same 09:30-NY exit, same Mon-Thu rule — the ONLY difference is WHO decides whether tonight is traded:
  v1: price > 200d SMA          v2: RF P(green overnight) >= 0.52 (artifact data/overnight_rf_v2.joblib,
      trained through 2026-07-15; walk-forward DSR 0.988; annual refits via notebook Cell 12 ONLY).
Decision inputs pulled at entry time: SPY + ^VIX daily from Yahoo (features via overnight_rf_features
= the SAME module the notebook parity-asserted). FAIL-SAFE: if the data pull fails or the latest SPY
row is not TODAY, we skip the night and alert — never trade on stale features.

  python overnight_live_v2.py --dry --once   # validate: connect, pull features, print p + intent
  python overnight_live_v2.py                 # LIVE daemon (AutoTrading enabled)
"""
import os
import sys
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import MetaTrader5 as mt5
import pandas as pd

NY = ZoneInfo("America/New_York")
ENG = r"C:\Users\User\backtest_engine\backtest_engine2"
DATA = ENG + r"\data"
LOCK = DATA + r"\.overnight_v2_daemon.lock"
ALERTLOG = DATA + r"\overnight_alert.log"
STATUS = DATA + r"\overnight_v2_status.txt"
LOG = DATA + r"\overnight_live_v2_log.parquet"
ARTIFACT = DATA + r"\overnight_rf_v2.joblib"

MAGIC = 160164                      # v1 overnight = 160162; fx = 160160/1/3 — never share
LOT = 0.10
WANT = ["USA500", "US500", "SP500", "SPX500", "US500.cash"]
ENTER = (18, 0)
EXIT = (9, 30)
DRY = "--dry" in sys.argv
ONCE = "--once" in sys.argv

sys.path.insert(0, ENG)
import overnight_rf_features as orf  # noqa: E402  (shared, parity-asserted feature builder)

_ART = None


def artifact():
    global _ART
    if _ART is None:
        import joblib
        _ART = joblib.load(ARTIFACT)
    return _ART


def alert(msg):
    line = f"{datetime.now(timezone.utc).astimezone(NY):%Y-%m-%d %H:%M:%S} ALERT [023v2]: {msg}"
    print(">>> " + line, flush=True)
    try:
        with open(ALERTLOG, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def write_status(state, acct=None, extra=None, err=None):
    try:
        now = datetime.now(timezone.utc).astimezone(NY)
        lines = [f"variant=023v2_rf", f"time={now:%Y-%m-%d %H:%M:%S %Z}", f"state={state}",
                 f"pid={os.getpid()}"]
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
            print(f"{datetime.now(timezone.utc).astimezone(NY):%Y-%m-%d %H:%M} [v2] Connected: "
                  f"{a.server} | login {a.login} | balance {a.balance:,.0f} {a.currency} | "
                  f"{'DRY-RUN (no orders)' if DRY else 'LIVE ORDERS'}", flush=True)
            write_status("connected", a)
            return
        err = mt5.last_error()
        if attempt % 6 == 0:
            print(f"waiting for MT5 terminal... ({err})", flush=True)
            write_status("disconnected", err=err)
        if attempt > 0 and attempt % 180 == 0:
            alert(f"MT5 not connected for ~{attempt * 10 // 60} min ({err}).")
        attempt += 1
        time.sleep(10)
    print("could not connect")
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
    return t.bid, t.ask, mid, (t.ask - t.bid) / mid * 1e4 / 2


def model_says(now):
    """(p, detail) or (None, reason). Fail-safe: any doubt -> None -> SKIP tonight + alert."""
    try:
        px = orf.yahoo_ohlc("SPY")
        vixdf = orf.yahoo_ohlc("%5EVIX")
    except Exception as e:
        return None, f"data pull failed: {e}"
    last = px.index[-1].date()
    if last != now.date():
        return None, f"stale features: last SPY row {last} != today {now.date()}"
    art = artifact()
    row = orf.latest_feature_row(px, vixdf["close"])
    if row[art["feats"]].isna().any().any():
        return None, "NaN in tonight's feature row"
    p = float(art["model"].predict_proba(row[art["feats"]])[:, 1][0])
    return p, f"p={p:.3f} thr={art['threshold']} (features of {last})"


def regime_on(sym):
    """strat_tick --selftest compat shim (v1 API shape: (on, price, level)). For v2, 'regime' =
    the model's call: returns (p >= threshold, current mid, p). Real ticks never call this."""
    now = datetime.now(timezone.utc).astimezone(NY)
    p, _ = model_says(now)
    _, _, mid, _ = quote(sym)
    if p is None:
        return None, mid, None
    return p >= artifact()["threshold"], mid, p


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
                        "comment": comment, "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": filling(sym)})
    ok = r is not None and r.retcode == mt5.TRADE_RETCODE_DONE
    fill = ask
    if ok and getattr(r, "price", 0):
        fill = r.price
    return ok, fill, mid, half, (str(r.retcode) if r else "None")


def close_position(p):
    bid, ask, xmid, xhalf = quote(p.symbol)
    is_buy_close = p.type == mt5.ORDER_TYPE_SELL
    xfill = ask if is_buy_close else bid
    rc = "DRY"
    if not DRY:
        cr = mt5.order_send({"action": mt5.TRADE_ACTION_DEAL, "symbol": p.symbol, "volume": p.volume,
                             "type": mt5.ORDER_TYPE_BUY if is_buy_close else mt5.ORDER_TYPE_SELL,
                             "price": xfill, "position": p.ticket, "deviation": 20, "magic": MAGIC,
                             "comment": "023v2-out", "type_time": mt5.ORDER_TIME_GTC,
                             "type_filling": filling(p.symbol)})
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
    tag = f"{now:%Y-%m-%d %H:%M:%S} NY (wd {now.weekday()}) [v2]"
    eh, em = ENTER
    xh, xm = EXIT
    ENTER_RETRY_MIN = 12

    # ENTER: Mon-Thu evenings, 18:00-18:11 NY reopen window (same constraints as v1: post-snapshot
    # = swap-free; Mon-Thu so no weekend rollover). Decision = frozen RF, computed ONCE per night.
    if (now.weekday() <= 3 and now.hour == eh and em <= now.minute < em + ENTER_RETRY_MIN
            and st["open"] is None and st["entered_on"] != now.date()):
        if "tonight_p" not in st or st.get("tonight_date") != str(now.date()):
            p, detail = model_says(now)
            st["tonight_p"], st["tonight_detail"], st["tonight_date"] = p, detail, str(now.date())
            if p is None:
                alert(f"no decision tonight — {detail}. SKIPPING (fail-safe).")
        p, detail = st["tonight_p"], st["tonight_detail"]
        acct = mt5.account_info()
        if p is None:
            st["entered_on"] = now.date()
            log_rows([{"date": str(now.date()), "action": "skip_no_data", "detail": detail,
                       "dry": DRY, "account": (acct.login if acct else None)}])
        elif p < artifact()["threshold"]:
            print(f"{tag}: model says SKIP ({detail})")
            st["entered_on"] = now.date()
            log_rows([{"date": str(now.date()), "action": "skip_model", "prob": p, "detail": detail,
                       "dry": DRY, "account": (acct.login if acct else None)}])
        else:
            before = {q.ticket for q in our_positions()}
            ok, fill, mid, half, rc = buy(sym, "023v2-in")
            pos = wait_new_position(sym, before) if ok else None
            if pos:
                st["open"] = {"entry_fill": pos.price_open, "entry_mid": mid, "entry_half_bp": half,
                              "ticket": pos.ticket, "entry_time": str(now), "sym": sym,
                              "prob": p, "rc": rc}
                st["entered_on"] = now.date()
                print(f"{tag}: ENTER LONG ({detail}) fill {pos.price_open:.1f} "
                      f"half {half:.3f}bp ticket {pos.ticket} rc={rc}")
            else:
                print(f"{tag}: entry not filled (rc={rc}) — reopen edge, retrying...", flush=True)
                if now.minute >= em + ENTER_RETRY_MIN - 1:
                    alert(f"entry FAILED across the reopen window (last rc={rc}) — no position tonight.")
                    st["entered_on"] = now.date()

    # EXIT: next morning 09:30-09:50 NY
    if st["open"] is not None and now.hour == xh and xm <= now.minute < xm + 20:
        rec = st["open"]
        positions = [q for q in our_positions() if (rec.get("ticket") and q.ticket == rec["ticket"])
                     or (not rec.get("ticket") and q.symbol == sym)]
        if not positions:
            print(f"{tag}: EXIT — no open position found (already flat)")
            st["open"] = None
        else:
            rows, all_closed = [], True
            for q in positions:
                xfill, xmid, xhalf, rc = close_position(q)
                if not (DRY or rc == str(mt5.TRADE_RETCODE_DONE)):
                    all_closed = False
                    alert(f"EXIT close REJECTED rc={rc} — position {q.ticket} STILL OPEN, will retry")
                    continue
                try:
                    ef, emid = rec["entry_fill"], rec["entry_mid"]
                    net_bp = (xfill - ef) / ef * 1e4 if ef else float("nan")
                    gross_bp = (xmid - emid) / emid * 1e4 if emid else float("nan")
                    acct = mt5.account_info()
                    rows.append({"date": str(now.date()), "action": "trade", "symbol": sym,
                                 "side": "BUY", "entry_time": rec["entry_time"], "exit_time": str(now),
                                 "entry_fill": ef, "exit_fill": xfill, "prob": rec.get("prob"),
                                 "entry_half_bp": rec["entry_half_bp"], "exit_half_bp": xhalf,
                                 "gross_bp": gross_bp, "net_bp": net_bp, "dry": DRY,
                                 "account": (acct.login if acct else None)})
                    print(f"{tag}: EXIT close {xfill:.1f} | gross {gross_bp:+.2f}bp "
                          f"net {net_bp:+.2f}bp rc={rc}")
                except Exception as e:
                    print(f"{tag}: EXIT closed (rc={rc}) but metric/log failed: {e}")
            if rows:
                log_rows(rows)
            if all_closed:
                st["open"] = None

    # watchdog: 10:00-17:59 NY must be flat
    if st["open"] is not None and 10 <= now.hour < 18:
        print(f"{tag}: watchdog — position open in daytime, flattening (missed exit?)")
        flatten("watchdog")
        st["open"] = None


def main():
    if not DRY and not ONCE and lock_is_fresh():
        print("another 023v2 daemon already running — exiting", flush=True)
        sys.exit(0)
    heartbeat()
    connect()
    sym = resolve()
    print("symbol:", sym)
    if sym is None:
        print("US500 symbol not found — abort"); mt5.shutdown(); sys.exit(1)
    flatten("startup")
    st = {"open": None, "entered_on": None}

    if ONCE:
        now = datetime.now(timezone.utc).astimezone(NY)
        bid, ask, mid, half = quote(sym)
        p, detail = model_says(now)
        print(f"\nnow: {now:%Y-%m-%d %H:%M:%S} NY (weekday {now.weekday()})")
        print(f"{sym}: bid {bid:.1f} ask {ask:.1f} | half-spread {half:.3f}bp")
        print(f"model: {detail if p is not None else 'NO DECISION — ' + detail}")
        if p is not None:
            print(f"tonight: {'WOULD TRADE' if p >= artifact()['threshold'] else 'would SKIP'} "
                  f"(entries only Mon-Thu 18:00-18:11 NY)")
        art = artifact()
        print(f"artifact: trained through {art['trained_through']} | {art['recipe']}")
        print("\nplumbing OK — connection, symbol, feature pull, model, quotes all working.")
        mt5.shutdown(); return

    print("v2 daemon running — enter Mon-Thu 18:00 NY (RF-gated), exit 09:30 NY next day.")
    connected = True
    last_login = mt5.account_info().login
    try:
        while True:
            try:
                heartbeat()
                acct = mt5.account_info()
                if acct is None:
                    if connected:
                        alert(f"connection lost ({mt5.last_error()}) — retrying.")
                        connected = False
                    write_status("disconnected", err=mt5.last_error())
                    mt5.shutdown(); mt5.initialize(); time.sleep(10); continue
                if not connected:
                    print(f"reconnected: {acct.server} login {acct.login}", flush=True)
                    if acct.login != last_login:
                        alert(f"reconnected on DIFFERENT account {acct.login} (was {last_login}).")
                    flatten("reconnect"); st["open"] = None
                    connected, last_login = True, acct.login
                openinfo = "flat" if st["open"] is None else f"LONG@{st['open']['entry_fill']:.1f}"
                write_status("connected", acct, extra=f"position={openinfo}")
                step(sym, st)
            except Exception as e:
                print(f"step error: {e}", flush=True)
            time.sleep(15)
    except KeyboardInterrupt:
        print("stopped by user")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()

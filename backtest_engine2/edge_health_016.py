"""#016 — monthly EDGE-HEALTH report, v2: computed on MIDS (rebuilt 2026-07-16 after the bid-only
artifact postmortem; v1 of this script measured bid closes and its "all OK" was unreliable).

What it does (run monthly by FX_016_monthly_health, or manually):
  1. Maintains TWO hourly series per pair: BID ({sym}_h1.parquet, legacy file) and ASK
     ({sym}_h1_ask.parquet, assembled once from x016k's 2019-22 historic pull + the 1m ask files,
     then extended like bid). Both are extended from Dukascopy to today on every run.
  2. Recomputes per-(pair, hour) mean hourly return on the TRUE MID over the full window
     2019->today and trailing 2y / 1y / 180d — plus the same number on BID ONLY, so the
     quote-artifact component (bid − mid) is visible per leg. If |artifact| rivals the mid mean,
     the bid history is telling stories again.
  3. Writes a dated report + appends to edge_health_016_mid_history.csv (the old bid-based
     edge_health_016_history.csv is retired — do not compare across the two).

Live book note: session A (hour 21) was REMOVED from the live strategies on 2026-07-15 (artifact
leg). Hour 21 is still reported for monitoring, marked live='—'. Verdicts apply to live legs only.

This script CHANGES NOTHING live; a human reads it (data-hygiene.md; PROTOCOL §6).
"""
import lzma
import os
import struct
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

D = r"C:\Users\User\backtest_engine\backtest_engine2\data"
LOGFILE = os.path.join(D, "edge_health_016.out")
if sys.stdout is None:                      # pythonw: no console (see ops runbook)
    sys.stdout = sys.stderr = open(LOGFILE, "a", encoding="utf-8")

import httpx

PAIRS = {"EURUSD": 1e5, "GBPUSD": 1e5, "AUDUSD": 1e5, "NZDUSD": 1e5, "USDJPY": 1e3}
LIVE = {23: "BUY"}                          # hour 21 removed 2026-07-15 (artifact leg)
MONITOR_HOURS = (21, 23)
START = pd.Timestamp("2019-01-01")


def pull_days(sym, scale, d0, d1, feed):
    """Dukascopy 1-min candles (BID or ASK) for [d0, d1] -> hourly closes."""
    base = f"https://datafeed.dukascopy.com/datafeed/{sym}"
    frames, ok = [], 0
    client = httpx.Client(timeout=30)
    for day in pd.date_range(d0, d1, freq="D"):
        if day.weekday() == 5:
            continue
        url = f"{base}/{day.year}/{day.month - 1:02d}/{day.day:02d}/{feed}_candles_min_1.bi5"
        content = None
        for attempt in range(3):
            try:
                r = client.get(url)
                if r.status_code == 200 and len(r.content) > 0:
                    content = r.content
                break
            except Exception:
                time.sleep(2 * (attempt + 1))
        if content is None:
            continue
        try:
            raw = lzma.decompress(content)
        except lzma.LZMAError:
            continue
        n = len(raw) // 24
        recs = [struct.unpack(">5if", raw[i * 24:(i + 1) * 24]) for i in range(n)]
        df = pd.DataFrame(recs, columns=["sec", "open", "close", "low", "high", "vol"])
        df = df[df["vol"] > 0]
        if df.empty:
            continue
        df["open_time"] = day.tz_localize("UTC") + pd.to_timedelta(df["sec"], unit="s")
        df["close"] = df["close"] / scale
        frames.append(df[["open_time", "close"]])
        ok += 1
        time.sleep(0.05)
    client.close()
    if not frames:
        return None
    m1 = pd.concat(frames, ignore_index=True).set_index("open_time").sort_index()
    h1 = m1.resample("1h").agg({"close": "last"}).dropna().reset_index()
    h1["symbol"] = sym
    print(f"   {sym} {feed}: pulled {ok} days -> {len(h1)} hourly bars (to {h1.open_time.max()})")
    return h1


def _norm(h1):
    h1["open_time"] = pd.to_datetime(h1["open_time"], utc=True)
    return (h1.drop_duplicates(subset="open_time", keep="last")
              .sort_values("open_time").reset_index(drop=True))


def load_or_build_ask_h1(sym):
    """{sym}_h1_ask.parquet: assembled once from x016k hist (2019-22) + resampled 1m ask (2023+)."""
    p = os.path.join(D, f"{sym.lower()}_h1_ask.parquet")
    if os.path.exists(p):
        return _norm(pd.read_parquet(p)), p
    hist_p = os.path.join(D, f"{sym.lower()}_ask_h1_hist.parquet")
    m1_p = os.path.join(D, f"{sym.lower()}_ask_1m.parquet")
    if not os.path.exists(hist_p):
        raise FileNotFoundError(f"{sym}: {hist_p} missing — run x016k_pull_ask_h1.py {sym} first")
    parts = [pd.read_parquet(hist_p)[["open_time", "close", "symbol"]]]
    if os.path.exists(m1_p):
        m1 = pd.read_parquet(m1_p).set_index("open_time").sort_index()
        h1 = m1["close"].resample("1h").last().dropna().reset_index()
        h1["symbol"] = sym
        parts.append(h1)
    h1 = _norm(pd.concat(parts, ignore_index=True))
    h1.to_parquet(p, index=False)
    print(f"   {sym}: built h1_ask ({len(h1):,} bars, {h1.open_time.min().date()} -> "
          f"{h1.open_time.max().date()})")
    return h1, p


def extend(sym, scale):
    """Extend BOTH hourly feeds to today; return aligned DataFrame with bid/ask/mid closes."""
    bid_p = os.path.join(D, f"{sym.lower()}_h1.parquet")           # legacy file = BID closes
    bid = _norm(pd.read_parquet(bid_p))
    ask, ask_p = load_or_build_ask_h1(sym)
    today = pd.Timestamp.utcnow().normalize().tz_localize(None)
    for h1, p, feed in ((bid, bid_p, "BID"), (ask, ask_p, "ASK")):
        last = h1.open_time.max()
        if last.tz_localize(None) < today - pd.Timedelta(days=1):
            new = pull_days(sym, scale, (last + pd.Timedelta(hours=1)).tz_localize(None).normalize(),
                            today, feed)
            if new is not None:
                h1 = _norm(pd.concat([h1, new], ignore_index=True))
                h1.to_parquet(p, index=False)
        if feed == "BID":
            bid = h1
        else:
            ask = h1
    b = bid.set_index("open_time")["close"]
    a = ask.set_index("open_time")["close"]
    idx = b.index.intersection(a.index)
    df = pd.DataFrame({"bid": b.loc[idx], "ask": a.loc[idx]})
    df = df[(df.ask >= df.bid) & (df.bid > 0)]
    df["mid"] = (df.bid + df.ask) / 2
    return df[df.index >= START.tz_localize("UTC")]


def hour_returns(s):
    """Consecutive-hour returns by London hour (weekend/gap hours dropped)."""
    r = s.pct_change()
    r = r[(s.index.to_series().diff() == pd.Timedelta("1h")).values]
    ldn = r.index.tz_convert("Europe/London")
    return pd.DataFrame({"ret": r.values, "hr": ldn.hour}, index=r.index)


def mt(x):
    x = x.dropna()
    if len(x) < 20 or x.std() == 0:
        return np.nan, np.nan, len(x)
    return 1e4 * x.mean(), x.mean() / x.std() * np.sqrt(len(x)), len(x)


def run():
    now = pd.Timestamp.utcnow()
    stamp = f"{datetime.now():%Y-%m-%d}"
    head = f"#016 EDGE-HEALTH {stamp} — MID-based (v2; artifact column = bid − mid, watch it)"
    print(f"\n{head}\nextending bid+ask hourly data to today (Dukascopy)...")
    hdr = (f"{'pair':>7} {'hr':>3} {'live':>5} | {'MID full':>9} {'t':>6} {'n':>6} | "
           f"{'2y':>7} {'1y':>7} {'180d':>7} | {'artif':>7} | verdict")
    lines = [head, "", hdr, "-" * len(hdr)]
    rows = []
    for sym, scale in PAIRS.items():
        df = extend(sym, scale)
        dm = hour_returns(df["mid"])
        db = hour_returns(df["bid"])
        for hh in MONITOR_HOURS:
            xm = dm.loc[dm.hr == hh, "ret"]
            xb = db.loc[db.hr == hh, "ret"]
            full_m, full_t, n = mt(xm)
            m2y, _, _ = mt(xm[xm.index >= now - pd.Timedelta(days=730)])
            m1y, _, _ = mt(xm[xm.index >= now - pd.Timedelta(days=365)])
            m180, _, _ = mt(xm[xm.index >= now - pd.Timedelta(days=180)])
            bid_full, _, _ = mt(xb)
            artif = bid_full - full_m                     # quote-dynamics component of the bid number
            live = LIVE.get(hh, "—")
            if live == "—":
                verdict = "(not traded)"
            else:
                sg = +1 if live == "BUY" else -1
                if np.sign(m1y) != sg:
                    verdict = "FLIPPED"
                elif abs(m1y) < 0.4 * abs(full_m):
                    verdict = "WEAKENING"
                else:
                    verdict = "OK"
            line = (f"{sym:>7} {hh:>3} {live:>5} | {full_m:>+9.2f} {full_t:>+6.1f} {n:>6} | "
                    f"{m2y:>+7.2f} {m1y:>+7.2f} {m180:>+7.2f} | {artif:>+7.2f} | {verdict}")
            lines.append(line)
            print(line)
            rows.append({"date": stamp, "pair": sym, "hour": hh, "live": live,
                         "mid_full_bp": round(full_m, 3), "full_t": round(full_t, 2), "n": n,
                         "mid_2y": round(m2y, 3), "mid_1y": round(m1y, 3), "mid_180d": round(m180, 3),
                         "artifact_bp": round(artif, 3), "verdict": verdict})
    lines += ["", "read: MID columns are the tradable truth. 'artif' = what the bid-only number adds",
              "on top — quote mechanics, not price (hour 21 history: −0.4..−1.3; hour 23: +0.6..+1.6).",
              "Decay = mid trailing-1y fading across months (see mid_history csv). Changes are HUMAN."]
    with open(os.path.join(D, f"edge_health_016_{stamp}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    hist = os.path.join(D, "edge_health_016_mid_history.csv")
    pd.DataFrame(rows).to_csv(hist, mode="a", header=not os.path.exists(hist), index=False)
    print(f"\nwrote edge_health_016_{stamp}.txt + appended {len(rows)} rows to mid_history csv")
    flags = [r for r in rows if r["verdict"] not in ("OK", "(not traded)")]
    if flags:
        print(f"ATTENTION: {len(flags)} live leg(s) not OK: " +
              ", ".join(f"{r['pair']}@{r['hour']}={r['verdict']}" for r in flags))
    return 0


if __name__ == "__main__":
    sys.exit(run())

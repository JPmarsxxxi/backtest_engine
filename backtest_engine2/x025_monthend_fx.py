"""
x025 — Month-end FX rebalancing flow (Melvin-Prins 2015). Mechanical-flow family (like #016).

Hypothesis: high prior-month global-equity return -> USD bought into the month-end London 4pm
fix (funds re-hedge now-larger foreign equity holdings) -> EUR/GBP weaken into 16:00 London on
the last business day, often reversing after. Swap-free (intraday/last-day), FX majors.

Step 3 = is there ANY signal. Test, on the last trading day of each month:
  (a) UNCONDITIONAL month-end drift into the fix (is there a calendar effect at all?)
  (b) CONDITIONAL on prior-month S&P return sign (the actual Melvin-Prins mechanism).
Entry hours swept; exit at 16:00 London (the fix). Decay-first: data is 2019+ already.
"""
import numpy as np, pandas as pd
from scipy import stats

spx = pd.read_parquet("data/rates.parquet")["SPX"].dropna()
spx.index = pd.to_datetime(spx.index)
# prior-CALENDAR-month return, indexed by month
m_close = spx.resample("ME").last()
prior_m_ret = m_close.pct_change()              # return of month t (known at end of month t)

def load_fx(sym):
    d = pd.read_parquet(f"data/{sym}.parquet")[["open_time", "close"]].copy()
    t = pd.to_datetime(d["open_time"]).dt.tz_convert("Europe/London")
    d["date"] = t.dt.date
    d["hour"] = t.dt.hour
    d["ym"] = t.dt.to_period("M")
    return d[["date", "hour", "ym", "close"]].reset_index(drop=True)

def last_trading_days(d):
    last = d.groupby("ym")["date"].max()
    return last   # Period -> last date

def price_at(d_day, hour):
    """close of latest bar at/just before `hour` on this single-day frame."""
    sl = d_day[d_day["hour"] <= hour]
    return sl["close"].iloc[-1] if len(sl) else np.nan

FIX = 16   # 16:00 London fix
for sym in ["eurusd_h1", "gbpusd_h1"]:
    d = load_fx(sym)
    last = last_trading_days(d)
    by_date = {dt: g for dt, g in d.groupby("date")}
    rows = []
    for ym, day in last.items():
        # prior-month equity return = the month that just ended (this month's, known now)
        sel = prior_m_ret[prior_m_ret.index.to_period("M") == ym]
        eq = sel.iloc[0] if len(sel) else np.nan
        d_day = by_date.get(day)
        if d_day is None:
            continue
        fixpx = price_at(d_day, FIX)
        for h in [9, 11, 13]:
            ep = price_at(d_day, h)
            if np.isnan(ep) or np.isnan(fixpx):
                continue
            ret = fixpx / ep - 1.0          # EUR/GBP return from hour h into the fix
            rows.append({"day": day, "ym": ym, "eq": eq, "entry_h": h, "ret": ret})
    df = pd.DataFrame(rows).dropna(subset=["ret"])
    print(f"\n===== {sym}  (n month-ends ~{df['ym'].nunique()}) =====")

    print("(a) UNCONDITIONAL: mean EUR/GBP return into the 16:00 fix, by entry hour")
    for h, g in df.groupby("entry_h"):
        t = stats.ttest_1samp(g["ret"], 0)
        print(f"   entry {h:02d}:00->16:00  mean={g['ret'].mean()*1e4:+6.2f}bp  t={t.statistic:+4.1f}  n={len(g)}")

    print("(b) CONDITIONAL on prior-month equity sign (Melvin-Prins: eq UP -> USD up -> EUR/GBP DOWN)")
    for h, g in df.groupby("entry_h"):
        g = g.dropna(subset=["eq"])
        up = g[g["eq"] > 0]["ret"]; dn = g[g["eq"] <= 0]["ret"]
        # signal-aligned trade: short EUR when eq up, long when eq down => pnl = -sign(eq)*ret
        pnl = -np.sign(g["eq"]) * g["ret"]
        t = stats.ttest_1samp(pnl.dropna(), 0)
        print(f"   entry {h:02d}:00  eqUP_ret={up.mean()*1e4:+5.1f}bp  eqDN_ret={dn.mean()*1e4:+5.1f}bp  "
              f"|  TRADE mean={pnl.mean()*1e4:+5.1f}bp  t={t.statistic:+4.1f}  hit={(pnl>0).mean()*100:4.0f}%")

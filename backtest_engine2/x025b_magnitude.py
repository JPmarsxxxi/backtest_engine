"""
x025b — the ONE motivated refinement for #025 month-end FX rebalancing.

Melvin-Prins: the flow scales with the SIZE of the equity move (bigger move -> bigger
re-hedge). Pre-specified test (no sweep):
  - trade only LARGE prior-month equity moves (|eq| > median).
  - direction = USD-buying when equities rose: short EUR/short GBP/long USDJPY.
  - window A: approach to fix 11:00->16:00 London (entry).
  - window B: post-fix reversal 16:00->20:00 London (Melvin-Prins note a reversal).
  - 3 pairs + equal-weight combined book. Report mean bp, t, hit; large vs small split.
"""
import numpy as np, pandas as pd
from scipy import stats

spx = pd.read_parquet("data/rates.parquet")["SPX"].dropna()
spx.index = pd.to_datetime(spx.index)
prior_m_ret = spx.resample("ME").last().pct_change()

def load_fx(sym):
    d = pd.read_parquet(f"data/{sym}.parquet")[["open_time", "close"]].copy()
    t = pd.to_datetime(d["open_time"]).dt.tz_convert("Europe/London")
    d["date"] = t.dt.date; d["hour"] = t.dt.hour; d["ym"] = t.dt.to_period("M")
    return d[["date", "hour", "ym", "close"]].reset_index(drop=True)

def px_at(d_day, hour):
    sl = d_day[d_day["hour"] <= hour]
    return sl["close"].iloc[-1] if len(sl) else np.nan

# USD-buying trade sign per pair: EUR/GBP short (=-1*ret), USDJPY long (=+1*ret)
USD_SIGN = {"eurusd_h1": -1.0, "gbpusd_h1": -1.0, "usdjpy_h1": +1.0}
WINDOWS = {"A approach 11->16": (11, 16), "B postfix 16->20": (16, 20)}

panel = {}   # (pair,window) -> df
for sym, usd in USD_SIGN.items():
    d = load_fx(sym)
    last = d.groupby("ym")["date"].max()
    by_date = {dt: g for dt, g in d.groupby("date")}
    for wlab, (h0, h1) in WINDOWS.items():
        rows = []
        for ym, day in last.items():
            sel = prior_m_ret[prior_m_ret.index.to_period("M") == ym]
            eq = sel.iloc[0] if len(sel) else np.nan
            g = by_date.get(day)
            if g is None or np.isnan(eq): continue
            p0, p1 = px_at(g, h0), px_at(g, h1)
            if np.isnan(p0) or np.isnan(p1): continue
            ret = p1 / p0 - 1.0
            pnl = usd * np.sign(eq) * ret          # USD-buying trade aligned to equity sign
            rows.append({"ym": ym, "eq": eq, "absq": abs(eq), "pnl": pnl})
        panel[(sym, wlab)] = pd.DataFrame(rows)

def stat(pnl):
    pnl = pnl.dropna()
    if len(pnl) < 8: return (np.nan, np.nan, np.nan, len(pnl))
    t = stats.ttest_1samp(pnl, 0).statistic
    return (pnl.mean()*1e4, t, (pnl>0).mean()*100, len(pnl))

for wlab in WINDOWS:
    print(f"\n===== window {wlab} =====")
    print(f"{'pair':12s} {'ALL months':>22s} {'LARGE |eq|':>22s} {'small |eq|':>22s}")
    combined_large, combined_all = [], []
    for sym in USD_SIGN:
        df = panel[(sym, wlab)].copy()
        med = df["absq"].median()
        big = df[df["absq"] > med]; small = df[df["absq"] <= med]
        a = stat(df["pnl"]); b = stat(big["pnl"]); c = stat(small["pnl"])
        print(f"{sym:12s} "
              f"{a[0]:+5.1f}bp t{a[1]:+4.1f} h{a[2]:3.0f} (n{a[3]:2d}) "
              f"{b[0]:+5.1f}bp t{b[1]:+4.1f} h{b[2]:3.0f} (n{b[3]:2d}) "
              f"{c[0]:+5.1f}bp t{c[1]:+4.1f} h{c[2]:3.0f} (n{c[3]:2d})")
        combined_all.append(df.set_index("ym")["pnl"].rename(sym))
        combined_large.append(big.set_index("ym")["pnl"].rename(sym))
    # equal-weight combined book (large-move months)
    cl = pd.concat(combined_large, axis=1).mean(axis=1).dropna()
    ca = pd.concat(combined_all, axis=1).mean(axis=1).dropna()
    sa, sl = stat(ca), stat(cl)
    print(f"{'COMBINED':12s} "
          f"{sa[0]:+5.1f}bp t{sa[1]:+4.1f} h{sa[2]:3.0f} (n{sa[3]:2d}) "
          f"{sl[0]:+5.1f}bp t{sl[1]:+4.1f} h{sl[2]:3.0f} (n{sl[3]:2d})  <- large-move book")

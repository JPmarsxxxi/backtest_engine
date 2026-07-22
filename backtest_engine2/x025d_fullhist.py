"""
x025d — #025 month-end FX rebalancing on FULL history (2003-2026, *_h1_full).
Pre-committed test (from x025b): large prior-month |equity| moves, USD-buying trade
(short EUR/short GBP/long USDJPY), approach-to-fix window 11:00->16:00 London.
Split by ERA: 2003-2019 (OOS vs the 2019+ lean) vs 2019-2026. Decision rule pre-set:
CONFIRM if large-move book t>2 AND EUR+JPY magnitude lean holds in BOTH eras; else KILL.
Also reports window B (post-fix) for completeness.
"""
import numpy as np, pandas as pd
from scipy import stats

spx = pd.read_parquet("data/rates.parquet")["SPX"].dropna()
spx.index = pd.to_datetime(spx.index)
prior_m_ret = spx.resample("ME").last().pct_change()

def load_fx(path):
    d = pd.read_parquet(path)[["open_time", "close"]].copy()
    t = pd.to_datetime(d["open_time"]).dt.tz_convert("Europe/London")
    d["date"] = t.dt.date; d["hour"] = t.dt.hour; d["ym"] = t.dt.to_period("M")
    return d[["date", "hour", "ym", "close"]].reset_index(drop=True)

def px_at(g, hour):
    sl = g[g["hour"] <= hour]
    return sl["close"].iloc[-1] if len(sl) else np.nan

USD_SIGN = {"eurusd": -1.0, "gbpusd": -1.0, "usdjpy": +1.0}
WINDOWS  = {"A approach 11->16": (11, 16), "B postfix 16->20": (16, 20)}

panel = {}
for sym, usd in USD_SIGN.items():
    d = load_fx(f"data/{sym}_h1_full.parquet")
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
            rows.append({"ym": ym, "year": ym.year, "absq": abs(eq),
                         "pnl": usd * np.sign(eq) * (p1/p0 - 1.0)})
        panel[(sym, wlab)] = pd.DataFrame(rows)

def stat(pnl):
    pnl = pnl.dropna()
    if len(pnl) < 8: return (np.nan, np.nan, np.nan, len(pnl))
    return (pnl.mean()*1e4, stats.ttest_1samp(pnl, 0).statistic, (pnl>0).mean()*100, len(pnl))

def fmt(s): return f"{s[0]:+5.1f}bp t{s[1]:+4.1f} h{s[2]:3.0f} (n{s[3]:3d})"

ERAS = {"FULL 03-26": (2003, 2027), "OOS 03-19": (2003, 2019), "recent 19-26": (2019, 2027)}
for wlab in WINDOWS:
    print(f"\n========== window {wlab} — LARGE |eq| months (USD-buying trade) ==========")
    print(f"{'':12s} " + " ".join(f"{e:>26s}" for e in ERAS))
    bigbooks = {e: [] for e in ERAS}
    for sym in USD_SIGN:
        df = panel[(sym, wlab)].copy()
        med = df["absq"].median()              # magnitude split on full-sample median
        big = df[df["absq"] > med]
        cells = []
        for e, (lo, hi) in ERAS.items():
            sub = big[(big.year >= lo) & (big.year < hi)]
            cells.append(fmt(stat(sub["pnl"])))
            bigbooks[e].append(sub.set_index("ym")["pnl"].rename(sym))
        print(f"{sym:12s} " + " ".join(f"{c:>26s}" for c in cells))
    combo = []
    for e in ERAS:
        cl = pd.concat(bigbooks[e], axis=1).mean(axis=1).dropna()
        combo.append(fmt(stat(cl)))
    print(f"{'COMBINED':12s} " + " ".join(f"{c:>26s}" for c in combo))

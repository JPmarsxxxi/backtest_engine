"""#016h - ENHANCED book test (v3 candidate). Two fixes for the fragile USDJPY leg:
(1) add AUDUSD + NZDUSD as ECONOMICALLY-CONSISTENT legs (USD-quote pairs -> same 21:short/23:long as
    EUR/GBP if the effect is USD-strength-at-NY-fix; honest breadth, not fragile -0.4-corr diversification);
(2) test whether USDJPY's 21:short is REGIME-DEPENDENT (only works in risk-off / high VIX).
Rebuild combined Sharpe with clean legs + JPY dropped or regime-conditioned. Same engine as x016e.
"""
import numpy as np
import pandas as pd

ENG = r"C:\Users\User\backtest_engine\backtest_engine2"
# USD-quote pairs (EUR/GBP/AUD/NZD): USD strength -> pair DOWN -> 21:short/23:long expected.
# USD-base (JPY): USD strength -> pair UP -> 21:long/23:short expected (the "USD-driven" prediction).
EXPECT = {"EURUSD": {21: -1, 23: +1}, "GBPUSD": {21: -1, 23: +1},
          "AUDUSD": {21: -1, 23: +1}, "NZDUSD": {21: -1, 23: +1},
          "USDJPY": {21: +1, 23: -1}}
HALF = {"EURUSD": {21: 0.087, 23: 0.086}, "GBPUSD": {21: 0.187, 23: 0.187},
        "USDJPY": {21: 0.156, 23: 0.125},
        "AUDUSD": {21: 0.288, 23: 0.218}, "NZDUSD": {21: 0.531, 23: 0.529}}  # MT5-MEASURED London-hr 2026-07-12

def load_close(sym):
    import os
    f = rf"{ENG}\data\{sym.lower()}_h1.parquet"
    if os.path.exists(f):
        s = pd.read_parquet(f).set_index("open_time")["close"].sort_index()
    else:                                                # resample M15 -> H1 close
        m = pd.read_parquet(rf"{ENG}\data\{sym.lower()}_m15.parquet").set_index("open_time")["close"].sort_index()
        s = m.resample("1h").last().dropna()
    return s

def hourly(sym):
    s = load_close(sym)
    r = s.pct_change()
    r = r[(s.index.to_series().diff() == pd.Timedelta("1h")).values]
    ldn = r.index.tz_convert("Europe/London")
    d = pd.DataFrame({"ret": r.values, "hr": ldn.hour, "yr": ldn.year}, index=r.index)
    d["day"] = pd.to_datetime(pd.Series(ldn.date, index=d.index))
    return d

def sh(x):
    return x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0

PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY"]
D = {s: hourly(s) for s in PAIRS}

print(f"{'pair':>7} {'hr':>3} {'trainBp':>8} {'t':>6} {'fitSgn':>6} {'econ':>5} {'match':>6} {'testBp':>8} {'t':>6} {'OOS':>5}")
signs = {}
for sym in PAIRS:
    d = D[sym]; tr, te = d[d.yr <= 2021], d[d.yr >= 2022]; signs[sym] = {}
    for hh in (21, 23):
        xtr, xte = tr.loc[tr.hr == hh, "ret"], te.loc[te.hr == hh, "ret"]
        sg = int(np.sign(xtr.mean())); signs[sym][hh] = sg
        ttr = xtr.mean()/xtr.std()*np.sqrt(len(xtr)); tte = xte.mean()/xte.std()*np.sqrt(len(xte))
        econ = EXPECT[sym][hh]; match = "OK" if sg == econ else "**BACKWARDS**"
        oos = "OK" if np.sign(xte.mean()) == sg else "FLIP"
        print(f"{sym:>7} {hh:>3} {1e4*xtr.mean():>+7.2f} {ttr:>+6.1f} {sg:>+6d} {econ:>+5d} {match:>6} "
              f"{1e4*xte.mean():>+7.2f} {tte:>+6.1f} {oos:>5}")

def book(pairs, sign_override=None):
    parts = {}
    for sym in pairs:
        te = D[sym][D[sym].yr >= 2022]; bk = te[te.hr.isin([21, 23])].copy()
        sg = sign_override[sym] if sign_override and sym in sign_override else signs[sym]
        bk["sret"] = [sg[h] for h in bk.hr] * bk["ret"]
        bk["cost"] = [2 * HALF[sym][h] / 1e4 for h in bk.hr]
        parts[sym] = (bk.groupby("day")["sret"].sum() - bk.groupby("day")["cost"].sum())
    comb = pd.concat(parts.values(), axis=1).dropna().sum(axis=1)
    return comb

def report(name, comb):
    print(f"  {name:28}: net Sh {sh(comb):+.2f} | {1e4*comb.mean():+.2f} bp/day | "
          f"hit {100*(comb>0).mean():.0f}% | ann {comb.mean()*252*100:+.1f}% | n={len(comb)}")

print("\n=== COMBINED BOOKS (OOS 2022-26, NET est. spreads) ===")
report("ORIG 3-pair (EUR/GBP/JPY)", book(["EURUSD", "GBPUSD", "USDJPY"]))
report("CLEAN 4-pair (EUR/GBP/AUD/NZD)", book(["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"]))
report("5-pair (clean-4 + JPY)", book(PAIRS))
report("3-pair clean (EUR/GBP/AUD)", book(["EURUSD", "GBPUSD", "AUDUSD"]))

# --- VIX regime test for USDJPY 21:00 ---
print("\n=== USDJPY 21:00 regime test (is the short only a risk-off effect?) ===")
vix = pd.read_parquet(rf"{ENG}\data\regime_signals.parquet")["VIX"]
vix.index = pd.to_datetime(vix.index)
j = D["USDJPY"]; j21 = j[(j.hr == 21) & (j.yr >= 2022)].copy()
j21["vix"] = vix.reindex(j21["day"].values).values
j21 = j21.dropna(subset=["vix"])
med = j21["vix"].median()
for lab, mask in [("HIGH VIX (risk-off)", j21.vix > med), ("LOW VIX (risk-on)", j21.vix <= med)]:
    x = j21.loc[mask, "ret"]
    print(f"  {lab:22}: mean {1e4*x.mean():+.2f} bp  t {x.mean()/x.std()*np.sqrt(len(x)):+.1f}  "
          f"n={len(x)}  (short profits if mean<0)")

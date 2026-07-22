# #020 — Index intraday seasonality (US500) decay-first gate.
# Hypothesis: unconditional mean return by clock-hour (New York) is non-zero at fixed times
#   because of recurring mechanical session flows (cash open auction, lunch lull, close rebalance).
# DISTINCT from killed #012 (conditional first-move->last-move momentum) and #013 (conditional
#   close-imbalance -> overnight drift). This is the unconditional time-of-day cousin of #016.
# US500 CFD intraday -> swap only if held overnight; an hour-bucket book is intraday/flat -> ~no swap.
# Pre-registered survive rule (mirror #016): a clear, RECENT (2022+), significant hour-of-day
#   pattern that clears Bonferroni (n_hours -> |t|>~2.8) AND clears the US500 intraday spread
#   (~0.25bp half = 0.5bp round trip). Weekend/session-break gaps dropped (consecutive-hour only).
import numpy as np
import pandas as pd

ENG = r"C:\Users\User\backtest_engine\backtest_engine2"
s = pd.read_parquet(ENG + r"\data\us500_1m.parquet").set_index("open_time")["close"].sort_index()

# hourly close in New York time (handles DST); last 1m close in each ET hour
ny = s.tz_convert("America/New_York")
hourly = ny.resample("1h").last().dropna()
ret = hourly.pct_change()
dt_hours = hourly.index.to_series().diff().dt.total_seconds() / 3600.0
# keep only consecutive-hour returns (drop weekend gaps + the ~1h daily settlement break overhang)
consec = dt_hours <= 1.5
hr = hourly.index.hour

print(f"US500 hourly bars {len(hourly)} | {hourly.index[0]} -> {hourly.index[-1]} (New York)")
print(f"consecutive-hour returns kept: {int(consec.sum())} / {len(ret)-1}")

def hour_profile(year_mask, label, bonf_t):
    m = consec & year_mask & ret.notna()
    r = ret[m]
    h_ = pd.Series(hr, index=hourly.index)[m]
    print(f"\n[{label}] mean return by NY hour (bps, t-stat)  Bonferroni |t|>{bonf_t:.2f}:")
    surv = []
    for hh in range(24):
        x = r[h_.values == hh].dropna()
        if len(x) < 100:
            continue
        t = x.mean() / x.std() * np.sqrt(len(x))
        flag = ""
        if abs(t) > bonf_t:
            flag = " <<< BONF"
            surv.append((hh, 1e4 * x.mean(), t))
        elif abs(t) > 2:
            flag = " <"
        print(f"   {hh:02d}:00 NY  {1e4*x.mean():+6.3f} bps  t={t:+5.1f}  (n={len(x)}){flag}")
    return surv

# count valid hours for Bonferroni
nh = sum(1 for hh in range(24) if (consec & ret.notna() & (pd.Series(hr,index=hourly.index)==hh)).sum() >= 100)
bonf = abs(__import__("scipy.stats", fromlist=["norm"]).norm.ppf(0.025/nh))
print(f"valid hour buckets: {nh}  ->  Bonferroni |t| threshold {bonf:.2f}")

surv_full = hour_profile(pd.Series(hourly.index.year >= 2019, index=hourly.index), "full 2019-2026", bonf)
surv_rec  = hour_profile(pd.Series(hourly.index.year >= 2022, index=hourly.index), "2022-2026 (decay check)", bonf)

print("\n=== GATE ===")
full_h = {h for h,_,_ in surv_full}
rec_h  = {h for h,_,_ in surv_rec}
both = full_h & rec_h
print(f"Bonferroni survivors full: {sorted(full_h)}")
print(f"Bonferroni survivors 2022+: {sorted(rec_h)}")
print(f"survive BOTH (recent-persistent): {sorted(both)}")
for h,bps,t in surv_rec:
    if h in both:
        side = "LONG" if bps>0 else "SHORT"
        print(f"   {h:02d}:00 NY  {side}  {bps:+.3f} bps (recent t={t:+.1f}) -- clears 0.5bp r/t? {abs(bps)>0.5}")

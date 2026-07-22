# #016 — FX intraday seasonality decay gate (EURUSD hourly). Two sub-hypotheses:
# (a) Breedon-Ranaldo local-hours: a currency depreciates in its local business hours
#     -> EURUSD falls in EU hours, rises in US hours.
# (b) Krohn London-fix reversal: returns run up into 16:00 London then reverse after.
# FX intraday = NO SWAP (flat by close). Spread ~0.1bp half (EURUSD, very tight) -> low cost gate.
# Pre-registered: survives iff a clear, RECENT (2022+), significant (|t|>2) hour-of-day pattern
# clears ~0.2bp round-trip spread. London local time handles DST.
import numpy as np
import pandas as pd

ENG = r"C:\Users\User\backtest_engine\backtest_engine2"
h = pd.read_parquet(ENG + r"\data\eurusd_h1.parquet")
s = h.set_index("open_time")["close"].sort_index()
ldn = s.index.tz_convert("Europe/London")
ret = s.pct_change()
hr = ldn.hour

print(f"EURUSD hourly bars {len(s)} | {s.index[0].date()} -> {s.index[-1].date()}")

def hour_profile(mask, label):
    r = ret[mask]
    h_ = pd.Series(hr[mask], index=r.index)
    print(f"\n[{label}] mean return by London hour (bps, t-stat):")
    rows = []
    for hh in range(24):
        x = r[h_.values == hh].dropna()
        if len(x) < 50:
            continue
        t = x.mean() / x.std() * np.sqrt(len(x))
        rows.append((hh, 1e4 * x.mean(), t, len(x)))
    for hh, bps, t, n in rows:
        flag = " <<<" if abs(t) > 2 else ""
        print(f"   {hh:02d}:00 London  {bps:+6.3f} bps  t={t:+5.1f}  (n={n}){flag}")

hour_profile(ret.index.year >= 2019, "full 2019-2026")
hour_profile(ret.index.year >= 2022, "2022-2026 (decay check)")

# (b) London-fix reversal: pre-fix run-up (14->16 London) vs post-fix (16->18 London), per day
df = pd.DataFrame({"ret": ret.values, "h": hr}, index=ret.index)
df["day"] = ldn.date
pre = df[(df.h >= 14) & (df.h < 16)].groupby("day")["ret"].sum()
post = df[(df.h >= 16) & (df.h < 18)].groupby("day")["ret"].sum()
j = pd.concat([pre.rename("pre"), post.rename("post")], axis=1).dropna()
j.index = pd.to_datetime(j.index)
for lab, m in [("full", j.index.year >= 2019), ("2022+", j.index.year >= 2022)]:
    jj = j[m]
    corr = jj["pre"].corr(jj["post"])
    # reversal trade: fade the pre-fix move over 16->18
    tr = (-np.sign(jj["pre"]) * jj["post"]).dropna()
    t = tr.mean() / tr.std() * np.sqrt(len(tr))
    print(f"\n[fix {lab}] corr(pre,post)={corr:+.3f} | fade-pre-fix gross {1e4*tr.mean():+.2f} bps/day "
          f"t={t:+.1f} hit {100*(tr>0).mean():.0f}% (n={len(tr)})")

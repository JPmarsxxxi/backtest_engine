# #016 final — 2-hour book {21:00 short, 23:00 long} with MEASURED per-hour spreads from MT5.
# 22:00 dropped (rollover spread 1.2bp half = untradeable). 21:00 & 23:00 measured ~0.087bp half.
import numpy as np
import pandas as pd

ENG = r"C:\Users\User\backtest_engine\backtest_engine2"
s = pd.read_parquet(ENG + r"\data\eurusd_h1.parquet").set_index("open_time")["close"].sort_index()
ret = s.pct_change()
ret = ret[(s.index.to_series().diff() == pd.Timedelta("1h")).values]   # consecutive hours only
ldn = ret.index.tz_convert("Europe/London")
df = pd.DataFrame({"ret": ret.values, "hr": ldn.hour, "yr": ldn.year}, index=ret.index)
df["day"] = pd.to_datetime(pd.Series(ldn.date, index=df.index))

BOOK = {21: -1.0, 23: +1.0}                       # signs from train 2019-2021
HALF = {21: 0.087, 23: 0.086}                     # MEASURED half-spread bp (MT5, 12d ticks)

t2 = df[df.yr >= 2022]
t2 = t2[t2.hr.isin(BOOK)].copy()
t2["sret"] = [BOOK[hh] for hh in t2.hr] * t2["ret"]
daily_gross = t2.groupby("day")["sret"].sum()
# per-day cost = sum over the day's traded hours of 2*half_bp (round trip)
cost_per_hr = {hh: 2 * HALF[hh] / 1e4 for hh in BOOK}
daily_cost = t2.assign(c=[cost_per_hr[hh] for hh in t2.hr]).groupby("day")["c"].sum()
daily_net = (daily_gross - daily_cost).dropna()

def sh(x):
    return x.mean() / x.std() * np.sqrt(252)

print(f"2-HOUR BOOK {{21:short, 23:long}} | TEST 2022-2026 | {len(daily_net)} days | MEASURED spreads")
print(f"  GROSS {1e4*daily_gross.mean():+.2f} bp/day Sh {sh(daily_gross):+.2f} | "
      f"NET {1e4*daily_net.mean():+.2f} bp/day Sh {sh(daily_net):+.2f} | "
      f"hit {100*(daily_net>0).mean():.0f}% | ann {daily_net.mean()*252*100:+.1f}%")
print("  per-year NET (measured spreads):")
for yr, g in daily_net.groupby(daily_net.index.year):
    print(f"    {yr}: net {1e4*g.mean():+.2f} bp/day | Sh {sh(g):+.2f} | ann {g.mean()*252*100:+.1f}%")

# compare: what 22:00 would have cost (rollover)
g22 = df[(df.yr >= 2022) & (df.hr == 22)]["ret"]
print(f"\n  (22:00 dropped: gross {1e4*g22.mean():+.2f}bp/day vs rollover cost 2*1.205={2*1.205:.1f}bp -> "
      f"net {1e4*g22.mean()-2*1.205:+.1f}bp = untradeable, correctly excluded)")

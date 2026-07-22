# #016 deepening — OOS hour-portfolio + spread sensitivity (the make-or-break).
# Rigor fixes over the first gate: (1) drop weekend-gap returns (prior bar must be exactly 1h before -
#     the 23:00 effect could be a Sunday-open artifact); (2) select hours on TRAIN 2019-2021 only,
#     validate on TEST 2022-2026 (no in-sample hour cherry-picking - the trap that sank #017b).
# Portfolio: each train-significant hour traded daily with its train sign; net = gross - n_hours*2*half_bp.
import numpy as np
import pandas as pd

ENG = r"C:\Users\User\backtest_engine\backtest_engine2"
h = pd.read_parquet(ENG + r"\data\eurusd_h1.parquet")
s = h.set_index("open_time")["close"].sort_index()
ret = s.pct_change()
consec = s.index.to_series().diff() == pd.Timedelta("1h")    # only consecutive-hour returns
ret = ret[consec.values]
ldn = ret.index.tz_convert("Europe/London")
df = pd.DataFrame({"ret": ret.values, "hr": ldn.hour, "yr": ldn.year}, index=ret.index)

train = df[df.yr <= 2021]
test = df[df.yr >= 2022]

# select hours on TRAIN (|t|>2), record sign
sel = {}
print("hour selection on TRAIN 2019-2021 (|t|>2 kept):")
for hh in range(24):
    x = train.loc[train.hr == hh, "ret"]
    if len(x) < 50:
        continue
    t = x.mean() / x.std() * np.sqrt(len(x))
    if abs(t) > 2:
        sel[hh] = np.sign(x.mean())
        # test-side check
        xt = test.loc[test.hr == hh, "ret"]
        tt = xt.mean() / xt.std() * np.sqrt(len(xt))
        print(f"   {hh:02d}:00 London  train {1e4*x.mean():+.2f}bp t{t:+.1f} sign{int(sel[hh]):+d}  ->  "
              f"TEST {1e4*xt.mean():+.2f}bp t{tt:+.1f} {'OK' if np.sign(xt.mean())==sel[hh] else 'FLIP'}")

# build daily portfolio return on TEST (sum of signed selected-hour returns per day)
test = test.assign(day=ldn[df.yr.values >= 2022].date if False else pd.Series(ldn[ (df.yr.values>=2022) ].date, index=test.index))
test["day"] = pd.Series(test.index.tz_convert("Europe/London").date, index=test.index)
hours = list(sel)
def signed(row):
    return sel.get(row["hr"], 0.0) * row["ret"]
test["sret"] = [sel.get(hh, 0.0) for hh in test.hr] * test["ret"]
daily = test[test.hr.isin(hours)].groupby("day")["sret"].sum()
daily.index = pd.to_datetime(daily.index)
n_h = len(hours)

g_sh = daily.mean() / daily.std() * np.sqrt(252)
print(f"\nTEST 2022-2026 portfolio: {n_h} hours {sorted(hours)} | {len(daily)} days")
print(f"  GROSS: {1e4*daily.mean():+.2f} bp/day | Sharpe {g_sh:+.2f} | hit {100*(daily>0).mean():.0f}%")
print(f"\n  NET by half-spread (cost = {n_h} hrs x 2 x half_bp per day):")
for hb in [0.1, 0.2, 0.3, 0.5, 1.0]:
    net = daily - n_h * 2 * hb / 1e4
    nsh = net.mean() / net.std() * np.sqrt(252)
    print(f"    half {hb:.1f}bp (r/t {2*hb:.1f}): net {1e4*net.mean():+.2f} bp/day | Sharpe {nsh:+.2f} | "
          f"ann {net.mean()*252*100:+.1f}%")
print("\n  per-year TEST net @ half 0.3bp:")
net03 = daily - n_h * 2 * 0.3 / 1e4
for yr, gy in net03.groupby(net03.index.year):
    print(f"    {yr}: {1e4*gy.mean():+.2f} bp/day | Sh {gy.mean()/gy.std()*np.sqrt(252):+.2f} | ann {gy.mean()*252*100:+.1f}%")

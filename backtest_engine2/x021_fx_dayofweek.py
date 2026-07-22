# #021 — Day-of-week FX seasonality gate (EURUSD/GBPUSD/USDJPY).
# Hypothesis: a given weekday's return is predictably signed (weekend-risk positioning / Monday gap).
# Tested SWAP-FREE & gap-free: within-day open->close return (no overnight hold, no weekend gap),
#   the #016 spirit. Pre-registered survive rule: a weekday clears Bonferroni (5 days -> |t|>2.58)
#   AND holds recent (2022+) AND pooled cross-pair book reaches at least |t|>2.
# RESULT: KILLED. No pair-weekday clears Bonferroni; pooled risk book only t+1.75 (2022+), hit 53%.
import numpy as np, pandas as pd
from scipy.stats import norm

ENG = r"C:\Users\User\backtest_engine\backtest_engine2"
pairs = {"EURUSD": "eurusd_h1", "GBPUSD": "gbpusd_h1", "USDJPY": "usdjpy_h1"}
wd = ["Mon", "Tue", "Wed", "Thu", "Fri"]
bonf = abs(norm.ppf(0.025 / 5))

def intraday_by_day(fn):
    h = pd.read_parquet(f"{ENG}/data/{fn}.parquet").set_index("open_time")["close"].sort_index()
    ldn = h.tz_convert("Europe/London")
    df = pd.DataFrame({"px": ldn.values}, index=ldn.index)
    df["d"] = df.index.normalize()
    g = df.groupby("d")["px"].agg(["first", "last"])
    g["r"] = g["last"] / g["first"] - 1                       # within-day, no overnight / no gap
    g["wd"] = pd.to_datetime(g.index).dayofweek
    return g[g.wd < 5]

G = {nm: intraday_by_day(fn) for nm, fn in pairs.items()}
print(f"Bonferroni over 5 weekdays -> |t|>{bonf:.2f}\nSWAP-FREE within-day return by weekday (bps, t):")
for nm, g in G.items():
    for lab, yr in [("full", 2019), ("2022+", 2022)]:
        gg = g[pd.to_datetime(g.index).year >= yr]
        cells = [(wd[i], 1e4 * gg[gg.wd == i]["r"].mean(),
                  gg[gg.wd == i]["r"].mean() / gg[gg.wd == i]["r"].std() * np.sqrt(len(gg[gg.wd == i]))) for i in range(5)]
        print(f"  {nm} [{lab:5s}] " + " | ".join(f"{w}{b:+5.1f}(t{t:+.1f}){'*' if abs(t)>bonf else ''}" for w, b, t in cells))

print("\nPOOLED book Mon-long/Fri-short EUR&GBP, USDJPY sign-flipped (swap-free within-day):")
for lab, yr in [("full", 2019), ("2022+", 2022)]:
    streams = []
    for nm, g in G.items():
        gg = g[pd.to_datetime(g.index).year >= yr]; sgn = -1 if nm == "USDJPY" else +1
        streams.append(pd.concat([sgn * gg[gg.wd == 0]["r"], -sgn * gg[gg.wd == 4]["r"]]))
    allr = pd.concat(streams)
    print(f"  [{lab:5s}] {1e4*allr.mean():+.2f} bps/trade  t={allr.mean()/allr.std()*np.sqrt(len(allr)):+.2f}  "
          f"hit {100*(allr>0).mean():.0f}%  n={len(allr)}")

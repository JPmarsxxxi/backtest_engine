# #015 — BTC weekly TSM (Liu-Tsyvinski RFS 2021) decay+cost gate. Single-leg directional; crypto CFD
# swap 30%/yr PER SIDE = 0.577%/week drag regardless of long/short. Signal must avg >0.58%/wk to break
# even. Pre-registered K=3 (1/2/4-week lookback). Survive: NET positive AND majority years positive.
import numpy as np
import pandas as pd

ENG = r"C:\Users\User\backtest_engine\backtest_engine2"
b = pd.read_parquet(ENG + r"\data\btc_30m.parquet")
s = b.set_index("open_time")["close"].sort_index()
s.index = s.index.tz_localize(None)
daily = s.resample("1D").last().dropna()
wk = daily.resample("W-MON").last().dropna()
rw = wk.pct_change()
SWAP = 0.30 / 52.0                                  # per-week per-side drag

print(f"BTC weekly bars {len(wk)} | {wk.index[0].date()} -> {wk.index[-1].date()}")
print(f"swap {SWAP*100:.3f}%/wk | buy-hold: {rw.mean()*100:+.2f}%/wk Sh {rw.mean()/rw.std()*np.sqrt(52):+.2f}")
print(f"\n{'k':>3} {'wks':>4} {'gross%/wk':>10} {'grossSh':>8} {'NET%/wk':>9} {'netSh':>7} {'hit':>5} {'posYrs(net)':>11}")
for k in [1, 2, 4]:
    sig = np.sign(wk.pct_change(k)).shift(1)         # trailing k-wk sign, held next week (PIT)
    r = (sig * rw).dropna()
    r = r[sig.reindex(r.index) != 0]
    net = r - SWAP                                    # always in a position -> always pay swap
    shg = r.mean() / r.std() * np.sqrt(52)
    shn = net.mean() / net.std() * np.sqrt(52)
    yrs = sorted(set(net.index.year))
    posn = sum(1 for y in yrs if net[net.index.year == y].mean() > 0)
    print(f"{k:>3} {len(r):>4} {r.mean()*100:>+10.2f} {shg:>+8.2f} {net.mean()*100:>+9.2f} {shn:>+7.2f} "
          f"{100*(r>0).mean():>4.0f}% {posn:>5}/{len(yrs)}")
    py = {int(y): round(net[net.index.year == y].mean() * 100, 1) for y in yrs}
    print(f"      net %/wk by yr: {py}")

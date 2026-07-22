"""#029e - add a POINT-IN-TIME SPREAD GATE to the wide crossing pool (user's idea). Spread is known
at decision time, so only adjust a pair's position when its CURRENT half-spread <= cap; otherwise hold
(no trade). => every trade is executed at spread<=cap, so cost/trade is bounded. Not look-ahead: uses
only the contemporaneous ask-bid. Sweep cap on all 28 pairs, real Dukascopy spreads. lag1/lag2.

Mechanics: gated_pos = target.where(half<=cap).ffill()  -> position changes ONLY on tradable (tight) bars,
so |Δpos| is nonzero only when half<=cap. cost = |Δpos|*half (<=cap on every trade)."""
import numpy as np
import pandas as pd
import x029c_wide_crossing as c   # panel(mid), cols, idx, rets, books (incl #027 on 28 pairs)

cols, idx, rets = c.cols, c.idx, c.rets
BARS, ANN = c.BARS, c.ANN
mid = c.panel
ask = pd.read_parquet('data/fx_wide_ask_m15.parquet').sort_index()[cols]
half = ((ask - mid) / mid).clip(lower=0, upper=10e-4)     # real per-bar half-spread (frac)
Wc = sum(c.books.values()) / len(c.books)                 # crossed combined target

def gated_eval(target, lag, cap_bp):
    cap = cap_bp / 1e4
    tradable = half <= cap
    # position changes only on tradable bars; else hold last position
    gpos = target.where(tradable).ffill().fillna(0.0)
    gross = (gpos.shift(lag) * rets).sum(axis=1)
    dpos = (gpos - gpos.shift(1)).abs()
    cost = (dpos * half).sum(axis=1)                       # every trade at half<=cap by construction
    net = gross - cost
    turn = dpos.sum(axis=1).mean() * BARS
    frac_tradable = tradable.mean().mean()
    def sh(r): r = r.dropna(); return r.mean()/r.std()*ANN if r.std() > 0 else 0
    def pct(r): return r.dropna().mean()*BARS*252*100
    return dict(gsh=sh(gross), nsh=sh(net), gpct=pct(gross), npct=pct(net), turn=turn, ftr=frac_tradable, net=net)

print('#029e SPREAD-GATED crossing pool (28 pairs, real Dukascopy spread, point-in-time gate)')
print('only trade a pair when its current half-spread <= cap; else hold. cost bounded by cap per trade.\n')
for lag in (1, 2):
    print(f'=== lag{lag} ({"idealized" if lag==1 else "realistic"}) ===')
    # no-gate baseline
    base_g = (Wc.shift(lag)*rets).sum(axis=1); base_c = ((Wc-Wc.shift(1)).abs()*half).sum(axis=1)
    bn = (base_g-base_c).dropna(); bsh = bn.mean()/bn.std()*ANN
    print(f'  NO GATE           : NET Sh {bsh:+.2f}')
    for cap in (1.0, 0.5, 0.3, 0.2, 0.1, 0.05):
        r = gated_eval(Wc, lag, cap)
        print(f'  cap {cap:.2f}bp (trade {r["ftr"]*100:4.1f}% of pair-bars): '
              f'NET Sh {r["nsh"]:+.2f} ({r["npct"]:+.0f}%/yr)  gross Sh {r["gsh"]:+.2f}  turn {r["turn"]:.0f}/d')
    print()

# best cap per-year check (lag1)
best = gated_eval(Wc, 1, 0.1)
yr = best['net'].dropna().groupby(lambda t: t.year).agg(lambda r: r.mean()/r.std()*ANN if r.std()>0 else 0)
print('lag1 cap0.10 per-year NET Sharpe:', '  '.join(f'{y}:{v:+.1f}' for y, v in yr.items()))

"""#029g - THE decisive FTMO test: crossing pool + point-in-time spread gate on REAL FTMO spreads.
Builds a per-bar FTMO half-spread panel by mapping each 2019-2026 M15 bar to its (pair, UTC-hour)
median half-spread from the MT5-measured profile (x029f). FX spread structure is hour-of-day driven
(tight in London/NY, wide at rollover/Asia), so this captures WHEN each pair is cheap to trade.
Then runs the gate: trade a pair only when its FTMO spread that hour <= cap. Sweep cap, lag1/lag2."""
import numpy as np
import pandas as pd
import x029c_wide_crossing as c

cols, idx, rets = c.cols, c.idx, c.rets
BARS, ANN = c.BARS, c.ANN
Wc = sum(c.books.values()) / len(c.books)

prof = pd.read_parquet('data/ftmo_spread_profile.parquet')
piv = prof.pivot(index='pair', columns='hour', values='med_half_bp')      # pair x hour (bp)
# fill missing (pair,hour) cells with the pair's own median across observed hours
piv = piv.apply(lambda r: r.fillna(r.median()), axis=1)
overall_med = prof.groupby('pair')['med_half_bp'].median()

hour = idx.hour
half_ftmo = pd.DataFrame(index=idx, columns=cols, dtype=float)
for p in cols:
    row = piv.loc[p] if p in piv.index else None
    if row is None:
        half_ftmo[p] = overall_med.get(p, 1.0)
    else:
        half_ftmo[p] = pd.Series(hour, index=idx).map(row).fillna(overall_med.get(p, 1.0)).values
half_ftmo = half_ftmo / 1e4                                                 # bp -> fraction

print('FTMO median half-spread bp by pair (from MT5 profile):')
print({p: round(float(overall_med.get(p, np.nan)), 3) for p in cols})

def gated_eval(target, lag, cap_bp):
    cap = cap_bp / 1e4
    tradable = half_ftmo <= cap
    gpos = target.where(tradable).ffill().fillna(0.0)
    gross = (gpos.shift(lag) * rets).sum(axis=1)
    dpos = (gpos - gpos.shift(1)).abs()
    net = gross - (dpos * half_ftmo).sum(axis=1)
    turn = dpos.sum(axis=1).mean() * BARS
    def sh(r): r = r.dropna(); return r.mean()/r.std()*ANN if r.std() > 0 else 0
    def pct(r): return r.dropna().mean()*BARS*252*100
    return dict(nsh=sh(net), npct=pct(net), gsh=sh(gross), turn=turn, ftr=tradable.mean().mean(), net=net)

print('\n=== crossing pool + spread gate on REAL FTMO spreads (28 pairs) ===')
for lag in (1, 2):
    print(f'lag{lag} ({"idealized" if lag==1 else "realistic"}):')
    # no gate = charge whatever the FTMO spread is, always trade
    ng = (Wc.shift(lag)*rets).sum(axis=1) - ((Wc-Wc.shift(1)).abs()*half_ftmo).sum(axis=1)
    ng = ng.dropna(); print(f'  NO GATE          : NET Sh {ng.mean()/ng.std()*ANN:+.2f} ({ng.mean()*BARS*252*100:+.0f}%/yr)')
    for cap in (0.30, 0.20, 0.15, 0.10, 0.05):
        r = gated_eval(Wc, lag, cap)
        print(f'  cap {cap:.2f}bp (trade {r["ftr"]*100:4.1f}%): NET Sh {r["nsh"]:+.2f} ({r["npct"]:+.1f}%/yr)  '
              f'gross Sh {r["gsh"]:+.2f}  turn {r["turn"]:.0f}/d')

# per-year for the best lag1 cap
best = gated_eval(Wc, 1, 0.15)
yr = best['net'].dropna().groupby(lambda t: t.year).agg(lambda r: r.mean()/r.std()*ANN if r.std()>0 else 0)
print('\nlag1 cap0.15 per-year NET Sharpe:', '  '.join(f'{y}:{v:+.1f}' for y, v in yr.items()))
ret = best['net'].dropna().groupby(lambda t: t.year).sum() * 100
print('lag1 cap0.15 per-year NET return%:', '  '.join(f'{y}:{v:+.1f}' for y, v in ret.items()))

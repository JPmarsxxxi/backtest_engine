"""#029d - THE DECISIVE TEST: crossing-pool NET on the WIDE 28-pair universe with REAL per-bar spreads.
Does more pairs (56 triangles + breadth) finally make the pooled reversion book clear net-positive
after cost, where the 12-pair version couldn't? Reuses x029c's 28-pair member books (#027,S2,S4,S7,S8,M4).
Real half-spread from the wide ASK panel (Dukascopy, wider than FTMO = conservative). lag1 idealized /
lag2 realistic. Reports all-28 and a tight-spread subset + no-trade-band sweep."""
import numpy as np
import pandas as pd
import x029c_wide_crossing as c   # reuses panel, cols, idx, logp, rets, books (incl #027)

cols, idx, rets = c.cols, c.idx, c.rets
BARS, ANN = c.BARS, c.ANN
mid = c.panel
ask = pd.read_parquet('data/fx_wide_ask_m15.parquet').sort_index()[cols]
half = ((ask - mid) / mid).clip(lower=0, upper=10e-4)      # real per-bar half-spread (frac)

# real median half-spread per pair (bp) -> pick tight tradable subset
med_bp = (half.median() * 1e4).sort_values()
TIGHT = list(med_bp[med_bp < 0.6].index)                    # < 0.6bp half

def evaluate(w, lag, band=0.0, subset=None):
    if subset is not None:
        w = w.copy(); w[[x for x in cols if x not in subset]] = 0.0
        w = w.sub(w[subset].mean(axis=1), axis=0); w[[x for x in cols if x not in subset]] = 0.0
        g = w.abs().sum(axis=1); w = w.div(g.where(g > 0), axis=0).fillna(0.0)
    if band > 0:
        tgt = w.values; T, n = tgt.shape; held = np.zeros(n); out = np.zeros((T, n))
        for t in range(T):
            mv = np.abs(tgt[t] - held) > band
            held = np.where(mv, tgt[t], held); out[t] = held
        w = pd.DataFrame(out, index=w.index, columns=w.columns)
    gross = (w.shift(lag) * rets).sum(axis=1)
    cost = ((w - w.shift(1)).abs() * half).sum(axis=1)
    net = gross - cost
    turn = (w - w.shift(1)).abs().sum(axis=1).mean() * BARS
    def sh(r): r = r.dropna(); return r.mean() / r.std() * ANN if r.std() > 0 else 0
    def pct(r): return r.dropna().mean() * BARS * 252 * 100
    return dict(gsh=sh(gross), nsh=sh(net), gpct=pct(gross), npct=pct(net), turn=turn, net=net)

Wc = sum(c.books.values()) / len(c.books)

print('\n############ #029d WIDE (28-pair) CROSSING NET TEST — real Dukascopy spreads ############')
print(f'tight subset (<0.6bp half): {TIGHT}')
print('half-spread bp per pair:', {k: round(v, 2) for k, v in med_bp.items()})

print('\n=== ALL 28 pairs, real spreads ===')
for lag in (1, 2):
    r = evaluate(Wc, lag)
    print(f'  lag{lag} {"ideal " if lag==1 else "realist"}: gross Sh {r["gsh"]:+.2f} ({r["gpct"]:+.0f}%/yr)  '
          f'NET Sh {r["nsh"]:+.2f} ({r["npct"]:+.0f}%/yr)  turn {r["turn"]:.0f}/d')

print('\n=== TIGHT subset + no-trade-band sweep ===')
for lag in (1, 2):
    print(f'  lag{lag} ({"idealized" if lag==1 else "realistic"}):')
    for band in (0.0, 0.05, 0.10, 0.20, 0.30):
        r = evaluate(Wc, lag, band=band, subset=TIGHT)
        print(f'    band {band:.2f}: NET Sh {r["nsh"]:+.2f} ({r["npct"]:+.0f}%/yr)  '
              f'gross Sh {r["gsh"]:+.2f}  turn {r["turn"]:.0f}/d')

# FAIREST best-case: 28-pair signals (rich factor/triangle structure) traded on 6 FTMO-tight majors
# with FTMO-measured spreads (comparable to x028c's +0.72). Overrides half with FTMO constants.
FMAJ = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF']
FTMO_HALF = pd.Series({'EURUSD': 0.10, 'USDJPY': 0.16, 'GBPUSD': 0.19, 'AUDUSD': 0.25,
                       'USDCAD': 0.30, 'USDCHF': 0.30})
half_ftmo = pd.Series({x: FTMO_HALF.get(x, 9.0) for x in cols}) / 1e4

def eval_ftmo(w, lag, band=0.0):
    w = w.copy(); w[[x for x in cols if x not in FMAJ]] = 0.0
    w = w.sub(w[FMAJ].mean(axis=1), axis=0); w[[x for x in cols if x not in FMAJ]] = 0.0
    g = w.abs().sum(axis=1); w = w.div(g.where(g > 0), axis=0).fillna(0.0)
    if band > 0:
        tgt = w.values; T, n = tgt.shape; held = np.zeros(n); out = np.zeros((T, n))
        for t in range(T):
            mv = np.abs(tgt[t] - held) > band; held = np.where(mv, tgt[t], held); out[t] = held
        w = pd.DataFrame(out, index=w.index, columns=w.columns)
    gross = (w.shift(lag) * rets).sum(axis=1)
    cost = ((w - w.shift(1)).abs() * half_ftmo).sum(axis=1)
    net = (gross - cost)
    def sh(r): r = r.dropna(); return r.mean()/r.std()*ANN if r.std() > 0 else 0
    return sh(gross), sh(net), net.dropna().mean()*BARS*252*100

print('\n=== FAIREST best-case: 28-pair signals traded on 6 FTMO majors, FTMO-tight spreads ===')
for lag in (1, 2):
    print(f'  lag{lag} ({"idealized" if lag==1 else "realistic"}):')
    for band in (0.0, 0.10, 0.20, 0.30):
        gs, ns, npct = eval_ftmo(Wc, lag, band)
        print(f'    band {band:.2f}: NET Sh {ns:+.2f} ({npct:+.0f}%/yr)  gross Sh {gs:+.2f}')

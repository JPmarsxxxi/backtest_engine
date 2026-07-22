"""#028c - crossing pool BEST-CASE honest shot: trade only tight-spread MAJORS with FTMO-measured
spreads (not wide Dukascopy). If the pool can't clear here, it can't clear anywhere retail.

Restrict each member's tradable legs to {EURUSD,USDJPY,GBPUSD,AUDUSD,USDCAD,USDCHF} (signals still
COMPUTED on all 12 - triangles/PCA need the crosses - but positions zeroed outside majors, renormalized).
Spread = FTMO-measured/estimated CONSTANT half (from #016 MT5 measurement), tighter than Dukascopy.
Also sweeps a no-trade band on the pooled book to confirm bands can't rescue a 1-bar-horizon edge.
"""
import numpy as np
import pandas as pd
import x027b_intraday_statarb as e
import x028b_crossing_pool as p   # reuse loaded panels, signals, books, half, rets

cols, idx, rets, logp, z = p.cols, p.idx, p.rets, p.logp, p.z
BARS_PER_DAY, ANN = p.BARS_PER_DAY, p.ANN

MAJORS = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF']
# FTMO-measured half-spread bp (majors from #016 MT5; others n/a - not traded)
FTMO_HALF = pd.Series({'EURUSD': 0.10, 'USDJPY': 0.16, 'GBPUSD': 0.19,
                       'AUDUSD': 0.25, 'USDCAD': 0.30, 'USDCHF': 0.30})
half_ftmo = pd.Series({c: FTMO_HALF.get(c, 5.0) for c in cols}) / 1e4   # non-majors huge (blocked anyway)

def to_majors(w):
    w2 = w.copy()
    w2[[c for c in cols if c not in MAJORS]] = 0.0
    w2 = w2.sub(w2[MAJORS].mean(axis=1), axis=0)        # re-neutralize within majors
    w2[[c for c in cols if c not in MAJORS]] = 0.0
    g = w2.abs().sum(axis=1)
    return w2.div(g.where(g > 0), axis=0).fillna(0.0)

books_maj = {k: to_majors(w) for k, w in p.books.items()}
Wc = sum(books_maj.values()) / len(books_maj)

def evaluate(w, lag, band=0.0):
    """optionally apply a no-trade band: only step toward target when |target-held|>band (per pair)."""
    if band > 0:
        tgt = w.values; T, n = tgt.shape
        held = np.zeros(n); out = np.zeros((T, n))
        for t in range(T):
            move = np.abs(tgt[t] - held) > band
            held = np.where(move, tgt[t], held)
            out[t] = held
        w = pd.DataFrame(out, index=w.index, columns=w.columns)
    gross = (w.shift(lag) * rets).sum(axis=1)
    dpos = (w - w.shift(1)).abs()
    cost = (dpos * half_ftmo).sum(axis=1)
    net = gross - cost
    turn = dpos.sum(axis=1).mean() * BARS_PER_DAY
    def sh(r): r = r.dropna(); return r.mean() / r.std() * ANN if r.std() > 0 else 0
    def pct(r): return r.dropna().mean() * BARS_PER_DAY * 252 * 100
    return dict(gsh=sh(gross), nsh=sh(net), gpct=pct(gross), npct=pct(net), turn=turn, net=net)

print('=== #028c MAJORS-ONLY pool, FTMO-measured spreads (best honest retail shot) ===')
print(f'majors traded: {MAJORS}  |  FTMO half-spreads bp: {FTMO_HALF.to_dict()}\n')
for lag in (1, 2):
    tag = 'idealized' if lag == 1 else 'realistic (1 bar late)'
    r = evaluate(Wc, lag)
    print(f'lag{lag} {tag:22}: gross Sh {r["gsh"]:+.2f} ({r["gpct"]:+.0f}%/yr)  '
          f'NET Sh {r["nsh"]:+.2f} ({r["npct"]:+.0f}%/yr)  turn {r["turn"]:.1f}/d')

print('\n--- no-trade band sweep (pool, lag1) — tests whether banding rescues a 1-bar-horizon edge ---')
for band in (0.0, 0.02, 0.05, 0.10, 0.20):
    r = evaluate(Wc, 1, band=band)
    print(f'  band {band:.2f}: NET Sh {r["nsh"]:+.2f} ({r["npct"]:+.0f}%/yr)  gross Sh {r["gsh"]:+.2f}  turn {r["turn"]:.1f}/d')
print('\n(if net stays negative & ~flat across bands -> edge lives in 1 bar, cost scales with edge, unrescuable)')

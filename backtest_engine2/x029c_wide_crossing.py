"""#029c - GROSS crossing preview on the WIDE 28-pair universe (cost test waits for the ASK pull).
Shows whether more pairs (56 triangles + breadth) lifts the pooled GROSS Sharpe and how much the
crossing effect nets out turnover. Members: #027 PCA-resid, S2 XS-rev, S4 triangular(56), S7 bloc,
S8 range, M4 jump. Equal-weight combine. lag1 (idealized) + lag2 (realistic)."""
import numpy as np
import pandas as pd
import x027b_intraday_statarb as e
import x029b_wide_pool as w

BARS = 96
ANN = np.sqrt(BARS * 252)
panel, cols = w.load_wide()
N = len(cols)
idx = panel.index
logp = np.log(panel)
rets = logp.diff()
gap = (idx.to_series().diff() > pd.Timedelta(f'{e.GAP_MIN}min')).values
rets[gap] = np.nan
z = (rets / rets.rolling(e.VOLW, min_periods=e.VOLW // 2).std()).clip(-8, 8)
tris = w.all_triangles(cols)

def zdev(s, W):
    d = s - s.rolling(W, min_periods=W // 2).mean()
    return (d / d.rolling(W, min_periods=W // 2).std().replace(0, np.nan)).clip(-8, 8)

def S2(): return -z
def S4():
    sig = pd.DataFrame(0.0, index=idx, columns=cols)
    for legs in tris:
        g = sum(c * logp[s] for s, c in legs); d = zdev(g, 32)
        for s, c in legs: sig[s] = sig[s] - c * d
    return sig
def S8(K=16):
    hi = logp.rolling(K, min_periods=K // 2).max(); lo = logp.rolling(K, min_periods=K // 2).min()
    return -((logp - lo) / (hi - lo).replace(0, np.nan) - 0.5)
def M4(K=16, thr=3.0):
    bpv = np.sqrt((np.pi / 2) * (rets.abs() * rets.abs().shift(1)).rolling(K, min_periods=K // 2).mean())
    L = (rets / bpv.replace(0, np.nan)).clip(-15, 15)
    return (-L).where(L.abs() > thr, 0.0)
def S7(W=48):
    # auto correlated-bloc: strongest same-base/quote reversion pairs
    sig = pd.DataFrame(0.0, index=idx, columns=cols)
    for a, b in [('AUDUSD', 'NZDUSD'), ('EURUSD', 'GBPUSD'), ('USDCAD', 'USDCHF'), ('AUDNZD', 'AUDCAD')]:
        if a in cols and b in cols:
            d = zdev(logp[a] - logp[b], W); sig[a] -= d; sig[b] += d
    return sig

print(f'computing #027 s-score on {N} pairs (slow)...', flush=True)
s027, _, _ = e.compute_sscore(panel, cols)

def unit(sig):
    wn = sig.sub(sig.mean(axis=1), axis=0); g = wn.abs().sum(axis=1)
    return wn.div(g.where(g > 0), axis=0).fillna(0.0)

books = {'#027': unit(s027), 'S2': unit(S2()), 'S4': unit(S4()),
         'S7': unit(S7()), 'S8': unit(S8()), 'M4': unit(M4())}

def gross(wbook, lag):
    r = (wbook.shift(lag) * rets).sum(axis=1).dropna()
    return r.mean() / r.std() * ANN if r.std() > 0 else 0

for lag in (1, 2):
    print(f'\n=== lag{lag} ({"idealized" if lag==1 else "realistic"}) GROSS Sharpe, {N} pairs ===')
    sum_turn = 0
    for name, wb in books.items():
        wu = wb; t = (wu - wu.shift(1)).abs().sum(axis=1).mean() * BARS; sum_turn += t
        print(f'  {name:6} gross Sh {gross(wb, lag):+.2f}   turn {t:.0f}/d')
    Wc = sum(books.values()) / len(books)
    tc = (Wc - Wc.shift(1)).abs().sum(axis=1).mean() * BARS
    print(f'  {"POOL":6} gross Sh {gross(Wc, lag):+.2f}   turn {tc:.0f}/d  '
          f'(crossing: {tc:.0f} vs sum {sum_turn:.0f} = {(1-tc/sum_turn)*100:.0f}% netted)')

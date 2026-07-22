"""#029b - re-screen the pool on the WIDE 28-pair universe (data/fx_wide_m15.parquet). The point:
more pairs multiplies the axes that actually diversify - triangular no-arb (auto-generate ALL 3-currency
triangles, up to C(8,3)=56 vs the hand-coded 5) and cross-sectional breadth (IR = IC*sqrt(breadth)).

Runs the same cheap IC/uniqueness/decay screen as x028d, plus reports breadth. Mid-only (ASK pull for
the crossing COST test comes after this screens well). Robust to a partial pull (uses whatever pairs
have >90% coverage).
"""
import numpy as np
import pandas as pd
from itertools import combinations
import x027b_intraday_statarb as e

BARS_PER_DAY = 96
WIDE = 'data/fx_wide_m15.parquet'


def load_wide():
    panel = pd.read_parquet(WIDE).sort_index()
    keep = panel.columns[panel.notna().mean() > 0.90]
    return panel[keep], list(keep)


def currencies(sym):
    return sym[:3], sym[3:]


def all_triangles(cols):
    """every 3-currency set whose 3 pairs are all present; return list of (sym, coeff) legs where
    g = sum coeff*logp[sym] is the cyclic no-arb residual (=0 ideally). coeff from orientation."""
    ccy = sorted({c for s in cols for c in currencies(s)})
    pair_of = {}
    for s in cols:
        b, q = currencies(s); pair_of[(b, q)] = s
    tris = []
    for X, Y, Z in combinations(ccy, 3):
        legs = []
        ok = True
        for A, B in [(X, Y), (Y, Z), (Z, X)]:      # cyclic: log(A/B)
            if (A, B) in pair_of:
                legs.append((pair_of[(A, B)], +1))   # symbol is A/B -> +logp
            elif (B, A) in pair_of:
                legs.append((pair_of[(B, A)], -1))   # symbol is B/A -> -logp = log(A/B)
            else:
                ok = False; break
        if ok:
            tris.append(legs)
    return tris


def build(panel, cols):
    idx = panel.index
    logp = np.log(panel)
    rets = logp.diff()
    gap = (idx.to_series().diff() > pd.Timedelta(f'{e.GAP_MIN}min')).values
    rets[gap] = np.nan
    z = (rets / rets.rolling(e.VOLW, min_periods=e.VOLW // 2).std()).clip(-8, 8)
    ret_neu = rets.sub(rets.mean(axis=1), axis=0)
    tris = all_triangles(cols)

    def zdev(s, W):
        d = s - s.rolling(W, min_periods=W // 2).mean()
        return (d / d.rolling(W, min_periods=W // 2).std().replace(0, np.nan)).clip(-8, 8)

    def S2():
        return -z

    def S4():
        sig = pd.DataFrame(0.0, index=idx, columns=cols)
        for legs in tris:
            g = sum(c * logp[s] for s, c in legs)
            d = zdev(g, 32)
            for s, c in legs:
                sig[s] = sig[s] - c * d
        return sig

    def M4_jump(K=16, thr=3.0):
        bpv = np.sqrt((np.pi / 2) * (rets.abs() * rets.abs().shift(1)).rolling(K, min_periods=K // 2).mean())
        L = (rets / bpv.replace(0, np.nan)).clip(-15, 15)
        return (-L).where(L.abs() > thr, 0.0)

    return dict(idx=idx, logp=logp, rets=rets, z=z, ret_neu=ret_neu, tris=tris,
                sigs={'S2': S2, 'S4': S4, 'M4_jump': M4_jump})


def neu(sig): return sig.sub(sig.mean(axis=1), axis=0)
def unit(sig):
    w = neu(sig); g = w.abs().sum(axis=1)
    return w.div(g.where(g > 0), axis=0).fillna(0.0)


def main():
    panel, cols = load_wide()
    N = len(cols)
    B = build(panel, cols)
    ret_neu, idx = B['ret_neu'], B['idx']
    print(f'#029b WIDE screen | {N} pairs | {len(panel):,} bars {idx.min().date()}->{idx.max().date()}')
    print(f'triangles auto-generated: {len(B["tris"])} (was 5 hand-coded on 12 pairs)')
    print(f'pairs: {cols}\n')

    def ic(wn, lag):
        a = wn.shift(lag).values.ravel(); b = ret_neu.values.ravel()
        m = np.isfinite(a) & np.isfinite(b)
        return (np.corrcoef(a[m], b[m])[0, 1], m.sum()) if m.sum() > 100 else (np.nan, 0)

    print(f'{"signal":10} {"IC1":>8} {"IC2":>8} {"sigma":>6} {"turn/d":>7}')
    for name, fn in B['sigs'].items():
        wn = unit(fn())
        i1, n = ic(wn, 1); i2, _ = ic(wn, 2)
        g = wn.abs().sum(axis=1); wu = wn.div(g.where(g > 0), axis=0).fillna(0.0)
        turn = (wu - wu.shift(1)).abs().sum(axis=1).mean() * BARS_PER_DAY
        print(f'{name:10} {i1:+8.4f} {i2:+8.4f} {abs(i1)*np.sqrt(n) if n else 0:6.1f} {turn:7.1f}')
    print(f'\nbreadth note: cross-sectional signals now span {N} pairs (was 12); '
          f'IR = IC*sqrt(breadth) so wider N lifts the achievable Sharpe on the reversion + triangular axes.')


if __name__ == '__main__':
    main()

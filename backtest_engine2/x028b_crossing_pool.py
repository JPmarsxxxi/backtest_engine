"""#028b - CROSSING-POOL engine test. The #027 thesis: a batch of weak, low-correlation, standalone-
uneconomic FX-intraday reversion alphas becomes net-positive when POOLED on one universe, because
(a) gross diversifies (low mutual corr -> Sharpe up ~sqrt(N_eff)) and (b) the CROSSING EFFECT nets
opposing desired positions BEFORE paying spread -> combined turnover < sum of member turnovers.

Honest accounting: REAL per-bar half-spread from the ask panel (Dukascopy, WIDER than FTMO demo =
conservative). Cost = |Delta position| * real half-spread (each open/close crosses half). Realistic
execution shown at lag1 (trade at forming close, idealized upper bound) AND lag2 (one bar late).
Swap: not charged (intraday, dollar-neutral ~balanced book); rollover-flat variant is a TODO if alive.

Members (5), all unit-gross dollar-neutral on the same 12-pair M15 universe:
  #027 PCA-residual s-score (8h OU)   S2 1-bar XS reversal   S4 triangular no-arb
  S7 correlated-bloc spread reversion  S8 range-position overreaction
"""
import numpy as np
import pandas as pd
import x027b_intraday_statarb as e

BARS_PER_DAY = 96
ANN = np.sqrt(BARS_PER_DAY * 252)
GAP_MIN = e.GAP_MIN

mid = pd.read_parquet('data/fx_intraday_m15.parquet').sort_index()
ask = pd.read_parquet('data/fx_intraday_ask_m15.parquet').sort_index()
keep = mid.columns[mid.notna().mean() > 0.90]
mid, ask = mid[keep], ask[keep]
cols = list(keep); N = len(cols); idx = mid.index
logp = np.log(mid)
rets = logp.diff()
gap = (idx.to_series().diff() > pd.Timedelta(f'{GAP_MIN}min')).values
rets[gap] = np.nan
vol = rets.rolling(e.VOLW, min_periods=e.VOLW // 2).std()
z = (rets / vol).clip(-8, 8)

# REAL time-varying half-spread (bp -> fraction); clip data-error spikes at 10bp
half = ((ask - mid) / mid).clip(lower=0, upper=10e-4)

def neu(sig):
    return sig.sub(sig.mean(axis=1), axis=0)

def unit(sig):
    w = neu(sig); g = w.abs().sum(axis=1)
    return w.div(g.where(g > 0), axis=0).fillna(0.0)

# ---- member signals ----
def sig_S2():
    return -z

TRI = [('EURJPY', [('EURUSD', +1), ('USDJPY', +1)]), ('GBPJPY', [('GBPUSD', +1), ('USDJPY', +1)]),
       ('AUDJPY', [('AUDUSD', +1), ('USDJPY', +1)]), ('EURGBP', [('EURUSD', +1), ('GBPUSD', -1)]),
       ('EURAUD', [('EURUSD', +1), ('AUDUSD', -1)])]
def sig_S4(W=32):
    sig = pd.DataFrame(0.0, index=idx, columns=cols)
    for cross, legs in TRI:
        g = logp[cross].copy()
        for leg, s in legs: g = g - s * logp[leg]
        d = g - g.rolling(W, min_periods=W // 2).mean()
        d = (d / d.rolling(W, min_periods=W // 2).std()).clip(-8, 8)
        sig[cross] = sig[cross] - d
        for leg, s in legs: sig[leg] = sig[leg] + s * d
    return sig

def sig_S7(W=48):
    sig = pd.DataFrame(0.0, index=idx, columns=cols)
    for a, b in [('AUDUSD', 'NZDUSD'), ('EURUSD', 'GBPUSD')]:
        r = logp[a] - logp[b]
        d = r - r.rolling(W, min_periods=W // 2).mean()
        d = (d / d.rolling(W, min_periods=W // 2).std()).clip(-8, 8)
        sig[a] -= d; sig[b] += d
    return sig

def sig_S8(K=16):
    hi = logp.rolling(K, min_periods=K // 2).max(); lo = logp.rolling(K, min_periods=K // 2).min()
    return -((logp - lo) / (hi - lo).replace(0, np.nan) - 0.5)

print('computing #027 s-score (slow)...', flush=True)
s027, _, _ = e.compute_sscore(mid, cols)

books = {'#027': unit(s027), 'S2': unit(sig_S2()), 'S4': unit(sig_S4()),
         'S7': unit(sig_S7()), 'S8': unit(sig_S8())}

def evaluate(w, lag):
    """gross/net Sharpe, ann%, turnover for unit-gross book w at execution lag."""
    gross = (w.shift(lag) * rets).sum(axis=1)
    dpos = (w - w.shift(1)).abs()
    cost = (dpos * half).sum(axis=1)
    net = gross - cost
    turn = dpos.sum(axis=1).mean() * BARS_PER_DAY
    def sh(r): r = r.dropna(); return r.mean() / r.std() * ANN if r.std() > 0 else 0
    def pct(r): return r.dropna().mean() * BARS_PER_DAY * 252 * 100
    return dict(gsh=sh(gross), nsh=sh(net), gpct=pct(gross), npct=pct(net),
                turn=turn, net=net)

for lag in (1, 2):
    tag = 'lag1 (idealized: trade at forming close)' if lag == 1 else 'lag2 (realistic: one bar late)'
    print(f'\n================ EXECUTION {tag} ================')
    print(f'{"member":8} {"grossSh":>8} {"netSh":>7} {"gross%":>8} {"net%":>8} {"turn/d":>8}')
    sum_turn = 0.0
    for name, w in books.items():
        r = evaluate(w, lag)
        sum_turn += r['turn']
        print(f'{name:8} {r["gsh"]:+8.2f} {r["nsh"]:+7.2f} {r["gpct"]:+8.1f} {r["npct"]:+8.1f} {r["turn"]:8.1f}')
    # combined pool = equal-weight mean of member books (crossing happens here)
    Wc = sum(books.values()) / len(books)
    rc = evaluate(Wc, lag)
    cross_ratio = rc['turn'] / sum_turn
    print(f'{"-"*50}')
    print(f'{"POOL":8} {rc["gsh"]:+8.2f} {rc["nsh"]:+7.2f} {rc["gpct"]:+8.1f} {rc["npct"]:+8.1f} {rc["turn"]:8.1f}')
    print(f'  crossing: pool turnover {rc["turn"]:.1f}/d vs sum-of-members {sum_turn:.1f}/d  '
          f'-> ratio {cross_ratio:.2f} ({(1-cross_ratio)*100:.0f}% netted out)')
    print(f'  pool gross Sharpe {rc["gsh"]:+.2f} (diversification of {N} pairs x 5 low-corr books)')
    if lag == 1:
        yr = rc['net'].dropna().groupby(lambda t: t.year).agg(
            lambda r: r.mean() / r.std() * ANN if r.std() > 0 else 0)
        print('  POOL per-year NET Sharpe:', '  '.join(f'{y}:{v:+.1f}' for y, v in yr.items()))

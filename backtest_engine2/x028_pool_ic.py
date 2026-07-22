"""#028 - FX-intraday POOL candidate screen (Step 3/6 cheap gate).

Goal: feed a CROSSING POOL for #027 (parked: real IC but standalone-uneconomic). We are NOT
judging standalone economics here - per finding-alphas Ch.15 we judge the BATCH. The two things
that matter for a pool ingredient are:
  (1) is the IC REAL?  -> pooled IC by execution lag, with sigma (n huge -> tiny IC still real)
  (2) is it UNIQUE?    -> |corr| of its neutral book vs #027's s-score book (<0.3 = good diversifier)

Same 12-pair M15 universe as #027 (so opposing pooled trades net out -> crossing effect).
Timing: signal formed at close of bar t predicts return of bar t+1. IC = corr(sig.shift(lag), ret).
  lag=1 = trade at forming close (idealized upper bound); lag>=2 = realistic execution latency.
Target = cross-sectionally DEMEANED forward return (dollar-neutral book target).

Candidates (mechanistically distinct on purpose):
  S1 rev1     own 1-bar reversal (control; expect high corr to #027)
  S2 xsrev    cross-sectional 1-bar reversal (single-factor demean)
  S3 leadlag  dollar-factor CONTINUATION (laggards follow the USD move) - Huth/cross-currency lead-lag
  S4 tri      triangular no-arbitrage residual reversion (EURJPY=EURUSD+USDJPY etc.) - hard constraint
  S5 semivar  downside-vs-upside realized semivariance asymmetry -> reversal (vol-shape axis)
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import x027b_intraday_statarb as e

BARS_PER_DAY = 96
GAP_MIN = e.GAP_MIN

panel, cols = e.load()
N = len(cols)
idx = panel.index
logp = np.log(panel)
rets = logp.diff()
gap = (idx.to_series().diff() > pd.Timedelta(f'{GAP_MIN}min')).values
rets[gap] = np.nan
vol = rets.rolling(e.VOLW, min_periods=e.VOLW // 2).std()
z = (rets / vol).clip(-8, 8)                       # standardized bar returns

# dollar orientation: +1 USD is BASE (USDxxx), -1 USD is QUOTE (xxxUSD), 0 = cross
usd_sign = {}
for c in cols:
    if c.startswith('USD'):   usd_sign[c] = +1.0
    elif c.endswith('USD'):   usd_sign[c] = -1.0
    else:                     usd_sign[c] = 0.0
usd_sign = pd.Series(usd_sign)

# neutral forward-return target (what a dollar-neutral book actually earns)
ret_neutral = rets.sub(rets.mean(axis=1), axis=0)


def neutralize(sig):
    """cross-sectionally demean -> dollar-neutral book weights (unit-gross not needed for corr/IC)."""
    return sig.sub(sig.mean(axis=1), axis=0)


# ---------------- candidate signals (each: DataFrame aligned to panel, NaN where undefined) --------
def S1_rev1():
    return -z

def S2_xsrev():
    return -(z.sub(z.mean(axis=1), axis=0))

def S3_leadlag():
    """USD pairs continue the contemporaneous dollar-factor move into the next bar."""
    usd_cols = [c for c in cols if usd_sign[c] != 0]
    usd_ret = pd.Series(0.0, index=idx)
    zu = z[usd_cols].mul(usd_sign[usd_cols], axis=1)     # each oriented so +1 = USD strengthens
    usd_ret = zu.mean(axis=1)
    sig = pd.DataFrame(0.0, index=idx, columns=cols)
    for c in usd_cols:
        sig[c] = usd_sign[c] * usd_ret                    # pair follows USD move next bar
    return sig

# triangles present in the 12-pair set: log(cross) = a*log(leg1) + b*log(leg2)
TRIANGLES = [
    ('EURJPY', [('EURUSD', +1), ('USDJPY', +1)]),
    ('GBPJPY', [('GBPUSD', +1), ('USDJPY', +1)]),
    ('AUDJPY', [('AUDUSD', +1), ('USDJPY', +1)]),
    ('EURGBP', [('EURUSD', +1), ('GBPUSD', -1)]),
    ('EURAUD', [('EURUSD', +1), ('AUDUSD', -1)]),
]

def S4_tri():
    """g = log(cross) - sum(legs); g~0 by no-arb. Deviation of g from its rolling mean predicts
    the three legs move to close it. Signal per leg = -d(g)/d(that leg's log price)."""
    sig = pd.DataFrame(0.0, index=idx, columns=cols)
    W = 32
    for cross, legs in TRIANGLES:
        if cross not in cols or any(l not in cols for l, _ in legs):
            continue
        g = logp[cross].copy()
        for leg, s in legs:
            g = g - s * logp[leg]
        d = g - g.rolling(W, min_periods=W // 2).mean()   # deviation (g has ~0 drift already)
        d = (d / d.rolling(W, min_periods=W // 2).std()).clip(-8, 8)
        # cross is rich if d>0 -> expect cross DOWN (-d), each leg moves +s*d to close gap
        sig[cross] = sig[cross] - d
        for leg, s in legs:
            sig[leg] = sig[leg] + s * d
    return sig

def S5_semivar(K=16):
    """downside minus upside realized semivariance over K bars -> mean-reversion up after down-vol."""
    up = rets.clip(lower=0) ** 2
    dn = rets.clip(upper=0) ** 2
    svu = up.rolling(K, min_periods=K // 2).sum()
    svd = dn.rolling(K, min_periods=K // 2).sum()
    asym = (svd - svu) / (svd + svu).replace(0, np.nan)   # +1 = all downside vol
    return asym                                            # high downside -> expect up -> +signal

def S6_mom(K=8):
    """2h intraday momentum (continuation) - sign-diverse member (helps crossing)."""
    return z.rolling(K, min_periods=K // 2).mean()

CORREL_PAIRS = [('AUDUSD', 'NZDUSD'), ('EURUSD', 'GBPUSD')]

def S7_pairspread(W=48):
    """correlated-pair (commodity AUD/NZD, majors EUR/GBP) log-spread reversion - NOT a triangle."""
    sig = pd.DataFrame(0.0, index=idx, columns=cols)
    for a, b in CORREL_PAIRS:
        if a not in cols or b not in cols:
            continue
        r = logp[a] - logp[b]
        d = r - r.rolling(W, min_periods=W // 2).mean()
        d = (d / d.rolling(W, min_periods=W // 2).std()).clip(-8, 8)
        sig[a] = sig[a] - d       # a rich vs b -> fade a, buy b
        sig[b] = sig[b] + d
    return sig

def S8_range(K=16):
    """position in recent range -> fade extremes (short-horizon overreaction)."""
    hi = logp.rolling(K, min_periods=K // 2).max()
    lo = logp.rolling(K, min_periods=K // 2).min()
    pos = (logp - lo) / (hi - lo).replace(0, np.nan) - 0.5
    return -pos                    # near top -> short

CANDS = {'S1_rev1': S1_rev1, 'S2_xsrev': S2_xsrev, 'S3_leadlag': S3_leadlag,
         'S4_tri': S4_tri, 'S5_semivar': S5_semivar,
         'S6_mom': S6_mom, 'S7_pairspread': S7_pairspread, 'S8_range': S8_range}

# ---------------- #027 reference book (uniqueness target) ------------------------------------------
print('computing #027 s-score reference (slow: daily PCA loop)...', flush=True)
s027, _, _ = e.compute_sscore(panel, cols)
w027 = neutralize(s027)

def pooled_corr(A, B):
    a = A.values.ravel(); b = B.values.ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 100: return np.nan
    return np.corrcoef(a[m], b[m])[0, 1]

print(f'\n#028 pool screen | {len(panel):,} M15 bars {idx.min().date()}->{idx.max().date()} | N={N} pairs')
print(f'target = cross-sectionally demeaned forward return | ann bets/yr ~ {BARS_PER_DAY*252:,}\n')
hdr = f'{"cand":11} {"IC(l1)":>8} {"IC(l2)":>8} {"IC(l3)":>8} {"rIC(l1)":>8} {"IRl2":>7} {"turn/d":>7} {"corr027":>8}'
print(hdr); print('-' * len(hdr))

results = {}
books = {}
for name, fn in CANDS.items():
    sig = fn()
    wn = neutralize(sig)                               # neutral book weights
    books[name] = wn
    # IC vs neutral forward return
    ics = []
    for lag in (1, 2, 3):
        a = wn.shift(lag).values.ravel()
        b = ret_neutral.values.ravel()
        m = np.isfinite(a) & np.isfinite(b)
        ics.append(np.corrcoef(a[m], b[m])[0, 1] if m.sum() > 100 else np.nan)
    # rank-IC lag1
    a = wn.shift(1).values.ravel(); b = ret_neutral.values.ravel()
    m = np.isfinite(a) & np.isfinite(b)
    ric1 = np.corrcoef(rankdata(a[m]), rankdata(b[m]))[0, 1]
    n_used = m.sum()
    ir_l2 = ics[1] * np.sqrt(BARS_PER_DAY * 252)
    # turnover of unit-gross neutral book
    g = wn.abs().sum(axis=1); wu = wn.div(g.where(g > 0), axis=0).fillna(0.0)
    turn = (wu - wu.shift(1)).abs().sum(axis=1).mean() * BARS_PER_DAY
    c027 = pooled_corr(wn, w027)
    results[name] = dict(ic1=ics[0], ic2=ics[1], ic3=ics[2], ric1=ric1, ir2=ir_l2,
                         turn=turn, c027=c027, n=n_used)
    print(f'{name:11} {ics[0]:+8.4f} {ics[1]:+8.4f} {ics[2]:+8.4f} {ric1:+8.4f} '
          f'{ir_l2:+7.1f} {turn:7.1f} {c027:+8.3f}')

# rough sigma on IC lag1 (n huge): se ~ 1/sqrt(n)
print('\nsigma check (IC lag1 vs 0, se~1/sqrt(n)):')
for name, r in results.items():
    se = 1 / np.sqrt(r['n'])
    print(f'  {name:11} IC1 {r["ic1"]:+.4f}  ~{abs(r["ic1"])/se:5.1f} sigma   corr#027 {r["c027"]:+.3f}  '
          f'({"UNIQUE" if abs(r["c027"])<0.3 else "redundant"} )')
print('\nread: real IC = many-sigma; POOL KEEPER = real IC AND |corr#027|<0.3. lag2 = realistic exec.')

# per-year IC lag1 (DECAY-FIRST gate: is the signal recent-persistent or front-loaded?)
print('\n=== per-year IC (lag1) — decay check; want recent years still positive ===')
yrs = sorted(set(idx.year))
print(f'{"cand":13}' + ''.join(f'{y:>8}' for y in yrs))
b_neu = ret_neutral.values
yr_of = idx.year.values
for name in CANDS:
    a_all = books[name].shift(1).values
    line = f'{name:13}'
    for y in yrs:
        rowm = (yr_of == y)
        a = a_all[rowm].ravel(); b = b_neu[rowm].ravel()
        m = np.isfinite(a) & np.isfinite(b)
        ic = np.corrcoef(a[m], b[m])[0, 1] if m.sum() > 100 else np.nan
        line += f'{ic:>+8.3f}'
    print(line)

# pairwise correlation matrix (pool diversity: members must be low-corr to EACH OTHER too)
print('\n=== pairwise |book| correlation (candidates + #027) — pool needs OFF-DIAGONAL < ~0.3 ===')
allbooks = dict(books); allbooks['#027'] = w027
names = list(allbooks)
print(f'{"":13}' + ''.join(f'{n[:9]:>10}' for n in names))
for n1 in names:
    row = []
    for n2 in names:
        row.append(pooled_corr(allbooks[n1], allbooks[n2]))
    print(f'{n1:13}' + ''.join(f'{v:>+10.2f}' for v in row))

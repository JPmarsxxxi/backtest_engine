"""#028d - BIG mathematical/quant candidate batch for the FX-intraday pool. Screens ~11 new signals
(each grounded in a real quant methodology) alongside the 5 existing keepers, on the same 12-pair M15
universe. Same cheap gate as x028: pooled IC by exec-lag + sigma, per-year IC (decay), turnover, and
MAX |corr| vs every existing pool member (uniqueness -> only low-corr adds diversity).

Data: M15 close (mid+ask) + quote_volume per pair. No OHLC.

New candidates & methodology:
  M1_ou       Ornstein-Uhlenbeck reversion: z of log-price vs rolling mean (half-life-scaled)
  M2_varratio variance-ratio-gated reversion (Lo-MacKinlay VR<1 = anti-persistent -> revert)
  M4_jump     Lee-Mykland/bipower jump reversal (ret/local-bipower-vol; jumps overreact -> revert)
  M6_rvolrev  vol-scaled reversal (-ret / realized vol; Ch.27 Sharpe-max form)
  M7_amihud   Amihud-illiquidity-weighted reversal (|ret|/volume; reversal premium bigger when illiquid)
  M8_volshock volume-shock reversal (high-volume bar move -> revert)
  M9_ofi      tick-rule signed order-flow imbalance -> reversal (VPIN/OFI proxy on retail data)
  M10_kalman  Kalman dynamic-hedge pair spread reversion (time-varying beta vs S7 static)
  M11_llag    lagged lead-lag CONTINUATION from EURUSD leader (proper lag, vs S3 contemporaneous)
  M13_multihzn multi-horizon blended reversal (reversion term structure 1/4/16 bars)
  M14_autocorr autocorrelation-conditioned reversion (trade reversal only when returns anti-persistent)
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import x027b_intraday_statarb as e

BARS_PER_DAY = 96
GAP_MIN = e.GAP_MIN

mid = pd.read_parquet('data/fx_intraday_m15.parquet').sort_index()
ask = pd.read_parquet('data/fx_intraday_ask_m15.parquet').sort_index()
keep = mid.columns[mid.notna().mean() > 0.90]
mid, ask = mid[keep], ask[keep]
cols = list(keep); N = len(cols); idx = mid.index

# volume panel (quote_volume per pair, aligned)
volp = pd.DataFrame(index=idx)
for c in cols:
    v = pd.read_parquet(f'data/{c.lower()}_m15.parquet')
    v = v.set_index('open_time')['quote_volume']
    v.index = pd.to_datetime(v.index, utc=True)
    volp[c] = v.reindex(idx)
volp = volp.clip(lower=1e-9)

logp = np.log(mid)
rets = logp.diff()
gap = (idx.to_series().diff() > pd.Timedelta(f'{GAP_MIN}min')).values
rets[gap] = np.nan
vol = rets.rolling(e.VOLW, min_periods=e.VOLW // 2).std()
z = (rets / vol).clip(-8, 8)
ret_neutral = rets.sub(rets.mean(axis=1), axis=0)


def zdev(series_df, W):
    d = series_df - series_df.rolling(W, min_periods=W // 2).mean()
    return (d / d.rolling(W, min_periods=W // 2).std().replace(0, np.nan)).clip(-8, 8)


def rollz(df, W):
    return ((df - df.rolling(W, min_periods=W // 2).mean())
            / df.rolling(W, min_periods=W // 2).std().replace(0, np.nan)).clip(-8, 8)


# ---------------- existing 5 keepers (for uniqueness comparison) ----------------
TRI = [('EURJPY', [('EURUSD', +1), ('USDJPY', +1)]), ('GBPJPY', [('GBPUSD', +1), ('USDJPY', +1)]),
       ('AUDJPY', [('AUDUSD', +1), ('USDJPY', +1)]), ('EURGBP', [('EURUSD', +1), ('GBPUSD', -1)]),
       ('EURAUD', [('EURUSD', +1), ('AUDUSD', -1)])]

def K_S2(): return -z
def K_S4(W=32):
    s = pd.DataFrame(0.0, index=idx, columns=cols)
    for cross, legs in TRI:
        g = logp[cross].copy()
        for leg, sg in legs: g = g - sg * logp[leg]
        d = zdev(g.to_frame('x'), W)['x']
        s[cross] = s[cross] - d
        for leg, sg in legs: s[leg] = s[leg] + sg * d
    return s
def K_S7(W=48):
    s = pd.DataFrame(0.0, index=idx, columns=cols)
    for a, b in [('AUDUSD', 'NZDUSD'), ('EURUSD', 'GBPUSD')]:
        d = zdev((logp[a] - logp[b]).to_frame('x'), W)['x']
        s[a] -= d; s[b] += d
    return s
def K_S8(K=16):
    hi = logp.rolling(K, min_periods=K // 2).max(); lo = logp.rolling(K, min_periods=K // 2).min()
    return -((logp - lo) / (hi - lo).replace(0, np.nan) - 0.5)

# ---------------- new mathematical candidates ----------------
def M1_ou(W=12):
    return -zdev(logp, W)

def M2_varratio(q=4, WV=48):
    var1 = rets.rolling(WV, min_periods=WV // 2).var()
    varq = logp.diff(q).rolling(WV, min_periods=WV // 2).var()
    vr = varq / (q * var1)
    gate = (1 - vr).clip(lower=0, upper=1)         # >0 only when anti-persistent
    return (-z) * gate

def M4_jump(K=16, thr=3.0):
    bpv = np.sqrt((np.pi / 2) * (rets.abs() * rets.abs().shift(1)).rolling(K, min_periods=K // 2).mean())
    L = (rets / bpv.replace(0, np.nan)).clip(-15, 15)
    return (-L).where(L.abs() > thr, 0.0)          # only fade detected jumps

def M6_rvolrev(K=8):
    rv = rets.rolling(K, min_periods=K // 2).std()
    return -rets / rv.replace(0, np.nan)

def M7_amihud(K=32):
    illiq = (rets.abs() / volp).rolling(K, min_periods=K // 2).mean()
    w = illiq.rank(axis=1, pct=True)               # cross-sectional illiquidity percentile
    return (-z) * w

def M8_volshock():
    volz = rollz(volp, 48).clip(lower=0)           # positive volume surprise only
    return (-z) * volz

def M9_ofi(K=8):
    signed = np.sign(rets) * volp
    return -rollz(signed.rolling(K, min_periods=K // 2).sum(), 48)

def M10_kalman(dv=1e-4, ve=1e-3):
    """random-walk-beta Kalman hedge for 2 blocs; spread z-score -> reversion."""
    s = pd.DataFrame(0.0, index=idx, columns=cols)
    for a, b in [('AUDUSD', 'NZDUSD'), ('EURUSD', 'GBPUSD')]:
        y = logp[a].values; x = logp[b].values
        beta = np.zeros(len(y)); P = 1.0; bcur = 1.0
        spread = np.full(len(y), np.nan)
        for t in range(len(y)):
            if not (np.isfinite(y[t]) and np.isfinite(x[t])):
                beta[t] = bcur; continue
            P += dv
            err = y[t] - bcur * x[t]
            Kg = P * x[t] / (x[t] * P * x[t] + ve)
            bcur = bcur + Kg * err
            P = (1 - Kg * x[t]) * P
            beta[t] = bcur
            spread[t] = y[t] - bcur * x[t]
        sp = pd.Series(spread, index=idx)
        d = zdev(sp.to_frame('x'), 48)['x']
        s[a] -= d; s[b] += d
    return s

def M11_llag(leader='EURUSD'):
    zl = z[leader]
    corr_sign = z.rolling(96, min_periods=48).corr(zl).apply(np.sign)  # dynamic sign vs leader
    return corr_sign.mul(zl, axis=0)               # continuation of leader's move (lagged by 1 in eval)

def M13_multihzn():
    return -(z + zdev(logp, 4) + zdev(logp, 16)) / 3.0

def M14_autocorr(K=48):
    ac = rets.rolling(K, min_periods=K // 2).corr(rets.shift(1))
    return (-z) * (-ac).clip(lower=0)              # revert harder when anti-persistent

KEEPERS = {'#027': None, 'S2': K_S2, 'S4': K_S4, 'S7': K_S7, 'S8': K_S8}
NEW = {'M1_ou': M1_ou, 'M2_varratio': M2_varratio, 'M4_jump': M4_jump, 'M6_rvolrev': M6_rvolrev,
       'M7_amihud': M7_amihud, 'M8_volshock': M8_volshock, 'M9_ofi': M9_ofi, 'M10_kalman': M10_kalman,
       'M11_llag': M11_llag, 'M13_multihzn': M13_multihzn, 'M14_autocorr': M14_autocorr}

def neu(sig): return sig.sub(sig.mean(axis=1), axis=0)
def unit(sig):
    w = neu(sig); g = w.abs().sum(axis=1)
    return w.div(g.where(g > 0), axis=0).fillna(0.0)

print('computing #027 s-score (slow)...', flush=True)
s027, _, _ = e.compute_sscore(mid, cols)
keeper_books = {'#027': unit(s027)}
for k, fn in KEEPERS.items():
    if fn is not None: keeper_books[k] = unit(fn())

def pooled_corr(A, B):
    a = A.values.ravel(); b = B.values.ravel()
    m = np.isfinite(a) & np.isfinite(b)
    return np.corrcoef(a[m], b[m])[0, 1] if m.sum() > 100 else np.nan

def ic_lag(wn, lag):
    a = wn.shift(lag).values.ravel(); b = ret_neutral.values.ravel()
    m = np.isfinite(a) & np.isfinite(b)
    return (np.corrcoef(a[m], b[m])[0, 1], m.sum()) if m.sum() > 100 else (np.nan, 0)

print(f'\n#028d MATH candidate batch | {len(idx):,} M15 bars | screening {len(NEW)} new vs {len(keeper_books)} keepers')
hdr = f'{"cand":13} {"IC1":>8} {"IC2":>8} {"sigma":>6} {"turn/d":>7} {"maxCorrPool":>12} {"vs":>7}  verdict'
print(hdr); print('-' * len(hdr))
new_books = {}
for name, fn in NEW.items():
    wn = unit(fn())
    new_books[name] = wn
    ic1, n = ic_lag(wn, 1); ic2, _ = ic_lag(wn, 2)
    sig_sd = abs(ic1) * np.sqrt(n) if n else 0
    g = wn.abs().sum(axis=1); wu = wn.div(g.where(g > 0), axis=0).fillna(0.0)
    turn = (wu - wu.shift(1)).abs().sum(axis=1).mean() * BARS_PER_DAY
    corrs = {k: abs(pooled_corr(wn, b)) for k, b in keeper_books.items()}
    mx = max(corrs, key=corrs.get); mxv = corrs[mx]
    real = sig_sd > 5
    uniq = mxv < 0.30
    verdict = 'KEEP' if (real and uniq) else ('redundant' if real else 'weak')
    print(f'{name:13} {ic1:+8.4f} {ic2:+8.4f} {sig_sd:6.1f} {turn:7.1f} {mxv:12.2f} {mx:>7}  {verdict}')

# per-year IC (decay) for the new candidates
print('\n=== per-year IC lag1 (decay check) ===')
yrs = sorted(set(idx.year)); yr_of = idx.year.values
print(f'{"cand":13}' + ''.join(f'{y:>7}' for y in yrs))
for name, wn in new_books.items():
    line = f'{name:13}'
    a_all = wn.shift(1).values
    for y in yrs:
        rm = (yr_of == y)
        a = a_all[rm].ravel(); b = ret_neutral.values[rm].ravel()
        m = np.isfinite(a) & np.isfinite(b)
        line += f'{(np.corrcoef(a[m], b[m])[0,1] if m.sum()>100 else np.nan):>+7.3f}'
    print(line)

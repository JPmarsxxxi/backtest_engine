"""#027 - Information Coefficient of the residual s-score: does the signal actually predict?
IC = pooled corr(s-score at t-lag, factor-neutral residual return at t). lag=1 = trade at the
signal-forming close (idealized); lag>=2 = realistic execution latency. Separates signal skill
from cost/turnover. Kernel-independent (pure vectorized recompute of x027b's residual+s-score)."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import x027b_intraday_statarb as e

panel, cols = e.load()
W_DAYS, M, L, VOLW, GAP = e.W_DAYS, e.M, e.L, e.VOLW, e.GAP_MIN
N = len(cols)

rets = np.log(panel).diff()
rets[(panel.index.to_series().diff() > pd.Timedelta(f"{GAP}min")).values] = np.nan
z = (rets / rets.rolling(VOLW, min_periods=VOLW // 2).std()).clip(-8, 8)

day = panel.index.normalize(); udays = pd.Index(day.unique())
resid = pd.DataFrame(np.nan, index=panel.index, columns=cols)
for k, d in enumerate(udays):
    if k < W_DAYS:
        continue
    Zw = z[day.isin(udays[k - W_DAYS:k])].dropna()
    if len(Zw) < N * 5:
        continue
    C = np.nan_to_num(np.corrcoef(Zw.values.T))
    ev, V = np.linalg.eigh(C)
    Vk = V[:, np.argsort(ev)[::-1][:M]]
    P = Vk @ Vk.T
    dm = (day == d)
    Zd = z.values[dm]
    resid.values[dm] = Zd - Zd @ P.T

X = resid.fillna(0.0).rolling(L, min_periods=L).sum()
s = -(X - X.rolling(L, min_periods=L).mean()) / X.rolling(L, min_periods=L).std().replace(0, np.nan)

print(f"#027 IC of residual s-score (W={W_DAYS}d M={M} L={L} vol={VOLW}) - predicts forward residual return")
print("lag=1: trade at signal-forming close (idealized) | lag>=2: realistic execution latency\n")
for lag in [1, 2, 3, 4]:
    a = s.shift(lag).values.ravel()
    b = resid.values.ravel()
    m = np.isfinite(a) & np.isfinite(b)
    ic = np.corrcoef(a[m], b[m])[0, 1]
    ric = np.corrcoef(rankdata(a[m]), rankdata(b[m]))[0, 1]
    ir = ic * np.sqrt(96 * 252)      # IR = IC * sqrt(bets/yr), one bet/bar/pair upper bound
    print(f"  lag {lag} (+{lag*15}min): IC {ic:+.4f}  rank-IC {ric:+.4f}   (implied IR~{ir:+.1f}, n={m.sum():,})")

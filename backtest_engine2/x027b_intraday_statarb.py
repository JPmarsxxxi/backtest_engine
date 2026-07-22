"""
#027 — Intraday PCA-residual statistical arbitrage (Avellaneda-Lee) on the M15 FX cross-section.
The stat-arb family's live frontier: does the FX cross-section's factor-neutral residual mean-revert
at the HOUR scale (swap-free, high-breadth)? GROSS-first / decay-first — measure (a) residual OU
half-life in bars/hours, (b) gross Sharpe of the residual-reversal s-score book, (c) turnover.
Cost (spread, NO swap) only if gross has life.

Construction:
  - M15 log returns, weekend-gap returns masked
  - standardize returns by trailing-window vol
  - DAILY re-estimated PCA on trailing W days -> top-M eigenportfolios (dollar/JPY/EUR factors)
  - residual_z = z - V(V'z)   (factor-neutral)
  - OU state X_i = trailing-L-bar cumulative residual ; s-score = -(X - mean)/std over L
  - book = cross-sectionally demeaned s-score, dollar-neutral, unit gross, trade next bar
Data: data/fx_intraday_m15.parquet (from x027_pull_fx_intraday.py).
"""
import sys
import numpy as np
import pandas as pd

W_DAYS = 3         # PCA estimation window (trading days); overridden from argv in __main__
M = 3              # factors removed
L = 32             # s-score / OU window (bars; 32*15min = 8h)
VOLW = 96                                                 # vol-standardization window (bars; 1 day)
BARS_PER_DAY = 96                                         # M15, 24h FX
GAP_MIN = 20                                              # consecutive-bar gap threshold (min)
DATA = 'data/fx_intraday_m15.parquet'
S_IN, S_OUT = 1.25, 0.50


def load():
    panel = pd.read_parquet(DATA).sort_index()
    # keep pairs with decent coverage
    keep = panel.columns[panel.notna().mean() > 0.90]
    panel = panel[keep].dropna(how='all')
    return panel, list(keep)


def ou_halflife_bars(x):
    x = pd.Series(x).dropna()
    if len(x) < 50:
        return np.nan
    dx = x.diff().dropna(); lag = x.shift(1).dropna()
    idx = dx.index.intersection(lag.index)
    v = np.var(lag.loc[idx])
    if v == 0:
        return np.nan
    b = np.cov(dx.loc[idx], lag.loc[idx])[0, 1] / v
    return np.log(2) / (-b) if b < 0 else np.nan


def compute_sscore(panel, cols):
    """Shared: factor-neutral residual -> OU s-score. Returns (s, rets, hl_bars_array)."""
    N = len(cols)
    rets = np.log(panel).diff()
    # mask weekend/gap returns
    dt = panel.index.to_series().diff()
    gap = (dt > pd.Timedelta(f'{GAP_MIN}min')).values
    rets[gap] = np.nan
    vol = rets.rolling(VOLW, min_periods=VOLW // 2).std()
    z = (rets / vol).clip(-8, 8)                      # standardized returns, outlier-guarded

    idx = panel.index
    day = idx.normalize()
    udays = pd.Index(day.unique())
    resid = pd.DataFrame(np.nan, index=idx, columns=cols)

    for k, d in enumerate(udays):
        if k < W_DAYS:
            continue
        win_mask = day.isin(udays[k - W_DAYS:k])
        Zw = z[win_mask].dropna()
        if len(Zw) < N * 5:
            continue
        C = np.nan_to_num(np.corrcoef(Zw.values.T))
        evals, evecs = np.linalg.eigh(C)
        V = evecs[:, np.argsort(evals)[::-1][:M]]      # N x M top eigenvectors
        P = V @ V.T                                    # projection onto factor space
        dmask = (day == d)
        Zd = z.values[dmask]
        resid.values[dmask] = Zd - Zd @ P.T            # factor-neutral residual

    # OU state = trailing-L cumulative residual; s-score = -(X - mean)/std over L
    X = resid.fillna(0.0).rolling(L, min_periods=L).sum()
    s = -(X - X.rolling(L, min_periods=L).mean()) / X.rolling(L, min_periods=L).std().replace(0, np.nan)

    # residual half-life: AR(1) reversion of cumulative residual PRICE around its ~1-day trailing mean
    resid_price = resid.fillna(0.0).cumsum()
    dev = resid_price - resid_price.rolling(BARS_PER_DAY, min_periods=BARS_PER_DAY // 2).mean()
    hl = np.array([h for h in (ou_halflife_bars(dev[c]) for c in cols) if np.isfinite(h)])
    return s, rets, hl


def run(panel, cols):
    N = len(cols)
    s, rets, hl = compute_sscore(panel, cols)

    # continuous s-score book, dollar-neutral, unit gross
    w = s.sub(s.mean(axis=1), axis=0)
    g = w.abs().sum(axis=1)
    w = w.div(g.where(g > 0), axis=0).fillna(0.0)
    w_lag = w.shift(1).fillna(0.0)
    port = (w_lag * rets).sum(axis=1)

    ann = np.sqrt(BARS_PER_DAY * 252)
    def stat(r, lab):
        r = r.dropna()
        sh = r.mean() / r.std() * ann if r.std() > 0 else 0
        print(f'  {lab:14}: Sharpe {sh:+.2f}  ann.ret {r.mean()*BARS_PER_DAY*252*100:+.0f}%  '
              f'per-bar {r.mean()*1e4:+.3f}bp')
        return sh

    turn = (w - w_lag).abs().sum(axis=1)
    print(f'\n=== #027b intraday PCA-residual FX stat-arb '
          f'(W={W_DAYS}d, M={M} factors, L={L} bars={L*15/60:.1f}h, N={N}) ===')
    print(f'coverage {panel.index.min()} -> {panel.index.max()}  '
          f'({len(panel)} M15 bars, {panel.index.normalize().nunique()} days)')
    print(f'pairs: {cols}')
    if len(hl):
        print(f'residual OU half-life: median {np.median(hl):.1f} bars = {np.median(hl)*15/60:.1f} h  '
              f'(25-75pct {np.percentile(hl,25):.1f}-{np.percentile(hl,75):.1f} bars)')
    stat(port, 'GROSS')
    print(f'  turnover: {turn.mean()*BARS_PER_DAY:.1f} x/day  (avg |pos| flip/bar {turn.mean():.3f})')
    # per-year
    yr = port.dropna().groupby(lambda t: t.year).agg(
        lambda r: r.mean()/r.std()*ann if r.std() > 0 else 0)
    print('  per-year GROSS Sharpe:', '  '.join(f'{y}:{v:+.1f}' for y, v in yr.items()))
    return port, s


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1: W_DAYS = int(sys.argv[1])
    if len(sys.argv) > 2: M = int(sys.argv[2])
    if len(sys.argv) > 3: L = int(sys.argv[3])
    panel, cols = load()
    run(panel, cols)

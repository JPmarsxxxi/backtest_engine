"""
#026c — PCA residual statistical arbitrage (Avellaneda-Lee) on the daily FX cross-section.
Flagship stat-arb cut. Decay-first / GROSS-first: does the FX cross-section have mean-reverting
idiosyncratic residuals after stripping the top eigenportfolios? Measure (a) residual OU half-life,
(b) gross edge of a residual-reversal (s-score) book. No FTMO swap/cost yet — that's the next gate
if gross has life.

Construction (Avellaneda-Lee 2010):
  - trailing window W of daily standardized returns across N FX pairs
  - PCA -> top m eigenportfolios = common factors (dollar factor, EUR/commodity blocs)
  - regress each pair's returns on the m factor returns -> residual e_i(t)
  - cumulative residual X_i = OU process; s-score_i = -(X_i - mean)/std  (centered reversion signal)
  - book: long low s-score / short high s-score, dollar-neutral, daily rebalance
Data: data/ftmo_daily.parquet (28 FX crosses).
"""
import numpy as np
import pandas as pd

import sys
W = 252            # trailing window for PCA + residual OU
M = int(sys.argv[1]) if len(sys.argv) > 1 else 3   # eigenportfolios (common factors) removed
S_IN, S_OUT = 1.25, 0.50   # Avellaneda s-score entry/exit


def load_fx():
    df = pd.read_parquet('data/ftmo_daily.parquet')
    fx = [s for s in df['symbol'].unique()
          if len(s) == 6 and s[:3] in {'AUD','CAD','CHF','EUR','GBP','JPY','NZD','USD'}
          and s[3:] in {'AUD','CAD','CHF','EUR','GBP','JPY','NZD','USD'}]
    px = df[df['symbol'].isin(fx)].pivot(index='date', columns='symbol', values='close')
    px = px.dropna(how='all').ffill(limit=3)
    # keep pairs with full-ish coverage
    keep = px.columns[px.notna().mean() > 0.95]
    px = px[keep].dropna()
    return px, list(keep)


def ou_halflife(x):
    x = pd.Series(x).dropna()
    dx = x.diff().dropna(); lag = x.shift(1).dropna()
    idx = dx.index.intersection(lag.index)
    b = np.cov(dx.loc[idx], lag.loc[idx])[0, 1] / np.var(lag.loc[idx])
    return np.log(2) / (-b) if b < 0 else np.inf


def run(px, cols):
    rets = np.log(px).diff().dropna()
    dates = rets.index
    N = len(cols)
    print(f'\n=== #026c PCA-residual FX stat-arb (W={W}, M={M} factors, N={N} pairs) ===')
    print(f'pairs: {cols}')
    print(f'coverage {dates.min().date()} -> {dates.max().date()} ({len(dates)} days)')

    sscore = pd.DataFrame(index=dates, columns=cols, dtype=float)
    hl_samples = []
    for t in range(W, len(dates)):
        win = rets.iloc[t-W:t]
        # standardize each pair over the window
        mu, sd = win.mean(), win.std()
        Z = (win - mu) / sd
        # PCA via covariance/corr of standardized returns
        C = np.cov(Z.values.T)
        evals, evecs = np.linalg.eigh(C)
        order = np.argsort(evals)[::-1]
        V = evecs[:, order[:M]]                    # N x M top eigenvectors
        # factor returns over window (eigenportfolio returns): F = Z @ V  (T x M)
        F = Z.values @ V
        # regress each standardized-return series on factors -> residuals
        # betas = (F'F)^-1 F' Z
        FtF_inv = np.linalg.pinv(F.T @ F)
        B = FtF_inv @ F.T @ Z.values               # M x N
        resid = Z.values - F @ B                    # T x N residuals (standardized units)
        # cumulative residual (OU state) per pair
        X = np.cumsum(resid, axis=0)                # T x N
        # s-score = -(X_last - mean_X)/std_X per pair (centered reversion signal)
        Xmu = X.mean(axis=0); Xsd = X.std(axis=0)
        s = -(X[-1] - Xmu) / np.where(Xsd > 0, Xsd, np.nan)
        sscore.iloc[t] = s
        # collect residual half-lives occasionally
        if t % 250 == 0:
            for j in range(N):
                hl = ou_halflife(X[:, j])
                if np.isfinite(hl): hl_samples.append(hl)

    hl_samples = np.array(hl_samples)
    print(f'\nresidual OU half-life (days): median {np.median(hl_samples):.1f}  '
          f'25-75pct {np.percentile(hl_samples,25):.1f}-{np.percentile(hl_samples,75):.1f}')

    # ---- gross book: weight = s-score signal, dollar-neutral, unit gross ----
    # trade when |s|>S_IN, hold until |s|<S_OUT (persist prior weight otherwise)
    sig = sscore.copy()
    # simple continuous version: target weight = s-score, cross-sectionally demeaned & gross-normalized
    w = sig.sub(sig.mean(axis=1), axis=0)
    gross = w.abs().sum(axis=1)
    w = w.div(gross.where(gross > 0), axis=0).fillna(0.0)
    w_lag = w.shift(1).fillna(0.0)                  # trade next day
    port_ret = (w_lag * rets).sum(axis=1)

    def stats(r, label):
        r = r.dropna()
        sh = r.mean()/r.std()*np.sqrt(252) if r.std() > 0 else 0
        to = None
        print(f'  {label:16}: Sharpe {sh:+.2f}  ann.ret {r.mean()*252*100:+.1f}%  cum {r.sum()*100:+.0f}%')
        return sh

    print('\ncontinuous s-score book (GROSS, dollar-neutral, unit gross):')
    stats(port_ret, 'gross')
    turn = (w - w_lag).abs().sum(axis=1)
    print(f'  turnover: {turn.mean()*252:.0f} x/yr (avg daily {turn.mean():.3f})')
    # per year
    yr = port_ret.dropna().groupby(lambda d: d.year).agg(lambda s: s.mean()/s.std()*np.sqrt(252) if s.std()>0 else 0)
    print('  per-year gross Sharpe:')
    print('   ', '  '.join(f'{y}:{v:+.1f}' for y, v in yr.items()))
    return sscore, port_ret


if __name__ == '__main__':
    px, cols = load_fx()
    run(px, cols)

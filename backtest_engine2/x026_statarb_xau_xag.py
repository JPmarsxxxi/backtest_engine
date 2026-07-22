"""
#026 — Statistical arbitrage, cheap first cut: XAU-XAG cointegration + OU half-life diagnostic.

Step 1/2 of the stat-arb family hunt. PURE MEASUREMENT — no strategy, no costs yet
(decay-first discipline). Question: is the gold-silver spread genuinely stationary
(cointegrated), and if so what is the OU mean-reversion half-life? A tradeable pair needs
a stationary spread with a half-life short enough to trade market-neutral (no beta) and
ideally intraday-ish (no swap).

Data: data/ftmo_daily.parquet (Dukascopy daily CFD closes, CFD-faithful).
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

pd.set_option('display.width', 140)


def load_pair(a='XAUUSD', b='XAGUSD'):
    df = pd.read_parquet('data/ftmo_daily.parquet')
    px = df[df['symbol'].isin([a, b])].pivot(index='date', columns='symbol', values='close')
    px = px.dropna()
    return px, a, b


def ou_halflife(spread):
    """Fit AR(1): d spread_t = a + b*spread_{t-1} + e. kappa=-b, halflife=ln2/kappa (bars)."""
    s = pd.Series(spread).dropna()
    ds = s.diff().dropna()
    lag = s.shift(1).dropna()
    idx = ds.index.intersection(lag.index)
    X = add_constant(lag.loc[idx].values)
    res = OLS(ds.loc[idx].values, X).fit()
    b = res.params[1]
    kappa = -b
    hl = np.log(2) / kappa if kappa > 0 else np.inf
    return hl, kappa, res.params[0]


def eg_test(y, x, label):
    """Engle-Granger: regress y on x (+const), ADF the residual. Returns beta, adf stat/p, resid."""
    X = add_constant(x.values)
    res = OLS(y.values, X).fit()
    beta = res.params[1]
    const = res.params[0]
    resid = pd.Series(y.values - (const + beta * x.values), index=y.index)
    adf = adfuller(resid.values, maxlag=1, autolag='AIC')
    print(f'  [{label}] beta={beta:.4f} const={const:.4f} | ADF stat={adf[0]:.3f} p={adf[1]:.4f} '
          f'(crit 5%={adf[4]["5%"]:.3f})')
    return beta, const, resid, adf


def summarize(px, a, b):
    print(f'\n=== #026 XAU-XAG cointegration/OU diagnostic ===')
    print(f'coverage: {px.index.min().date()} -> {px.index.max().date()}  ({len(px)} daily bars)')
    la, lb = np.log(px[a]), np.log(px[b])

    # Gold/silver ratio (the classic "spread") for reference
    ratio = px[a] / px[b]
    print(f'\ngold/silver ratio: min {ratio.min():.1f}  max {ratio.max():.1f}  '
          f'mean {ratio.mean():.1f}  last {ratio.iloc[-1]:.1f}')

    # (1) statsmodels coint (Engle-Granger, tests y~x)
    print('\n(1) Engle-Granger cointegration (log prices):')
    t_ab, p_ab, _ = coint(la, lb)
    print(f'  coint(logXAU, logXAG): t={t_ab:.3f}  p={p_ab:.4f}')

    # (2) explicit hedge-ratio regressions both directions + residual ADF
    print('\n(2) hedge-ratio regressions + residual stationarity:')
    beta1, c1, resid1, adf1 = eg_test(la, lb, 'logXAU ~ logXAG')
    beta2, c2, resid2, adf2 = eg_test(lb, la, 'logXAG ~ logXAU')

    # Choose the residual with the more negative ADF (more stationary)
    use_resid, use_lbl = (resid1, 'logXAU~logXAG') if adf1[0] < adf2[0] else (resid2, 'logXAG~logXAU')

    # (3) OU half-life on the chosen residual + on the raw ratio
    print('\n(3) OU mean-reversion (AR1 half-life, in DAYS):')
    hl_r, k_r, _ = ou_halflife(use_resid)
    hl_ratio, k_ratio, _ = ou_halflife(ratio)
    hl_lr, k_lr, _ = ou_halflife(np.log(ratio))
    print(f'  residual [{use_lbl}]: half-life = {hl_r:.1f} d   (kappa={k_r:.4f})')
    print(f'  raw ratio          : half-life = {hl_ratio:.1f} d   (kappa={k_ratio:.4f})')
    print(f'  log-ratio          : half-life = {hl_lr:.1f} d   (kappa={k_lr:.4f})')

    # (4) current z-score of the residual + how often it crosses
    z = (use_resid - use_resid.mean()) / use_resid.std()
    print(f'\n(4) residual z-score: last={z.iloc[-1]:.2f}  |z|>2 frac={np.mean(np.abs(z)>2):.3f}  '
          f'std(resid)={use_resid.std():.4f}')

    # (5) sub-sample stability: split in half, re-test ADF + beta
    print('\n(5) sub-sample stability (is the relationship stable OOS?):')
    n = len(px); mid = n // 2
    for name, sl in [('first half', slice(0, mid)), ('second half', slice(mid, n))]:
        laa, lbb = la.iloc[sl], lb.iloc[sl]
        bb, cc, rr, aa = eg_test(laa, lbb, f'{name}: logXAU~logXAG')
        hl, kk, _ = ou_halflife(rr)
        print(f'      -> {name} half-life {hl:.1f} d, beta {bb:.3f}')

    return use_resid, z


if __name__ == '__main__':
    import sys
    a = sys.argv[1] if len(sys.argv) > 1 else 'XAUUSD'
    b = sys.argv[2] if len(sys.argv) > 2 else 'XAGUSD'
    px, a, b = load_pair(a, b)
    summarize(px, a, b)

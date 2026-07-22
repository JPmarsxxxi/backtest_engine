"""
#026b — WTI-Brent daily OU-band pairs backtest, NET of real FTMO oil swap.
The honest kill-or-keep: causal rolling hedge ratio + rolling z (no full-sample lookahead),
z-entry/exit bands, real per-leg swap from ftmo_specs. Question: does the spread edge per
cycle beat the -37..-58%/yr swap on the hedged oil pair?
"""
import numpy as np
import pandas as pd

WIN = 250          # rolling window for hedge ratio + z (causal)
Z_IN, Z_OUT = 2.0, 0.5
# annualized swap on each LEG/side, from x026 decode (last px): %/yr
SWAP = {  # (long %/yr, short %/yr)
    'USOIL': (14.9, -68.7),
    'UKOIL': (11.0, -51.8),
}
RT_SPREAD_BP = 3.0   # round-trip cost per leg (bp of notional): retail oil CFD spread+comm, generous


def load():
    df = pd.read_parquet('data/ftmo_daily.parquet')
    px = df[df['symbol'].isin(['USOIL', 'UKOIL'])].pivot(index='date', columns='symbol', values='close').dropna()
    return px


def backtest(px):
    lw, lb = np.log(px['USOIL']), np.log(px['UKOIL'])
    n = len(px)
    # rolling causal hedge ratio beta_t (regress lw on lb over trailing WIN), rolling z of resid
    beta = pd.Series(index=px.index, dtype=float)
    z = pd.Series(index=px.index, dtype=float)
    for t in range(WIN, n):
        y = lw.iloc[t-WIN:t]; x = lb.iloc[t-WIN:t]
        b = np.cov(y, x)[0, 1] / np.var(x)
        a = y.mean() - b * x.mean()
        resid_hist = y - (a + b * x)
        resid_now = lw.iloc[t] - (a + b * lb.iloc[t])
        beta.iloc[t] = b
        z.iloc[t] = (resid_now - resid_hist.mean()) / resid_hist.std()
    # position on the SPREAD: pos=-1 short spread (short WTI/long Brent) when z>Z_IN; +1 when z<-Z_IN; exit |z|<Z_OUT
    pos = pd.Series(0.0, index=px.index)
    cur = 0.0
    for t in range(WIN, n):
        zz = z.iloc[t]
        if cur == 0.0:
            if zz > Z_IN: cur = -1.0
            elif zz < -Z_IN: cur = 1.0
        else:
            if abs(zz) < Z_OUT: cur = 0.0
            elif cur == -1.0 and zz < -Z_IN: cur = 1.0
            elif cur == 1.0 and zz > Z_IN: cur = -1.0
        pos.iloc[t] = cur
    pos = pos.shift(1).fillna(0.0)   # trade next bar (no lookahead on the signal fill)

    # daily spread return: spread = lw - beta*lb ; pos>0 means long spread (long WTI, short Brent)
    dlw, dlb = lw.diff(), lb.diff()
    beta_f = beta.shift(1)
    spread_ret = pos * (dlw - beta_f * dlb)     # gross log-return of the position (per $1 gross on WTI leg)

    # ---- costs ----
    # swap: charged each day a leg is held. daily rate = ann/100/365. legs held whenever pos!=0.
    def dayrate(ann): return ann / 100.0 / 365.0
    # when pos=+1: long WTI (swap long), short Brent*beta (swap short). when pos=-1: opposite.
    swap_daily = pd.Series(0.0, index=px.index)
    long_wti = (pos > 0)
    short_wti = (pos < 0)
    swap_daily[long_wti] = dayrate(SWAP['USOIL'][0]) + beta_f[long_wti].abs() * dayrate(SWAP['UKOIL'][1])
    swap_daily[short_wti] = dayrate(SWAP['USOIL'][1]) + beta_f[short_wti].abs() * dayrate(SWAP['UKOIL'][0])
    # transaction cost on turnover (both legs): when pos changes
    turn = pos.diff().abs().fillna(0.0)
    tcost = -(RT_SPREAD_BP / 1e4) * turn * (1 + beta_f.abs().fillna(1.0))

    gross = spread_ret.fillna(0.0)
    net = gross + swap_daily.fillna(0.0) + tcost.fillna(0.0)

    def stats(r, label):
        r = r.dropna()
        ann_ret = r.mean() * 252
        sh = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
        print(f'  {label:12}: Sharpe {sh:+.2f}  ann.ret {ann_ret*100:+.1f}%  cum {(r.sum())*100:+.0f}%')
        return sh

    ntrades = int((turn > 0).sum())
    print(f'\n=== #026b WTI-Brent daily OU pair (win={WIN}, z {Z_IN}/{Z_OUT}) ===')
    print(f'bars {n}, trades {ntrades}, avg |pos| {pos.abs().mean():.2f}, time-in-market {np.mean(pos!=0):.2f}')
    print(f'swap paid (ann, when in mkt): {swap_daily[pos!=0].mean()*365*100:+.1f}%/yr')
    stats(gross, 'GROSS')
    stats(gross + swap_daily.fillna(0), 'net swap')
    stats(net, 'NET all-in')
    # per-year net
    print('  per-year NET Sharpe:')
    yr = net.groupby(net.index.year).agg(lambda s: s.mean()/s.std()*np.sqrt(252) if s.std()>0 else 0)
    print('   ', '  '.join(f'{y}:{v:+.1f}' for y, v in yr.items()))


if __name__ == '__main__':
    backtest(load())

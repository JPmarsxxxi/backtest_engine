"""
#027c — Discrete-band intraday stat-arb + realistic per-pair FX spread cost. The NET make-or-break.
Reuses x027b.compute_sscore (factor-neutral residual s-score). Instead of rebalancing the full
s-score every bar (36x/day turnover), each pair runs a state machine: OPEN when |s|>S_IN, HOLD,
CLOSE when |s|<S_OUT. Fixed 1/N sizing per position (opening one doesn't churn the others).
Cost = per-pair half-spread on each open/close. NO swap (intraday). Question: does net survive?

Spreads (bp half): majors from #016 MT5-measured (EURUSD/GBPUSD/USDJPY); others = conservative
retail-FTMO estimates (crosses wider). Flagged ESTIMATE -> pull real MT5 spreads if net has life.
"""
import argparse
import numpy as np
import pandas as pd
import x027b_intraday_statarb as e

ap = argparse.ArgumentParser()
ap.add_argument('--sin', type=float, default=1.25)
ap.add_argument('--sout', type=float, default=0.50)
ap.add_argument('--L', type=int, default=32)          # s-score window (bars)
ap.add_argument('--M', type=int, default=3)           # factors removed
ap.add_argument('--W', type=int, default=3)           # PCA window (days)
ap.add_argument('--minhold', type=int, default=0)     # force min holding bars (anti-churn)
ap.add_argument('--majors', action='store_true')      # trade only tight-spread majors (PCA still uses all)
args = ap.parse_args()
S_IN, S_OUT = args.sin, args.sout
e.L, e.M, e.W_DAYS = args.L, args.M, args.W           # decouple from x027b's argv globals

MAJORS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF']  # tight-spread subset

# half-spread in bp. EURUSD/GBPUSD/USDJPY = #016 measured; rest = conservative estimates.
SPREAD_HALF_BP = {
    'EURUSD': 0.10, 'GBPUSD': 0.19, 'USDJPY': 0.16, 'AUDUSD': 0.25, 'NZDUSD': 0.35,
    'USDCAD': 0.30, 'USDCHF': 0.30,
    'EURGBP': 0.45, 'EURAUD': 0.65, 'EURJPY': 0.40, 'GBPJPY': 0.65, 'AUDJPY': 0.55,
}


def discrete_positions(s_df, s_in, s_out, minhold=0):
    """Per-pair state machine -> position matrix in {-1,0,+1}. Long when s>s_in (residual cheap).
    minhold: don't close before held this many bars (anti-churn)."""
    S = s_df.values
    T, N = S.shape
    pos = np.zeros((T, N))
    cur = np.zeros(N)
    held = np.zeros(N)
    for t in range(T):
        row = S[t]
        for i in range(N):
            v = row[i]
            if cur[i] != 0:
                held[i] += 1
            if np.isnan(v):
                pass                                  # hold through missing signal
            elif cur[i] == 0:
                if v > s_in:   cur[i] = 1.0; held[i] = 0
                elif v < -s_in: cur[i] = -1.0; held[i] = 0
            elif held[i] >= minhold:
                if cur[i] == 1.0 and v < s_out:   cur[i] = 0.0
                elif cur[i] == -1.0 and v > -s_out: cur[i] = 0.0
        pos[t] = cur
    return pd.DataFrame(pos, index=s_df.index, columns=s_df.columns)


def main():
    panel, cols = e.load()
    s, rets, hl = e.compute_sscore(panel, cols)
    N = len(cols)
    pos = discrete_positions(s, S_IN, S_OUT, args.minhold)
    if args.majors:
        pos[[c for c in cols if c not in MAJORS]] = 0.0   # PCA uses all; trade only majors
    Ntr = int((pos != 0).any().sum())
    w = pos / Ntr                                     # fixed 1/N sizing per traded position
    w_lag = w.shift(1).fillna(0.0)
    gross = (w_lag * rets).sum(axis=1)

    # cost: per-pair half-spread on |Δposition| (each open/close crosses half-spread)
    dpos = (pos - pos.shift(1)).abs().fillna(0.0)
    half = pd.Series(SPREAD_HALF_BP)[cols] / 1e4
    cost = (dpos * half).sum(axis=1) / Ntr            # /Ntr because weight per position is 1/N
    net = gross - cost

    ann = np.sqrt(e.BARS_PER_DAY * 252)
    def stat(r, lab):
        r = r.dropna()
        sh = r.mean()/r.std()*ann if r.std() > 0 else 0
        print(f'  {lab:12}: Sharpe {sh:+.2f}  ann.ret {r.mean()*e.BARS_PER_DAY*252*100:+.1f}%')
        return sh

    n_trades = int((dpos.sum(axis=1) > 0).sum())
    avg_active = (pos != 0).sum(axis=1).mean()
    turn_day = dpos.sum(axis=1).mean() * e.BARS_PER_DAY
    hold_bars = (pos != 0).sum().sum() / max(1, (dpos.sum().sum() / 2))
    print(f'\n=== #027c discrete-band + cost (W={e.W_DAYS}d M={e.M} L={e.L}b={e.L*15/60:.1f}h, '
          f'S_in={S_IN} S_out={S_OUT}, N={N}) ===')
    print(f'avg active positions {avg_active:.1f}/{N}, avg hold {hold_bars:.1f} bars '
          f'({hold_bars*15/60:.1f}h), turnover {turn_day:.1f} legs/day')
    stat(gross, 'GROSS')
    stat(net, 'NET')
    print(f'  cost drag: {(gross.mean()-net.mean())*e.BARS_PER_DAY*252*100:.1f}%/yr')
    yr = net.dropna().groupby(lambda t: t.year).agg(lambda r: r.mean()/r.std()*ann if r.std()>0 else 0)
    print('  per-year NET Sharpe:', '  '.join(f'{y}:{v:+.1f}' for y, v in yr.items()))
    # per-pair net contribution (drop-one sanity: is it broad or 1 pair?)
    contrib = {}
    for i, c in enumerate(cols):
        gi = (w_lag[c] * rets[c])
        ci = (dpos[c] * half[c]) / Ntr
        contrib[c] = (gi - ci).sum() * 100
    contrib = pd.Series(contrib).sort_values()
    print('  per-pair NET cum% (sorted):')
    print('   ', '  '.join(f'{k}:{v:+.0f}' for k, v in contrib.items()))


if __name__ == '__main__':
    main()

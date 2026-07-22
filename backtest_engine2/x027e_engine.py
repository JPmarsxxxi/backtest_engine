"""
#027e — Run the intraday stat-arb through the VALIDATED backtest engine (backtest package),
as a cross-check on the standalone x027b/c numbers. Precompute causal weights, register them
as a panel feature, replay through Engine with per-pair Spread costs -> the engine does the
position/cost/P&L accounting. If engine net ~= standalone net, the standalone isn't buggy.

Usage: python x027e_engine.py [book] [start_year]
  book: cont12 | majors | tight2   (default cont12)
"""
import sys
import numpy as np
import pandas as pd
from backtest.data import DataPanel
from backtest.engine import Engine
from backtest.strategy import Strategy
from backtest.risk import RiskConfig
from backtest.costs import Spread
import x027b_intraday_statarb as e
import x027c_discrete_cost as d

BOOK = sys.argv[1] if len(sys.argv) > 1 else 'cont12'
START = f'{sys.argv[2]}-01-01' if len(sys.argv) > 2 else '2024-01-01'


def build_weights(s, cols, book):
    if book == 'cont12':
        w = s.sub(s.mean(axis=1), axis=0)
        return w.div(w.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    pairs = {'majors': ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF'],
             'tight2': ['EURUSD', 'USDJPY']}[book]
    sin = 2.0 if book == 'majors' else 3.0
    pos = d.discrete_positions(s[pairs], sin, 0.5)
    w = pd.DataFrame(0.0, index=s.index, columns=cols)
    w[pairs] = pos / len(pairs)
    return w


class SignalReplay(Strategy):
    """Replay precomputed weights (feature 'w'); identity risk so the engine uses them as-is."""
    rebalance_frequency = staticmethod(lambda dates: dates)   # rebalance every bar
    risk = RiskConfig(max_position=1.0, max_gross=2.0, max_net=1.0)

    def generate_weights(self, data, t):
        return data.feature('w').iloc[-1]

    def apply_risk(self, proposed, state, data):
        return proposed


def main():
    panel, cols = e.load()
    s, rets, _ = e.compute_sscore(panel, cols)
    W = build_weights(s, cols, BOOK).reindex(columns=cols)
    prices = panel[cols].loc[START:]
    W = W.loc[START:]
    dp = DataPanel(prices=prices, features={'w': W.fillna(0.0)})

    half = pd.Series(d.SPREAD_HALF_BP)[cols]
    res = Engine(dp, SignalReplay(), costs=Spread(half_bps=half)).run()

    ann = np.sqrt(96 * 252)
    r = res.returns.dropna()
    gross_r = (res.gross_pnl / res.equity_curve.shift(1)).dropna()
    def sh(x): return x.mean() / x.std() * ann if x.std() > 0 else 0.0
    print(f'\n=== #027e ENGINE run | book={BOOK} | {START}->end | {res.metadata["n_bars"]} bars ===')
    print(f'  ENGINE   NET Sharpe {sh(r):+.2f}  ann.ret {r.mean()*96*252*100:+.1f}%  '
          f'gross Sharpe {sh(gross_r):+.2f}')
    print(f'  total cost {res.costs.sum():,.0f} on {res.metadata["initial_capital"]:,.0f} '
          f'= {res.costs.sum()/res.metadata["initial_capital"]*100:.1f}% over period')

    # standalone comparison on the SAME window + book
    ss = s.loc[START:]
    rr = rets.loc[START:]
    if BOOK == 'cont12':
        w = ss.sub(ss.mean(axis=1), axis=0)
        w = w.div(w.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        gross = (w.shift(1) * rr).sum(axis=1)
        cost = ((w - w.shift(1)).abs() * (half / 1e4)).sum(axis=1)
        net = (gross - cost).dropna()
        print(f'  STANDALONE NET Sharpe {sh(net):+.2f}  gross {sh(gross.dropna()):+.2f}  (should ~match engine)')


if __name__ == '__main__':
    main()

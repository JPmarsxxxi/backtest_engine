# Cell 4 — #014 trial 1: single-path run, MOP TSMOM vs real FTMO costs.
from backtest.engine import Engine

START = panel.dates[LOOKBACK + 1]
engine = Engine(panel, strat, costs=costs, liquidity=None, initial_capital=1_000_000)
result = engine.run(start=START)

eq = result.equity_curve
ret = result.returns.dropna()
yrs = (eq.index[-1] - eq.index[0]).days / 365.25
sh = ret.mean() / ret.std() * np.sqrt(252)
dd = (eq / eq.cummax() - 1).min()
print(f"Bars: {len(eq)} | rebalances {result.metadata['n_rebalances']} | "
      f"{eq.index[0].date()} -> {eq.index[-1].date()} ({yrs:.1f}y)")
print(f"Final equity:  ${eq.iloc[-1]:>14,.0f}  (total {(eq.iloc[-1] / 1e6 - 1):+.1%})")
print(f"Net Sharpe:    {sh:+.2f} | ann vol {ret.std() * np.sqrt(252):.1%} | maxDD {dd:.1%}")
print(f"Total costs:   ${result.costs.sum():>14,.0f}")
print(f"Max position:  {result.weights.abs().max().max():.1%} | "
      f"max gross {result.weights.abs().sum(axis=1).max():.2f}")

print(f"\n{'year':>5} {'net ret':>8} {'net Sh':>7} {'maxDD':>7}")
for yr, g in ret.groupby(ret.index.year):
    e = (1 + g).cumprod()
    print(f"{yr:>5} {e.iloc[-1] - 1:>+8.1%} "
          f"{g.mean() / g.std() * np.sqrt(252) if g.std() > 0 else float('nan'):>+7.2f} "
          f"{(e / e.cummax() - 1).min():>7.1%}")

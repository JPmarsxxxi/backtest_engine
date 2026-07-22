# Cell 4 — #019 single-path run: vol-targeted US500 reversal vs real FTMO costs.
# Warmup 130 bars (>lookback 5 + win 120). Annualize with ACTUAL blended bars/yr (~270), not 252.
from backtest.engine import Engine

START = panel.dates[130]
engine = Engine(panel, strat, costs=costs, liquidity=liq, initial_capital=1_000_000)
result = engine.run(start=START)

eq = result.equity_curve
ret = result.returns.dropna()
yrs = (eq.index[-1] - eq.index[0]).days / 365.25
ppy = len(ret) / yrs                                   # actual bars/yr (mixed calendar)
annf = np.sqrt(ppy)
ret_g = (result.gross_pnl / eq.shift(1)).dropna()      # gross (pre-cost) returns
sh = ret.mean() / ret.std() * annf
sh_g = ret_g.mean() / ret_g.std() * annf
dd = (eq / eq.cummax() - 1).min()

print(f"Bars {len(eq)} | rebalances {result.metadata['n_rebalances']} | "
      f"{eq.index[0].date()} -> {eq.index[-1].date()} ({yrs:.1f}y, {ppy:.0f} bars/yr)")
print(f"Final equity:  ${eq.iloc[-1]:>13,.0f}  (total {(eq.iloc[-1]/1e6-1):+.1%})")
print(f"Sharpe:  GROSS {sh_g:+.2f}  |  NET {sh:+.2f}  | ann vol {ret.std()*annf:.1%} | maxDD {dd:.1%}")
print(f"Worst day: {ret.min():+.2%}  (FTMO daily limit -5%) | best day {ret.max():+.2%}")
print(f"Total costs:   ${result.costs.sum():>13,.0f}  ({result.costs.sum()/1e6/yrs*100:.2f}%/yr of init)")
print(f"Max position {result.weights.abs().max().max():.1%} | max gross {result.weights.abs().sum(axis=1).max():.2f} | "
      f"avg gross {result.weights.abs().sum(axis=1).mean():.2f}")

print(f"\n{'yr':>5} {'net ret':>8} {'netSh':>6} {'maxDD':>7} {'worst d':>8}")
for yr, g in ret.groupby(ret.index.year):
    e = (1 + g).cumprod()
    print(f"{yr:>5} {e.iloc[-1]-1:>+8.1%} {g.mean()/g.std()*annf if g.std()>0 else float('nan'):>+6.2f} "
          f"{(e/e.cummax()-1).min():>7.1%} {g.min():>+8.2%}")

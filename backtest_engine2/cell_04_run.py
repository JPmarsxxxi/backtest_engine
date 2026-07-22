# Cell 4 — Single-path engine run (raw reversal backtest)
from backtest.engine import Engine

WARMUP_BARS = 21
engine = Engine(panel, strat, costs=costs, liquidity=liq, initial_capital=1_000_000)
result = engine.run(start=panel.dates[WARMUP_BARS])

print(f"Bars:                {len(result.equity_curve)}")
print(f"Rebalances:          {result.metadata['n_rebalances']}")
print(f"Date range:          {result.metadata['start'].date()} -> {result.metadata['end'].date()}")
print(f"Initial:             ${result.metadata['initial_capital']:>14,.0f}")
print(f"Final equity:        ${result.equity_curve.iloc[-1]:>14,.0f}")
print(f"Total return:        {(result.equity_curve.iloc[-1]/result.metadata['initial_capital']-1):+.2%}")
print(f"Total costs:         ${result.costs.sum():>14,.0f}")
print(f"Max position weight: {result.weights.abs().max().max():.2%}")
print(f"Max gross exposure:  {result.weights.abs().sum(axis=1).max():.2%}")

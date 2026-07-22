# Cell 6 — Stage 4: single-path engine run (one-day warmup for the 20-bar ADV lookback).
from backtest.engine import Engine

WARMUP_BARS = 48  # one UTC day of 30m bars

engine = Engine(panel, strat, costs=costs, liquidity=liq, initial_capital=1_000_000)
result = engine.run(start=panel.dates[WARMUP_BARS])

print(f"Bars:          {len(result.equity_curve)}")
print(f"Rebalances:    {result.metadata['n_rebalances']}")
print(f"Date range:    {result.metadata['start']} -> {result.metadata['end']}")
print(f"Initial:       ${result.metadata['initial_capital']:>14,.0f}")
print(f"Final equity:  ${result.equity_curve.iloc[-1]:>14,.0f}")
print(f"Total return:  {(result.equity_curve.iloc[-1] / result.metadata['initial_capital'] - 1):+.2%}")
print(f"Total costs:   ${result.costs.sum():>14,.0f}")
print(f"Trading days w/ a position: {int((result.trades.abs().sum(axis=1) > 0).sum() / 2)} round trips approx")
print(f"Unfilled (liq-capped) total: ${result.unfilled.abs().sum().sum():,.0f}")
print(f"Max position weight:  {result.weights.abs().max().max():.2%}")
print(f"Max gross exposure:   {result.weights.abs().sum(axis=1).max():.2%}")

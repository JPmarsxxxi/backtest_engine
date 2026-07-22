# Cell 4 — Single-path run: SkewLottery on FTMO-14, FTMO cost stack
from backtest.engine import Engine

WARMUP_BARS = 30   # strategy needs 26 (21 window + 5 decay)

engine_skew = Engine(
    panel_skew,
    strat_skew,
    costs=costs_ftmo,
    liquidity=liq_ftmo,
    initial_capital=1_000_000,
)
result_skew = engine_skew.run(start=panel_skew.dates[WARMUP_BARS])

print(f"Bars:          {len(result_skew.equity_curve)}")
print(f"Rebalances:    {result_skew.metadata['n_rebalances']}")
print(f"Date range:    {result_skew.metadata['start'].date()} -> {result_skew.metadata['end'].date()}")
print(f"Initial:       ${result_skew.metadata['initial_capital']:>14,.0f}")
print(f"Final equity:  ${result_skew.equity_curve.iloc[-1]:>14,.0f}")
print(f"Total return:  {(result_skew.equity_curve.iloc[-1] / result_skew.metadata['initial_capital'] - 1):+.2%}")
print(f"Total costs:   ${result_skew.costs.sum():>14,.0f}")
print(f"Max position weight:  {result_skew.weights.abs().max().max():.2%}")
print(f"Max gross exposure:   {result_skew.weights.abs().sum(axis=1).max():.2%}")

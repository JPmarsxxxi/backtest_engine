# Cell 5 — Stage 4: single-path run, trial 1 (paper window 2-3am ET, long-only).
from backtest.engine import Engine

WARMUP_BARS = 48
engine = Engine(panel, strat, costs=costs, liquidity=None, initial_capital=1_000_000)
result = engine.run(start=panel.dates[WARMUP_BARS])

print(f"Bars:          {len(result.equity_curve)}")
print(f"Rebalances:    {result.metadata['n_rebalances']}")
print(f"Final equity:  ${result.equity_curve.iloc[-1]:>14,.0f}")
print(f"Total return:  {(result.equity_curve.iloc[-1] / 1_000_000 - 1):+.2%}")
print(f"Total costs:   ${result.costs.sum():>14,.0f}")
print(f"Max position weight:  {result.weights.abs().max().max():.2%}")

# Gross per-night quality: window return = P(exit bar)/P(entry bar) - 1 on held nights.
px_ = prices["USA500"]
held = result.positions["USA500"]
entry_stamps = held.index[(held.abs() > 1.0).values & (_et_hour(held.index).hour == 2)]
exit_stamps = entry_stamps + pd.Timedelta("1h")
ret_n = pd.Series(px_.reindex(exit_stamps).values / px_.reindex(entry_stamps).values - 1.0,
                  index=entry_stamps).dropna()
sh_n = ret_n.mean() / ret_n.std() * np.sqrt(250)
print(f"\nGross window: {len(ret_n)} nights | hit {100 * (ret_n > 0).mean():.1f}% | "
      f"mean {1e4 * ret_n.mean():+.2f} bps/night | ann {250 * ret_n.mean():+.2%}/yr | "
      f"gross Sharpe {sh_n:+.2f}")
print("Paper benchmark: ~+1.4 bps/night, ~+3.6%/yr (1998-2019) | cost gate ~2.0 bp/night")

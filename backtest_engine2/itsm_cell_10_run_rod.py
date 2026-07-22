# Cell 10 — Stage 4 run, trial 2 (rest-of-day variant). Same costs/liq/capital as Cell 6.
engine_rod = Engine(panel, strat_rod, costs=costs, liquidity=liq, initial_capital=1_000_000)
result_rod = engine_rod.run(start=panel.dates[WARMUP_BARS])

print(f"Bars:          {len(result_rod.equity_curve)}")
print(f"Rebalances:    {result_rod.metadata['n_rebalances']}")
print(f"Final equity:  ${result_rod.equity_curve.iloc[-1]:>14,.0f}")
print(f"Total return:  {(result_rod.equity_curve.iloc[-1] / 1_000_000 - 1):+.2%}")
print(f"Total costs:   ${result_rod.costs.sum():>14,.0f}")
print(f"Unfilled (liq-capped) total: ${result_rod.unfilled.abs().sum().sum():,.0f}")
print(f"Max position weight:  {result_rod.weights.abs().max().max():.2%}")
print(f"Max gross exposure:   {result_rod.weights.abs().sum(axis=1).max():.2%}")

# Gross per-trade quality, same construction as Cell 8, for the held window 00:30->24:00
stamps_ = prices["BTCUSDT"].index
hm_ = stamps_.hour * 60 + stamps_.minute
entry_ = stamps_[hm_ == 30]
sign_ = np.sign(result_rod.positions["BTCUSDT"].reindex(entry_).fillna(0.0))
exit_ = entry_.normalize() + pd.Timedelta("1D")
px_ = prices["BTCUSDT"]
ret_ = pd.Series(px_.reindex(exit_).values / px_.reindex(entry_).values - 1.0, index=entry_)
tr_ = (sign_ * ret_)[sign_ != 0].dropna()
sh_ = tr_.mean() / tr_.std() * np.sqrt(365)
print(f"\nGross: {len(tr_)} trades | hit {100 * (tr_ > 0).mean():.1f}% | "
      f"mean {1e4 * tr_.mean():+.2f} bps/trade | gross Sharpe {sh_:+.2f} (vs cost gate ~16.5bp)")

# Cell 24 — Per-year regime robustness for the chosen 30d crypto momentum
from backtest.metrics import yearly_metrics

strat_best = MomentumRanked(lookback=30, clip=1.0, decay=5, neutralize=False)
result_best = Engine(panel_crypto, strat_best, costs=costs_crypto, liquidity=liq_crypto,
                     initial_capital=1_000_000).run(start=panel_crypto.dates[40])

yr = yearly_metrics(result_best.returns, result_best.equity_curve, ann_factor=365)
# annualize per-year sharpe column is already annualized by ann_factor in helper
print("Per-calendar-year (net, after costs):")
print(yr.round(3).to_string())
print()
pos = (yr["sharpe"] > 0).sum()
print(f"Years with positive net Sharpe: {pos}/{len(yr)}")
print(f"Worst year: {yr['sharpe'].idxmin()} (Sharpe {yr['sharpe'].min():+.2f}, "
      f"return {yr.loc[yr['sharpe'].idxmin(),'total_return']:+.1%})")

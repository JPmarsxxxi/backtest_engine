# Cell 4 — Stage 4: single-path OOS sanity run (first 6m test slice per spec).
# Train: 2005-01-01 .. 2014-12-31 (10y). Test: 2015-01-01 .. 2015-06-30 (6m).
import numpy as np
from backtest.engine import Engine

train_dates = panel.dates[panel.dates <= "2014-12-31"]
test_dates = panel.dates[(panel.dates >= "2015-01-01") & (panel.dates <= "2015-06-30")]

strat_run = ForecastToFillGoldStrategy()  # fresh instance — reset T+1 / trade state
engine = Engine(panel, strat_run, costs=costs, liquidity=liq, initial_capital=1_000_000)
result = engine.run(train_dates=train_dates, test_dates=test_dates)

eq = result.equity_curve
ret = result.returns.dropna()
yrs = (eq.index[-1] - eq.index[0]).days / 365.25
ann_vol = ret.std() * np.sqrt(252)
sh = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else float("nan")
dd = (eq / eq.cummax() - 1).min()
active = (result.weights[ASSET].abs() > 1e-4).sum()

print(f"OOS window: {eq.index[0].date()} -> {eq.index[-1].date()} ({len(eq)} bars, {yrs:.2f}y)")
print(f"fit_called: {result.metadata.get('fit_called')} | rebalances: {result.metadata['n_rebalances']}")
print(f"fitted lam: {strat_run._lam:.3f}")
print(f"Final equity: ${eq.iloc[-1]:,.0f}  (P&L {(eq.iloc[-1] / 1e6 - 1):+.2%})")
print(f"Net Sharpe: {sh:+.2f} | ann vol {ann_vol:.2%} | maxDD {dd:.2%}")
print(f"Total costs: ${result.costs.sum():,.2f}  ({1e4 * result.costs.sum() / 1e6:.1f} bps on $1M)")
print(f"Max |weight|: {result.weights[ASSET].abs().max():.4f} | "
      f"mean |weight| active: {result.weights[ASSET].abs()[result.weights[ASSET].abs() > 1e-4].mean():.4f}")
print(f"Active days (|w|>0.01%): {active} / {len(eq)} ({100 * active / len(eq):.1f}%)")
print(f"Unfilled notional sum: ${result.unfilled.abs().sum().sum():,.0f}")

eq.tail(3)

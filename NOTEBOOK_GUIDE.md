# Notebook Guide

The minimum-friction path from a notebook to a fully-evaluated backtest. Three lines for the common case.

---

## Install

```bash
pip install -e .[viz]   # the [viz] extra pulls in matplotlib for plotting
```

---

## The 3-line walk-forward

```python
from backtest.data import DataPanel
from backtest.engine import Engine

result = Engine(DataPanel(prices_df), MyStrategy()).run()
result.summary()
```

`prices_df` is a `pd.DataFrame[date × asset]` of adjusted prices. `MyStrategy` is your subclass of `backtest.strategy.Strategy` (see `STRATEGY_GUIDE.md`).

`result.summary()` does all of:

- Computes the full `MetricsReport` (Sharpe + SD + CI, Sortino, Calmar, modified Sharpe, PSR, MinTRL, VaR/cVaR, max DD, time-under-water, total/annualized return, annualized vol).
- Prints / `display()`s it (rich HTML table inside Jupyter, plain text elsewhere).
- Plots a 2×2 dashboard: equity curve, drawdown, rolling Sharpe, asymptotic SR distribution.
- Returns the `MetricsReport` so you can keep working with it.

```python
rep = result.summary()
rep.psr_at(1.0)              # PSR vs SR=1
rep.sharpe_ci_low, rep.sharpe_ci_high
rep.to_dict()                # for logging / comparison
```

---

## With volume, costs, and liquidity caps

```python
from backtest.costs import Commission, Spread, MarketImpact, CompositeCostModel, LiquidityCap

panel = DataPanel(prices_df, volume=volume_df)
costs = CompositeCostModel([
    Commission(bps=1.0),
    Spread(half_bps=2.0),
    MarketImpact(k=10.0, kind="sqrt"),
])
liq = LiquidityCap(cap_pct=0.05)

result = Engine(panel, MyStrategy(), costs=costs, liquidity=liq).run()
result.summary()
```

---

## Custom risk

```python
from backtest.risk import RiskConfig

class MyStrategy(Strategy):
    rebalance_frequency = "weekly"
    risk = RiskConfig(
        max_position=0.10,
        max_gross=1.0,
        max_net=0.5,
        target_vol=0.10,    # annualized vol target
    )
    ...
```

Risk is applied *after* `generate_weights` returns. If you want fully bespoke risk logic, override `apply_risk(self, proposed, state, data)` instead.

---

## Multi-path CPCV

```python
from backtest.engine import MultiPathEngine
from backtest.splitters import CombinatorialPurgedCV

cv = CombinatorialPurgedCV(n_splits=6, n_test_groups=2, embargo_pct=0.01)
mp = MultiPathEngine(panel, MyStrategy(), cv).run(n_jobs=-1)
mp.summary()
```

`mp.summary()` prints the per-path-aggregated metrics table (mean / std / min / max for each metric across paths) and plots the per-path equity overlay + per-path Sharpe distribution.

For paper-style selection-bias correction:

```python
from backtest.selection import dsr, effective_k

# Effective number of independent trials, accounting for correlated paths
K_eff = effective_k(mp.path_returns, threshold=0.5)

# Deflated Sharpe Ratio against that K
print(dsr(mp.path_returns.mean(axis=1), K=K_eff))
```

---

## Persisting trials with the registry

```python
from backtest.registry import TrialRegistry

reg = TrialRegistry("./trial_store")
result = reg.run(
    Engine(panel, MyStrategy(lookback=60)),
    causal_graph_path="diagrams/momentum_thesis.png",
    family="momentum_xs",
)
result.summary()

# DSR using all logged trials in this family as K
print(reg.dsr_for(1, family="momentum_xs", method="effective"))
```

`causal_graph_path` is required by default — pass `skip_causal_graph=True` to flag the run as exploratory and exclude it from K computations.

---

## Customizing the dashboard

`summary()` accepts a few knobs:

```python
result.summary(
    ann_factor=252,        # 12 for monthly returns, 52 for weekly
    ci_level=0.95,         # CI band on the SR distribution plot
    rolling_window=60,     # lookback for the rolling Sharpe panel
    figsize=(13, 8),       # matplotlib figure size
)
```

If you want a single plot, the underlying functions are public:

```python
from backtest.metrics.plot import plot_sharpe_distribution
plot_sharpe_distribution(result.returns, ann_factor=252, ci_level=0.95)
```

If you want just the metrics with no plots:

```python
from backtest.metrics import compute_metrics
rep = compute_metrics(result.returns, result.equity_curve)
rep   # in a Jupyter cell, renders as an HTML table
```

---

## What you get back from `Engine.run()`

```python
result.equity_curve   # pd.Series indexed by date
result.returns        # pd.Series, bar-to-bar % returns
result.weights        # pd.DataFrame[date × asset]
result.positions      # pd.DataFrame, dollar exposures
result.trades         # pd.DataFrame, dollar trades per bar
result.costs          # pd.Series, total cost per bar
result.unfilled       # pd.DataFrame, dollars not filled by liquidity cap
result.metadata       # dict: start, end, n_bars, n_rebalances, fit_called, initial_capital
```

All eight are public. `summary()` is the convenience layer over `equity_curve` + `returns`; everything else is there for custom analysis.

---

## Common gotchas

- **`prices_df` must have a `DatetimeIndex` and be sorted ascending.** The `DataPanel` constructor will raise if not.
- **The default outlier check warns on real spikes.** If your data legitimately has 50%+ moves (e.g., crypto on COVID-19), pass `DataPanel(prices, check_outliers=False)` to silence — but verify they aren't bad ticks first.
- **`MyStrategy` needs a stable `__repr__`.** The trial registry hashes by `repr(strategy)`. Default Python `repr` includes the memory address, which means two identical strategies hash differently. See `STRATEGY_GUIDE.md`.
- **`fit()` only runs if you pass `train_dates`.** For pure walk-forward without a train/test split, `fit()` is never called.
- **Multi-path runs deepcopy the strategy per split.** State on `self` doesn't leak between paths. Module-level globals do.

---

## End-to-end example

```python
import numpy as np
import pandas as pd
from backtest.data import DataPanel
from backtest.engine import Engine
from backtest.risk import RiskConfig
from backtest.strategy import Strategy

# -- your data --
rng = np.random.default_rng(0)
idx = pd.date_range("2020-01-01", periods=1260, freq="B")
prices = pd.DataFrame(
    np.cumprod(1 + rng.normal(0.0003, 0.015, (1260, 5)), axis=0) * 100,
    index=idx, columns=list("ABCDE"),
)

# -- your strategy --
class XSMomentum(Strategy):
    rebalance_frequency = "monthly"
    risk = RiskConfig(max_position=0.30, max_gross=1.0, max_net=1.0)
    def __init__(self, lookback=60):
        self.lookback = lookback
    def __repr__(self):
        return f"XSMomentum(lookback={self.lookback})"
    def generate_weights(self, data, t):
        if len(data.prices) < self.lookback + 1:
            return pd.Series(0.0, index=data.assets)
        rets = data.prices.iloc[-1] / data.prices.iloc[-self.lookback - 1] - 1
        rets = rets.reindex(data.assets).dropna()
        if len(rets) < 2:
            return pd.Series(0.0, index=data.assets)
        med = rets.median(); n = len(rets)
        w = pd.Series(0.0, index=rets.index)
        w[rets > med] =  1.0 / n
        w[rets < med] = -1.0 / n
        return w

# -- run --
result = Engine(DataPanel(prices), XSMomentum(60)).run(start=idx[120])
rep = result.summary()
```

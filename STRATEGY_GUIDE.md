# Strategy Authoring Guide

This guide is for translating a strategy from a notebook or sketch into a class the engine can run. Feed this file plus the notebook to an LLM and it should produce a complete, engine-compatible Strategy file.

---

## Quick start

A strategy is a Python class that:

1. Subclasses `backtest.strategy.Strategy`
2. Sets a class attribute `rebalance_frequency`
3. Implements `generate_weights(self, data, t)` returning a `pd.Series` of target weights

Everything else is optional.

---

## The contract

```python
from backtest.strategy import Strategy
from backtest.risk import RiskConfig
import pandas as pd

class MyStrategy(Strategy):
    rebalance_frequency = "weekly"           # required
    risk = RiskConfig(max_position=0.20)     # optional, defaults to permissive

    def __init__(self, lookback=60):
        self.lookback = lookback             # all params on self

    def __repr__(self):                      # required for trial-registry hashing
        return f"MyStrategy(lookback={self.lookback})"

    def required_data(self):                 # optional, declares dependencies
        return {"prices": None}

    def fit(self, data):                     # optional, called once per train window
        pass

    def generate_weights(self, data, t):     # required
        # build and return a pd.Series of target weights
        ...

    def apply_risk(self, proposed, state, data):   # optional, full custom risk logic
        return ...
```

---

## The `data` object (DataView)

The engine hands `generate_weights` a `DataView` snapshot. Only past data is exposed — there is no way to look ahead.

| Attribute / method | Type | What it gives you |
|---|---|---|
| `data.prices` | `pd.DataFrame` | All prices up to and including `t`. DatetimeIndex × asset columns. |
| `data.volume` | `pd.DataFrame` or `None` | Same shape as prices. `None` if not provided. |
| `data.feature(name)` | `pd.DataFrame` | Optional named feature panel registered with the panel. |
| `data.has_feature(name)` | `bool` | Quick existence check. |
| `data.assets` | `pd.Index` | Eligible assets at time `t` (universe filter applied, NaN prices excluded). |
| `data.t` | `pd.Timestamp` | The current bar timestamp. |

**Important:** `data.prices` includes only rows with index `≤ t`. Slicing further into the future is impossible — the rows aren't there.

---

## What to return from `generate_weights`

A `pd.Series` of target weights indexed by asset ID.

```python
return pd.Series({"AAPL": 0.5, "MSFT": -0.3, "GOOG": 0.2})
```

Rules:

- Weights are **fractions of total portfolio equity**, not dollar amounts.
- Negative weights = short. Positive = long.
- Sum doesn't need to equal 1 (long-short can sum to 0; long-only can leave cash).
- Assets not in `data.assets` are silently zeroed by the engine.
- Weights are then clipped/scaled by your `RiskConfig` before becoming trades.

For empty / insufficient-data cases, return zeros:

```python
return pd.Series(0.0, index=data.assets)
```

---

## `rebalance_frequency` options

| Value | Meaning |
|---|---|
| `"daily"` / `"D"` / `"B"` | Every bar |
| `"weekly"` | First bar of each ISO week |
| `"monthly"` | First bar of each month |
| `"quarterly"` | First bar of each quarter |
| `"yearly"` | First bar of each year |
| `"W-FRI"`, `"BMS"`, etc. | Any pandas frequency alias |
| `lambda dates: dates[::5]` | Custom callable |

Engine only calls `generate_weights` on rebalance dates. On non-rebalance bars, positions drift with prices; nothing is traded.

---

## `RiskConfig` integration

Set on the class. Engine applies it after your weights are returned.

```python
risk = RiskConfig(
    max_position=0.20,    # cap any single asset weight at ±20%
    max_gross=1.0,        # total leverage (sum |weights|) ≤ 1.0
    max_net=1.0,          # net exposure (sum weights) ≤ 1.0
    target_vol=None,      # if set, scale weights to hit annualized vol target
    vol_lookback=60,      # bars used for vol estimate
)
```

Default (when not set) is `RiskConfig(max_position=0.20, max_gross=1.0, max_net=1.0)`.

For fully custom risk logic, override `apply_risk` instead of using the config.

---

## Worked patterns

### Pattern 1 — Stateless rule-based

```python
import numpy as np
import pandas as pd
from backtest.strategy import Strategy

class RSIThresholds(Strategy):
    rebalance_frequency = "daily"

    def __init__(self, period=14, oversold=30, overbought=70):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def __repr__(self):
        return f"RSIThresholds(period={self.period}, oversold={self.oversold}, overbought={self.overbought})"

    def generate_weights(self, data, t):
        if len(data.prices) < self.period + 1 or len(data.assets) == 0:
            return pd.Series(0.0, index=data.assets)
        diffs = data.prices[data.assets].diff().tail(self.period)
        gains = diffs.clip(lower=0).mean()
        losses = -diffs.clip(upper=0).mean()
        rs = gains / losses.replace(0, np.nan)
        rsi = 100 - 100 / (1 + rs)
        n = len(data.assets)
        w = pd.Series(0.0, index=data.assets)
        w[rsi < self.oversold] = 1.0 / n
        w[rsi > self.overbought] = -1.0 / n
        return w
```

### Pattern 2 — Cross-sectional ranking

```python
import pandas as pd
from backtest.strategy import Strategy
from backtest.risk import RiskConfig

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
        ret = data.prices.iloc[-1] / data.prices.iloc[-self.lookback - 1] - 1
        ret = ret.reindex(data.assets).dropna()
        if len(ret) < 2:
            return pd.Series(0.0, index=data.assets)
        median = ret.median()
        n = len(ret)
        w = pd.Series(0.0, index=ret.index)
        w[ret > median] = 1.0 / n
        w[ret < median] = -1.0 / n
        return w
```

### Pattern 3 — Pre-fit ML model

`fit()` is called once on the train window before the test loop starts. State stored on `self` is preserved.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from backtest.strategy import Strategy

class RidgePredict(Strategy):
    rebalance_frequency = "weekly"

    def __init__(self, lookback=20, alpha=1.0):
        self.lookback = lookback
        self.alpha = alpha
        self.model = None
        self.feature_assets = None

    def __repr__(self):
        return f"RidgePredict(lookback={self.lookback}, alpha={self.alpha})"

    def _features(self, prices):
        # Past N-bar returns per asset, flattened
        rets = prices.pct_change().tail(self.lookback)
        return rets.fillna(0).values.flatten()[None, :]

    def fit(self, data):
        prices = data.prices
        self.feature_assets = list(data.assets)
        rows, labels = [], []
        for i in range(self.lookback + 1, len(prices) - 1):
            window = prices.iloc[i - self.lookback - 1:i]
            future_ret = prices.iloc[i + 1].mean() / prices.iloc[i].mean() - 1
            rows.append(self._features(window).flatten())
            labels.append(future_ret)
        if rows:
            self.model = Ridge(alpha=self.alpha).fit(np.array(rows), np.array(labels))

    def generate_weights(self, data, t):
        if self.model is None or len(data.prices) < self.lookback + 1:
            return pd.Series(0.0, index=data.assets)
        x = self._features(data.prices.tail(self.lookback + 1))
        prediction = float(self.model.predict(x)[0])
        # Long all if positive prediction, flat otherwise
        n = len(data.assets)
        if prediction > 0 and n > 0:
            return pd.Series(1.0 / n, index=data.assets)
        return pd.Series(0.0, index=data.assets)
```

### Pattern 4 — Pre-computed signals

If your notebook already produced a DataFrame of weights/signals indexed by date, wrap it.

```python
import pandas as pd
from backtest.strategy import Strategy

class FromSignals(Strategy):
    rebalance_frequency = "daily"

    def __init__(self, signals: pd.DataFrame, name: str = "signals"):
        self.signals = signals
        self.name = name

    def __repr__(self):
        return f"FromSignals(name={self.name!r}, shape={self.signals.shape})"

    def generate_weights(self, data, t):
        if t not in self.signals.index:
            return pd.Series(0.0, index=data.assets)
        return self.signals.loc[t].reindex(data.assets).fillna(0.0)
```

---

## Common pitfalls

1. **Look-ahead bias** — `data.prices` is already filtered to `≤ t`. Don't try to load extra data from outside `data`. If you need a feature that's not in the panel, register it on the `DataPanel` with a publication lag, not in the strategy.

2. **Empty / short universe** — at the first bar of a test window, `data.assets` may have 0 or 1 entries. Always handle this with an early return of `pd.Series(0.0, index=data.assets)`.

3. **Insufficient lookback** — `len(data.prices) < self.lookback + 1` is a normal early-window state, not an error. Return zeros.

4. **Weight reindexing** — assets in your weights but not in `data.assets` are silently dropped. If a notebook produced weights for `["AAPL", "MSFT"]` and `data.assets` is just `["AAPL"]`, only AAPL is kept. Reindex defensively when in doubt.

5. **Stateful strategies under multi-path** — `self` state persists across `generate_weights` calls within one Engine.run(). For CPCV multi-path, each split gets its own deepcopy, so state doesn't leak between paths. If you write to a global, that will leak.

6. **`__repr__` stability** — the trial registry hashes by `repr(strategy)`. Default Python `repr` includes the memory address, so two identical strategies hash differently. Always implement `__repr__` to depend only on `__init__` args.

---

## Wiring it in

```python
from backtest.data import DataPanel
from backtest.engine import Engine
from backtest.metrics import compute_metrics

panel = DataPanel(prices_df)               # plus volume= / features= / universe= as needed
engine = Engine(panel, MyStrategy(lookback=60))
result = engine.run(start="2020-01-01", end="2024-12-31")
report = compute_metrics(result.returns, result.equity_curve)
print(report)
```

For multi-path CPCV evaluation:

```python
from backtest.engine import MultiPathEngine
from backtest.splitters import CombinatorialPurgedCV

cv = CombinatorialPurgedCV(n_splits=6, n_test_groups=2, embargo_pct=0.01)
mp = MultiPathEngine(panel, MyStrategy(lookback=60), cv)
mp_result = mp.run(n_jobs=-1)
print(mp_result.aggregate_metrics())
```

For trial-registry persistence with causal-graph requirement:

```python
from backtest.registry import TrialRegistry

reg = TrialRegistry("./trial_store")
reg.run(
    Engine(panel, MyStrategy(lookback=60)),
    causal_graph_path="diagrams/momentum_thesis.png",
    family="momentum_xs",
)
print(reg.dsr_for(1, family="momentum_xs", method="effective"))
```

---

## LLM packaging checklist

When converting a notebook strategy into an engine-compatible file, verify:

- [ ] Class subclasses `backtest.strategy.Strategy`
- [ ] `rebalance_frequency` set as a class attribute
- [ ] `generate_weights(self, data, t)` is the only required method, returns `pd.Series`
- [ ] All notebook hyperparameters moved to `__init__` args (so they appear in `__repr__`)
- [ ] `__repr__` defined and depends only on `__init__` args
- [ ] Empty-universe and insufficient-lookback branches return `pd.Series(0.0, index=data.assets)`
- [ ] Notebook ML training moved into `fit(self, data)`; weights computation in `generate_weights`
- [ ] No data access outside `data.prices`, `data.volume`, `data.feature(name)`
- [ ] `data.prices` not sliced past `t` (already enforced — but no `.loc[future_date]` lookups either)
- [ ] If notebook computed signals into a DataFrame externally, use the `FromSignals` pattern
- [ ] `RiskConfig` set if the notebook had position sizing / leverage / vol targeting logic
- [ ] Imports limited to `backtest.*`, `pandas`, `numpy`, and any libraries the notebook used (e.g. `sklearn`)
- [ ] No global state, no module-level mutable variables

If unsure about a weight calculation, output the literal weights returned by the notebook on a known date and verify the strategy class returns the same values when called with that date.

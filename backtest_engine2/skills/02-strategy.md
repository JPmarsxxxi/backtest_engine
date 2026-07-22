# Skill 02 — Strategy

> **Read `PROTOCOL.md` and `skills/00-overview.md` first.** Cell-loop and announcement rules in PROTOCOL apply.

---

## When to load

- **Stage 2** of every backtest (defining the `Strategy` subclass).
- Any later cell that modifies the strategy class (new param, changed `rebalance_frequency`, custom `apply_risk`, etc.). A modified strategy is a **new trial** — its `__repr__` will hash differently in the registry (`§10` below, `11-registry.md`).

---

## What you'll find here

1. The contract — every required and optional class attribute / method.
2. `rebalance_frequency` — every accepted form, with verified semantics.
3. The execution flow — exactly what the engine does around your `generate_weights` call.
4. The `state` dict — what the engine passes to `apply_risk`.
5. `generate_weights` — return-value contract, engine coercion, eligibility filter.
6. `fit` — when it's called, what it can do.
7. `apply_risk` — default delegation vs custom override.
8. `required_data` — what it actually does (and what it doesn't).
9. `__repr__` — required for registry hashing.
10. `generate_weights_batch` — optional vectorized fast-path (deferred to `10-simulation.md`).
11. Stateful strategies under multi-path.
12. Common mismatches.
13. Minimal valid cell.
14. Validation checklist.
15. Anti-patterns.

---

## 1. The contract

```python
from abc import ABC
from backtest.strategy import Strategy
from backtest.risk import RiskConfig
import pandas as pd

class MyStrategy(Strategy):
    rebalance_frequency = "monthly"          # REQUIRED — class attr, no default
    risk = RiskConfig(...)                   # optional, defaults to DEFAULT_RISK_CONFIG

    def __init__(self, lookback: int = 60):
        self.lookback = lookback             # all hyperparameters live on self

    def __repr__(self):                      # REQUIRED for stable registry hashing
        return f"MyStrategy(lookback={self.lookback})"

    def required_data(self) -> dict:         # optional, declarative only
        return {"prices": None}

    def fit(self, data) -> None:             # optional, called once per train window
        pass

    def generate_weights(self, data, t) -> pd.Series:   # REQUIRED, abstract
        ...

    def apply_risk(self, proposed, state, data) -> pd.Series:   # optional override
        return super().apply_risk(proposed, state, data)
```

`source: backtest/strategy/base.py:17–54`

| Member | Required? | Default | What it does |
|---|---|---|---|
| `rebalance_frequency` | **Yes** (class attr) | `None` (raises if unset) | When `generate_weights` is called. See §2. |
| `risk` | No (class attr) | `DEFAULT_RISK_CONFIG` | Consumed by the default `apply_risk`. See `03-risk.md`. |
| `__init__` | Practical yes | — | Hyperparameters live on `self` so `__repr__` can serialize them. |
| `__repr__` | **Yes if registry will be used** | object's address-based default | Used by `strategy_hash` for trial dedup. See §9. |
| `required_data` | No | `{"prices": None}` | **Validated by `Engine.__init__`.** Override whenever the strategy reads anything beyond `prices`. See §8. |
| `fit` | No | no-op | Called once if `engine.run(train_dates=...)` is provided. See §6. |
| `generate_weights` | **Yes (abstract)** | — | Per-rebalance weight production. See §5. |
| `apply_risk` | No | delegates to `RiskManager(self.risk)` | Override for fully custom risk logic. See §7. |
| `generate_weights_batch` | No | — | Vectorized fast-path for `run_batch`. See §10. |

If `generate_weights` is not implemented, `Strategy()` raises `TypeError` (ABC enforcement). `source: backtest/strategy/base.py:17, 32–33` and `backtest/strategy/strategy_test.py:37–39`

---

## 2. `rebalance_frequency` — every accepted form

```python
RebalanceSpec = Union[str, Callable[[pd.DatetimeIndex], pd.DatetimeIndex]]
```
`source: backtest/strategy/base.py:14`

`Strategy.rebalance_dates(dates)` resolves the value. Three forms are accepted:

### 2.1 Built-in string aliases

| Value | Behavior |
|---|---|
| `"daily"`, `"D"`, `"B"` | Every bar in `dates`. `source: backtest/strategy/base.py:67–68` |
| `"weekly"` | First bar of each ISO week (`year * 100 + iso_week`). `source: backtest/strategy/base.py:82–84` |
| `"monthly"` | First bar of each calendar month. `source: backtest/strategy/base.py:85–86` |
| `"quarterly"` | First bar of each calendar quarter. `source: backtest/strategy/base.py:87–88` |
| `"yearly"` | First bar of each calendar year. `source: backtest/strategy/base.py:89–90` |

For `"weekly"`/`"monthly"`/`"quarterly"`/`"yearly"`, the **first bar of the first bucket** is included (the implementation does `np.r_[True, keys[1:] != keys[:-1]]`). `source: backtest/strategy/base.py:71`

### 2.2 Pandas frequency alias (anything else string-like)

Anything that is a string but not in the built-ins is treated as a pandas frequency alias. The implementation is:

```python
grid = pd.date_range(dates[0], dates[-1], freq=freq)
pos = dates.searchsorted(grid)
pos = pos[pos < len(dates)]
return dates[np.unique(pos)]
```
`source: backtest/strategy/base.py:73–78`

So `"W-FRI"`, `"BMS"`, `"BME"`, `"Q-MAR"`, etc. all work — the engine builds a regular grid and snaps each grid point to the nearest **on-or-after** bar in your dates index. If `dates` is empty after the snap, `dates[:0]` is returned (no rebalances at all).

If you pass an unrecognized string, `pd.date_range` will raise — at the time `rebalance_dates` is called, not at class definition.

### 2.3 Callable

```python
class S(Strategy):
    rebalance_frequency = staticmethod(lambda dates: dates[::5])
```

A callable receives the full `dates` index and must return an iterable convertible to `DatetimeIndex`. **Wrap it with `staticmethod`** when assigning at class scope or Python will treat it as a bound method. `source: backtest/strategy/base.py:52–53` and `backtest/strategy/strategy_test.py:75–80`

### 2.4 Missing `rebalance_frequency`

`None` (the class default) → `ValueError("...rebalance_frequency is not set")` at the first call to `rebalance_dates`. `source: backtest/strategy/base.py:48–51` and `backtest/strategy/strategy_test.py:28–34`

---

## 3. The execution flow — what the engine does around your strategy

For one `Engine.run()`:

1. **Optional fit.** If `train_dates` is provided and non-empty, the engine calls
   ```python
   strategy.fit(self.data.as_of(train_dates[-1]))
   ```
   exactly once, with a `DataView` snapped to the last train date. `source: backtest/engine/engine.py:145–147`

2. **Resolve rebalance schedule.**
   ```python
   rebalance_dates = self.strategy.rebalance_dates(dates)
   rebalance_mask  = dates.isin(rebalance_dates)
   ```
   `source: backtest/engine/engine.py:149–150`

3. **Per-bar loop.** For each date `t`, the engine first marks-to-market (positions × ratio of price change). Then, **only on rebalance bars**:
   ```python
   state = {
       "drawdown": dd,        # current running drawdown (negative or 0)
       "equity":   equity,    # mark-to-market equity at t
       "positions": pos_series,  # pd.Series[asset → dollar position] at t
       "cash":     cash,      # cash at t
       "peak":     peak,      # running peak equity
   }

   proposed = strategy.generate_weights(view, t)
   proposed = proposed.reindex(assets).fillna(0.0).astype(float)

   eligible = view.assets
   ineligible = ~proposed.index.isin(eligible)
   if ineligible.any():
       proposed.loc[ineligible] = 0.0

   final_w = strategy.apply_risk(proposed, state, view)
   final_w = final_w.reindex(assets).fillna(0.0).astype(float)
   ```
   `source: backtest/engine/engine.py:197–217`

What this means for you:

- The strategy is **only consulted on rebalance bars**. Between them, positions drift with prices.
- The engine **coerces** your output to `assets_all`-aligned floats with NaN→0. You don't need to return a Series of length `assets_all`.
- The engine **zeros out ineligible assets** (per `view.assets`) **before** `apply_risk`. So a strategy that proposes weight on a delisted name still gets correctly muted; risk sees zero on that name, not the proposed weight.
- After `apply_risk`, the same coercion is applied to the final weights.

---

## 4. The `state` dict passed to `apply_risk`

Exact keys, in source order:

| Key | Type | Meaning |
|---|---|---|
| `drawdown` | `float` | Current drawdown (`(equity - peak) / peak`, ≤ 0). |
| `equity`   | `float` | Mark-to-market equity at `t`. |
| `positions`| `pd.Series[asset → float]` | Dollar positions at `t`, indexed by `assets_all`. |
| `cash`     | `float` | Cash balance at `t`. |
| `peak`     | `float` | Running peak equity to date. |

`source: backtest/engine/engine.py:200–206`

If you write a custom `apply_risk`, you read from this dict. The default `RiskManager.apply` consumes `equity` and `drawdown`; see `03-risk.md` for the full list.

`state` is **not** passed to `generate_weights`. If your strategy's weight logic needs to react to drawdown / equity / peak, that logic belongs in `apply_risk`, not `generate_weights`.

---

## 5. `generate_weights(self, data, t) -> pd.Series`

The abstract method. Returns target weights as fractions of equity — so `0.10` means "10% long this name". Long-short is allowed (negative values). `source: backtest/strategy/base.py:32–33`

### 5.1 Return value

- Must be a `pd.Series` indexed by asset id.
- Index can be a **subset** of `data.assets` — engine reindexes and fills missing with 0.
- `NaN` values are coerced to `0.0` before risk.
- Values are floats — anything else gets cast.
- Magnitudes are interpreted as fractions of equity, not dollar amounts.

### 5.2 Empty/insufficient-data branches

If you can't produce a meaningful weight at `t` (insufficient lookback, empty universe), return `pd.Series(0.0, index=data.assets)` — explicit zero exposure. The engine treats that as "fully in cash" for this rebalance.

### 5.3 What `data` is

A `DataView` snapped to `t` (see `01-data.md §3`). The only data you may read. **No lookahead** — `data.prices` is `loc[:t]` *inclusive*. Never call `.shift(-1)`, never compute `.pct_change()` on a series that includes future bars (there is no way to access future bars through `DataView`, but it's still a recurring failure mode if you pass `view.prices` through some helper that confuses orientation).

### 5.4 What `t` is

A `pd.Timestamp` — the snapped-to-prior date in the panel (per `as_of` in `01-data.md §4.2`). Use `t` for time-keying outputs (logging, debugging); use `data` for everything else.

---

## 6. `fit(self, data) -> None`

```python
def fit(self, data: DataView) -> None:
    return None
```
`source: backtest/strategy/base.py:35–36`

- Called **at most once per `Engine.run()`** — and only if `train_dates` is supplied to `run()` and non-empty. `source: backtest/engine/engine.py:145–147`
- Receives a `DataView` snapped to `train_dates[-1]`.
- Free to mutate `self` (store fitted parameters, learned models, scalers).
- Default is a no-op.

If your strategy contains an ML model (e.g. ridge regression on features), the model fitting goes here. The model itself lives on `self`. `generate_weights` then reads `self.model` and produces signals.

In a multi-path / CPCV setting (`07-multipath.md`), `fit` is called **once per path**, on the train slice for that path.

---

## 7. `apply_risk(self, proposed, state, data) -> pd.Series`

Default implementation:
```python
return RiskManager(self.risk).apply(proposed, state, data)
```
`source: backtest/strategy/base.py:38–44`

Two ways to use it:

### 7.1 Default (declarative) — set `risk = RiskConfig(...)`

If your risk needs are covered by `RiskConfig` fields (per-asset cap, gross/net cap, vol target, DD breaker), set the `risk` class attribute and let the default `apply_risk` handle everything. See `03-risk.md` for the full field list.

### 7.2 Override (custom) — implement `apply_risk` yourself

If the spec needs risk logic **not** in `RiskConfig` (sector neutrality, beta neutrality, factor exposure caps, per-strategy stop-loss, custom hedging), override the method:

```python
def apply_risk(self, proposed, state, data):
    # do whatever — read from state, read from data via DataView only
    return adjusted_weights
```
`source: backtest/strategy/strategy_test.py:93–104`

The override receives the engine's `state` dict (§4) and the `DataView`. It must return a Series. The engine will reindex it to `assets_all` and fill NaN with 0.

If you need *both* declarative limits *and* custom logic, call the default explicitly inside your override:
```python
def apply_risk(self, proposed, state, data):
    base = super().apply_risk(proposed, state, data)
    return my_extra_logic(base, state, data)
```

---

## 8. `required_data(self) -> dict`

Default: `{"prices": None}`. `source: backtest/strategy/base.py:29–30`

**The engine validates this at `Engine.__init__`.** `source: backtest/engine/engine.py:128–134`

```python
missing = [name for name in strategy.required_data() if not data.has_field(name)]
if missing:
    raise ValueError(
        f"Strategy {type(strategy).__name__} declared required_data "
        f"fields {missing} that are not registered on the panel. "
        f"Available: {data.available_fields()}."
    )
```

A field is recognized if it is `"prices"` (always present), `"volume"` (present iff the panel was built with a `volume` arg), or any key in the panel's `features` dict. Recognition logic lives in `DataPanel.has_field` / `DataPanel.available_fields`. `source: backtest/data/panel.py` (search for `has_field`).

What this means in practice:

- **Override `required_data` for any strategy that reads beyond `prices`.** Examples:
  ```python
  def required_data(self):
      return {"prices": None, "volume": None}                    # uses ADV
  def required_data(self):
      return {"prices": None, "eps": None, "sector": None}       # uses two features
  ```
- **Failure mode is fast and explicit.** If the user forgets to register a feature, `Engine(panel, strategy)` raises immediately with a list of what's missing and what's available. You don't have to guess at runtime.
- **Values in the dict are not inspected.** The engine only looks at the keys. You can pass `None`, a `FieldSpec`, or anything else as the value — the engine ignores it. (Documenting the field's expected `FieldSpec` in the value is fine for human readers.)
- **The values do not configure the panel.** If a feature needs a non-default `FieldSpec` (e.g. `lag=1`), the user passes that to `DataPanel(specs={...})` at panel construction. `required_data` does not propagate it.

---

## 9. `__repr__` — required for the trial registry

The trial registry hashes a strategy via:
```python
rep = f"{cls.__module__}.{cls.__name__}|{repr(strategy)}"
hash = sha256(rep.encode()).hexdigest()[:16]
```
`source: backtest/registry/hashing.py:11–21`

Default Python `__repr__` includes the instance memory address (`<MyStrategy object at 0x...>`), so two identically-configured strategies hash differently, which breaks trial deduplication.

**Always implement `__repr__`** to depend only on `__init__` args:
```python
def __repr__(self):
    return f"MyStrategy(lookback={self.lookback}, threshold={self.threshold})"
```

This is also why **all hyperparameters belong in `__init__`**, not as class attributes or globals — the registry can only see what `__repr__` exposes.

---

## 10. `generate_weights_batch` — vectorized fast-path

Optional. Used by `run_batch` (Monte Carlo simulation, `10-simulation.md §6`).

```python
def generate_weights_batch(self, view, t):
    prices = view.prices  # ndarray (N_paths, t+1, K_assets)
    ...
    return weights        # ndarray (N_paths, K_assets)
```

Defer details to `10-simulation.md`. **Do not implement this unless the user asks for Monte Carlo / stress-test runs.** It is bit-equivalent to per-path `Engine` only when there are no costs / liquidity / risk overlay (`end_to_end.ipynb` cell `8.6`).

---

## 11. Stateful strategies under multi-path

- Within one `Engine.run()`, `self` state persists across `generate_weights` calls. So you may store rolling buffers on `self`.
- Under `MultiPathEngine` (CPCV), each fold gets its own `deepcopy(strategy)` — state does **not** leak across paths. `source: backtest/engine/multi_path.py:1–17`
- If you write to a **module-level global** instead of `self`, that global will leak across paths. Don't.

---

## 12. Common mismatches — surface these by default

Use the question shape from `PROTOCOL.md §3`. Engine-faithful default first.

| Spec says | Engine has | Question to ask |
|---|---|---|
| Rebalance "every Wednesday" | Built-in `"weekly"` = first ISO-week bar (often Monday). Use `"W-WED"` (pandas alias) instead. | Use `"W-WED"`, **or** confirm the built-in `"weekly"` is acceptable. |
| Rebalance on a custom signal (e.g. "when vol exceeds X") | Callable form (`§2.3`). Build the signal-generated dates yourself. | Show me the signal definition, then I'll wire a callable. |
| "Close positions after N bars" | No engine-level exit logic. Either model it inside `generate_weights` (track entry dates on `self`) or via a custom `apply_risk`. | Implement on `self` inside `generate_weights`, **or** drop the constraint. |
| "Use end-of-day prices to size, then enter at next-day open" | Engine works on a single price field per bar. | Approximate as same-day-close fills (current behavior), **or** lag prices by 1 with `FieldSpec(lag=1)` (changes semantics; see `01-data.md §4`). |
| ML model training during the backtest | `fit()` is called once per `run()` per train window. No retraining on a schedule. | Restructure as walk-forward (`splitters` in `06-splitters.md`) where each fold trains once, **or** retrain inside `generate_weights` at chosen rebalance dates (slow). |
| Stop-loss / trailing stop / take-profit | Not in `RiskConfig`. Implement in custom `apply_risk` reading `state["drawdown"]` / `state["positions"]`. | Custom `apply_risk` with explicit logic, **or** drop the constraint. |
| Sector / factor / beta neutralization | Not in `RiskConfig`. Pass as features and implement in custom `apply_risk`. | Custom `apply_risk`, **or** approximate with per-asset caps. |
| Discrete share counts | Engine uses fractional weights (`prd.md §2`). | Approximate with fractional weights, **or** flag as out-of-scope. |
| Strategy needs to know the previous bar's *trade fills* | `state` does not include the previous trade vector. `state["positions"]` is the dollar position **at `t`** after mark-to-market. | Store on `self` between rebalances, **or** drop. |
| Returns dollar amounts instead of fractions | Engine multiplies by current `equity` to size. | Convert to fractions in the strategy, **or** implement custom `apply_risk` that bypasses the multiplication (not recommended). |

---

## 13. Minimal valid cell

Use this as the **shape** of Stage 2's cell, not a copy-paste. Adapt class name, params, and weight logic to the user's spec. This cell only **defines and instantiates** the strategy — it does not run anything yet (Stage 4 does).

```python
import pandas as pd
from backtest.strategy import Strategy
from backtest.risk import RiskConfig

class UserStrategy(Strategy):
    """<one-line summary from the spec>"""
    rebalance_frequency = "monthly"            # <- per spec
    risk = RiskConfig(max_position=0.20)       # <- per spec; see 03-risk.md

    def __init__(self, lookback: int = 60):
        self.lookback = lookback

    def __repr__(self):
        return f"UserStrategy(lookback={self.lookback})"

    def generate_weights(self, data, t):
        if len(data.prices) < self.lookback + 1:
            return pd.Series(0.0, index=data.assets)
        # <signal computation here, using only data.prices.iloc[:] / data.feature(...)>
        # return a pd.Series indexed by a subset of data.assets
        ...

strat = UserStrategy(lookback=60)
print(strat)                                            # validates __repr__
print(f"Rebalances/year: "
      f"{len(strat.rebalance_dates(panel.dates)) / (len(panel.dates) / 252):.1f}")
```

---

## 14. Validation checklist (after the cell runs)

- [ ] `print(strat)` shows the `__repr__` you defined (not `<UserStrategy object at 0x...>`).
- [ ] Rebalances/year is plausible for the declared frequency:
  - daily ≈ 252, weekly ≈ 52, monthly ≈ 12, quarterly ≈ 4, yearly ≈ 1.
  Wildly off → wrong `rebalance_frequency`.
- [ ] `strat.rebalance_dates(panel.dates)` has length > 0. (Empty = something is wrong with the alias.)
- [ ] If the strategy has hyperparameters in the spec, they all appear in `__repr__`.
- [ ] If the spec references custom risk constructs (sector caps, stop-losses), there is **either** a custom `apply_risk` defined in the class **or** an explicit "drop this requirement" decision recorded.

If any fails: fix in a new cell or modify the class. Do not proceed to Stage 3 (`04-costs.md`) until clean.

---

## 15. What NOT to do

- **Do not access `data._panel`, `data._panel._prices`, etc.** Use `data.prices`, `data.volume`, `data.feature(name)`, `data.assets`, `data.t`. Private attrs are private.
- **Do not call `data._panel.as_of(...)`** to peek at other times. The engine snapped you to `t` for a reason.
- **Do not return weights indexed by anything other than asset ids.** If you return `pd.Series([0.5, -0.5])` with a default `RangeIndex`, the engine's `reindex(assets)` will silently zero everything.
- **Do not put hyperparameters as class attributes** (`lookback = 60` at class scope). They won't appear in `__repr__` and the registry will treat all instances as identical.
- **Do not mutate `self` from `apply_risk`.** State that needs to persist across rebalances belongs in `generate_weights` or `fit`.
- **Do not skip overriding `required_data` for non-trivial strategies.** The engine validates it at construction (§8). A strategy that reads `data.feature("eps")` but returns the default `{"prices": None}` will silently slip past validation and fail later with a `KeyError`. Declaring requirements is how you let the engine catch a misconfigured panel for you.
- **Do not implement `generate_weights_batch`** unless the user explicitly asks for Monte Carlo / batch runs (§10).
- **Do not reach into the engine's `state` dict for keys that aren't in §4.** It has exactly five keys: `drawdown`, `equity`, `positions`, `cash`, `peak`. Anything else doesn't exist.
- **Do not write the strategy and the engine-run cell in the same turn.** Stage 2 defines and instantiates. Stage 4 runs. One cell per turn.

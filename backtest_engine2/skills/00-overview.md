# Skill 00 — Engine Overview

> **Read `PROTOCOL.md` first.** This file assumes you have. The cell-loop and announcement rules in PROTOCOL apply unconditionally; this file is just the map of what you'll be wiring together.

---

## When to load this skill

Every session, exactly once, immediately after `PROTOCOL.md`, before the session opener. You do not re-read it; you read the subsystem skills as you reach their stages.

---

## What you'll find here

1. The canonical pipeline (data flow, top to bottom).
2. The package map — every subsystem's public exports + the skill that covers it.
3. Cross-cutting invariants — things the engine guarantees and things it doesn't.
4. Glossary — terms you'll see in the spec and the source.
5. A short list of things to always remember when writing cells.

There are no implementation details here. Those live in the per-subsystem skills.

---

## 1. The canonical pipeline

```
                        ┌────────────────────────────────────────┐
                        │  user-provided                         │
                        │   • prices  (DataFrame, date × asset)  │
                        │   • volume  (optional)                 │
                        │   • features / universe (optional)     │
                        └────────────────┬───────────────────────┘
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │  DataPanel          │   ← data/
                              │  (PIT enforcement,  │
                              │   missing-data,     │
                              │   universe filter)  │
                              └──────────┬──────────┘
                                         │
                                  .as_of(t) → DataView
                                         │
              ┌──────────────────────────┴──────────────────────────┐
              │                                                     │
              ▼                                                     ▼
   ┌────────────────────┐                              ┌──────────────────────┐
   │  Strategy          │                              │  Engine              │   ← engine/
   │   .generate_weights│  ──── proposed weights ───▶  │   single-path        │
   │   .apply_risk      │  ◀──── DataView, state ──── │   walk-forward       │
   │  RiskConfig        │                              │   mark-to-market     │
   └────────────────────┘                              │   rebalance on dates │
                                                       │   apply CostModel    │
                                                       │   apply LiquidityCap │
                                                       └──────────┬───────────┘
                                                                  │
                                                                  ▼
                                                       ┌──────────────────────┐
                                                       │  BacktestResult      │
                                                       │   equity, returns,   │
                                                       │   weights, trades,   │
                                                       │   costs, unfilled    │
                                                       └──────────┬───────────┘
                                                                  │
                              compute_metrics ◀──────────────────┤
                                         │                        │
                                         ▼                        │
                              ┌─────────────────────┐             │
                              │  MetricsReport      │   ← metrics/│
                              │   21 fields         │             │
                              └─────────────────────┘             │
                                                                  │
                                       (parallel orchestration)   │
                                                                  ▼
                                              ┌──────────────────────────────┐
                                              │  MultiPathEngine + Splitter  │   ← engine/, splitters/
                                              │   WalkForward or CPCV        │
                                              │   purge + embargo            │
                                              │   joblib parallel folds      │
                                              └──────────────┬───────────────┘
                                                             │
                                                             ▼
                                              ┌──────────────────────────────┐
                                              │  MultiPathResult             │
                                              │   N equity paths             │
                                              │   per-path metrics           │
                                              │   aggregate (mean/std/...)   │
                                              └──────────────────────────────┘

Selection (DSR, effective-K) → consumes MetricsReport(s)   ← selection/
Registry (TrialRegistry)     → consumes Engine + causal-graph artifact   ← registry/
Simulation (MonteCarloEngine, batch, validator, ...)  ← simulation/
   ↳ alternate orchestrators that consume a Strategy and produce paths
```

What this diagram says, in one sentence: **the user supplies data and a Strategy; an orchestrator (`Engine`, `MultiPathEngine`, or `MonteCarloEngine`) feeds the strategy point-in-time slices via `DataView`, collects weights, applies costs and risk, and produces a result you measure with `compute_metrics`.**

---

## 2. Package map

Every subsystem is one Python package under `backtest/`. Each row's "Skill" column is where to go for details.

| Package | Public exports | Owns | Skill |
|---|---|---|---|
| `backtest.data` | `DataPanel`, `DataView`, `FieldSpec` | PIT enforcement, missing-data policy, universe filtering, panel construction. `source: backtest/data/__init__.py:1` | `01-data.md` |
| `backtest.strategy` | `Strategy` (ABC) | The contract a user's strategy class must satisfy: `rebalance_frequency`, `generate_weights`, optional `fit` / `apply_risk` / `required_data`. `source: backtest/strategy/__init__.py:1`, `backtest/strategy/base.py:17–44` | `02-strategy.md` |
| `backtest.risk` | `RiskConfig`, `DEFAULT_RISK_CONFIG`, `RiskManager` | Declarative risk limits (per-asset caps, gross/net, vol target, DD breaker) and the default `apply_risk` implementation. `source: backtest/risk/__init__.py:1–3` | `03-risk.md` |
| `backtest.costs` | `CostModel`, `CompositeCostModel`, `Commission`, `Spread`, `MarketImpact`, `ShortBorrow`, `LiquidityCap` | Composable per-trade cost components and the ADV-based liquidity throttle. `source: backtest/costs/__init__.py:1–13` | `04-costs.md` |
| `backtest.engine` | `Engine`, `BacktestResult`, `MultiPathEngine`, `MultiPathResult` | Single-path and parallel-CPCV orchestrators; results dataclasses. `source: backtest/engine/__init__.py:1–4` | `05-engine-single-path.md`, `07-multipath.md` |
| `backtest.splitters` | `Splitter`, `WalkForward`, `CombinatorialPurgedCV` | Train/test fold generation with purge and embargo. `source: backtest/splitters/__init__.py:1–7` | `06-splitters.md` |
| `backtest.metrics` | `sharpe_ratio`, `sortino_ratio`, `calmar_ratio`, `value_at_risk`, `conditional_value_at_risk`, `modified_sharpe`, `psr`, `min_trl`, `sharpe_distribution`, `sharpe_var_term`, `hit_rate`, `turnover`, `information_ratio`, `beta`, `max_drawdown`, `time_under_water`, `MetricsReport`, `compute_metrics`, `calendar_year_returns`, `median_calendar_year_return`, `median_arith_annual_return`, `yearly_metrics` | Scalar performance metrics (paper-faithful + practitioner extras), the `MetricsReport`, path-level drawdown stats, and calendar-year breakdown helpers. `source: backtest/metrics/__init__.py` | `08-metrics.md` |
| `backtest.selection` | `sidak_alpha`, `sidak_pvalue`, `bonferroni_alpha`, `bonferroni_pvalue`, `expected_max_sr`, `dsr`, `effective_k`, `EULER_MASCHERONI` | Multiple-testing corrections and Deflated Sharpe. `source: backtest/selection/__init__.py:1–19` | `09-selection.md` |
| `backtest.simulation` | `PathGenerator`, `panel_from_returns`, 8 generators, 3 adapters, `MonteCarloEngine`, `MonteCarloResult`, `GeneratorValidator`, `ValidationResult`, `Threshold`, `DEFAULT_THRESHOLDS`, `run_until_converged`, `AdaptiveResult`, `Probe`, `ProbeBattery`, `fingerprint`, `FingerprintCache`, `select_generators`, `BatchDataView`, `BatchResult`, `run_batch`, `PathTensorCache` | Monte Carlo layer: synthetic path generation, validation, fingerprinting, adaptive convergence, vectorized batch fast-path, on-disk path cache. `source: backtest/simulation/__init__.py:1–62` | `10-simulation.md` |
| `backtest.registry` | `TrialRegistry`, `strategy_hash`, `dataset_id_for` | Persistent trial registry. Requires a causal-graph artifact at run time. `source: backtest/registry/__init__.py:1–4` | `11-registry.md` |

---

## 3. Cross-cutting invariants

These hold across every subsystem. Treat them as facts; they constrain what you're allowed to write.

### 3.1 DataView is the only data interface for strategies

A `Strategy.generate_weights(data, t)` call receives a `DataView` — never the raw `DataPanel`. The view exposes only data with `event_time <= t` after lag is applied. **There is no API on `DataView` that lets you look ahead.** If the spec describes anything that requires future data, that is a mismatch — surface it, do not work around it. `source: backtest/strategy/base.py:33`, `backtest/data/__init__.py:1`

### 3.2 Strategy state lives on `self`

Strategy parameters (lookbacks, thresholds, learned weights) live on the strategy instance. `fit(data)` is called once per train window with a `DataView` and may mutate `self`. The engine does not provide per-bar mutable state. If the spec wants stateful logic across bars (e.g. trailing stops), it must be implemented on `self` inside `generate_weights`. `source: backtest/strategy/base.py:35–36`

### 3.3 `rebalance_frequency` is required and can be three things

- a string in `{"daily","weekly","monthly","quarterly","yearly"}`
- a pandas frequency alias (e.g. `"W-FRI"`, `"BMS"`)
- a callable `(dates: DatetimeIndex) -> DatetimeIndex`

Missing → `Strategy.rebalance_dates` raises `ValueError`. `source: backtest/strategy/base.py:46–54`

### 3.4 Costs and risk are orthogonal to the strategy

`Engine` accepts `costs: CostModel | None` and `liquidity: LiquidityCap | None` as constructor args, independent of the strategy. The strategy's `risk` attribute is consumed by its own `apply_risk` method, not the engine. This means a single strategy class is reused across cost/liquidity configurations without modification. `source: backtest/engine/engine.py:118–132`

### 3.5 `BacktestResult` is the canonical result shape

```python
BacktestResult(
    equity_curve, returns, weights, positions, trades,
    costs, unfilled, metadata,
)
```
plus `.pnl`, `.gross_pnl`, `.summary(...)` derived from those fields. Anything you want to measure must be derivable from this shape — if it isn't, that is a mismatch. `source: backtest/engine/engine.py:15–35`

### 3.6 `MultiPathEngine` clones the strategy per fold

CPCV produces multiple paths by running independent backtests on each fold combination. The orchestrator deep-copies the strategy for each path; you cannot share mutable state across paths via the strategy instance. `source: backtest/engine/multi_path.py:1–17`

### 3.7 Survivorship is the panel's job, not yours

`DataPanel` handles delisted-asset terminal knowledge times and universe filtering. You never write code in the strategy that tries to "drop NaN columns" or "keep only assets present at start". If the spec implies survivorship cleanup, surface it. `source: backtest/data/__init__.py:1`, see `01-data.md` for the policy.

### 3.8 No-cost runs are technically allowed but procedurally forbidden

`Engine(..., costs=None, liquidity=None)` will run. `PROTOCOL.md §6` forbids it. The only acceptable no-cost cell is an explicit opt-in comparison cell the user asked for.

---

## 4. Glossary

Terms you'll meet in the spec, in source comments, or in cell announcements.

| Term | Meaning |
|---|---|
| **PIT** | Point-in-time. Data exposed at time `t` reflects only what was knowable at `t` after applied lag. |
| **Lag mode** | The v1 PIT enforcement strategy: per-field `lag_bars` declared in `FieldSpec`; data is shifted before exposure. Knowledge-time mode is deferred. (See `prd.md §4.2`.) |
| **DataView** | The point-in-time slice of a panel returned by `DataPanel.as_of(t)`. The only data interface a strategy sees. |
| **Rebalance date** | A date on which the strategy is asked to produce new target weights. Between rebalances the engine marks the existing portfolio to market. |
| **ADV** | Average Daily Volume. Used by `MarketImpact` and `LiquidityCap`, computed over `adv_lookback` bars. |
| **CPCV** | Combinatorial Purged Cross-Validation. Generates many train/test fold combinations from one dataset; with purging and embargo, paths are reusable for variance reduction. |
| **Purge** | Removes train labels whose information would overlap with the test window. |
| **Embargo** | Drops a buffer of train bars immediately after each test window to prevent serial-correlation leakage. |
| **Sharpe** | `mean(returns) * ann_factor / (std(returns) * sqrt(ann_factor))`. Sample formula is in `08-metrics.md`. |
| **PSR** | Probabilistic Sharpe Ratio. P(true SR > `sr_star`) given the observed sample. |
| **MinTRL** | Minimum Track Record Length. How long a sample must be for PSR to exceed a confidence level. |
| **DSR** | Deflated Sharpe Ratio. Bayes-adjusted PSR that deflates by the number of trials `K`. |
| **Effective-K** | `K` discounted for trial correlation via hierarchical clustering on the trial return matrix. |
| **Sidak / Bonferroni** | Family-wise error corrections on `alpha` or `p-value` given `K` trials. |
| **Probe** | One of 6 deterministic synthetic datasets isolating a single market property (momentum, mean-reversion, vol clustering, jump diffusion, cross-section, IID baseline). |
| **Fingerprint** | A strategy's delta-Sharpe vector across the probe battery; tells you what kind of edge it has. |
| **Generator** | An object producing synthetic return paths `(n_paths, n_steps, n_assets)`. 8 are exported; choose by the strategy's fingerprint. |
| **Adapter** | A wrapper around a generator that modifies tails / introduces jumps / antithetic-pairs the output. |
| **Path tensor** | The 3-D ndarray `(n_paths, n_steps, n_assets)` of synthetic returns. Cached on disk by `PathTensorCache`. |

---

## 5. Things to remember on every cell

Repeated from `PROTOCOL.md` because they apply in every cell, not just the ones whose subsystem you happen to be reading right now:

1. **Cite or don't claim.** Every API signature, formula, default, or behavior you assert about the engine must end with a `source: backtest/<path>:<line>` citation. If you can't cite it, you don't know it.
2. **DataView only.** No raw panel access in strategies, no `.shift(-1)`, no `pct_change()` on the full series.
3. **Costs are mandatory** before any `Engine.run()` (see `PROTOCOL.md §6`).
4. **DSR is mandatory** even for a single trial — the user will iterate.
5. **One cell per turn.** If you're about to write a second cell, stop.
6. **Surface mismatches; never paper over.** If the engine doesn't do what the spec asks, that is a decision for the user, not for you.

---

## 6. Where to go next

After this file, the next thing you read is the matching skill for the cell you're about to announce. Stage map is in `PROTOCOL.md §8` and `README.md`.

The next skill to consult is almost always `01-data.md`, since Cell 1 is always data load.

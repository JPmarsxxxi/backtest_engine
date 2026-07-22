# Backtesting Engine — PRD

Reference: Joubert, Sestovic, Barziy, Distaso, Lopez de Prado (2024), "The Three Types of Backtests."

## 1. Goal

A strategy-agnostic Python backtesting engine that integrates the methodological recommendations from the reference paper. User brings strategy + data; engine handles PIT enforcement, costs, risk overlay, walk-forward and combinatorial purged cross-validation paths, performance metrics, and selection-bias-aware deflation.

Monte Carlo / DGP-based simulation is out of scope for v1 but the architecture leaves seams open.

## 2. Scope

In scope (v1):
- Walk-forward backtests
- Combinatorial Purged Cross-Validation (CPCV) with purging and embargo
- Multi-asset, long-short, fractional weights
- Configurable rebalance frequency (declared per strategy)
- Costs: commission, spread, market impact, short-borrow
- Liquidity caps as % of ADV
- Risk overlay (declarative config + override hook)
- Metrics: SR, PSR, MinTRL, max drawdown, VaR, cVaR, Calmar, Sortino, modified Sharpe, time-under-water
- SBuMT corrections: Sidak, Bonferroni, DSR, effective-K via hierarchical clustering
- Trial registry with causal-graph artifact requirement

Out of scope (v1, deferred to v2):
- Monte Carlo path generation (block bootstrap, GAN/GARCH DGP)
- Data fetching / ingestion (BYOD)
- Discrete-share fills
- Intraday data (architecture compatible, not exercised)
- Sector/region concentration limits (placeholder only)

## 3. User responsibilities

User provides:
1. A `Strategy` subclass with trading rules.
2. Data as pandas DataFrames (prices required; features, volume, universe optional).
3. Train/test date split (optional but recommended).
4. Causal graph image path before any registered run.

Engine does not fetch, clean, or vintage data.

## 4. Data contract (`data/`)

### 4.1 Inputs
- `prices: DataFrame[date x asset]` — adjusted prices (required)
- `volume: DataFrame[date x asset]` — required if liquidity caps active
- `features: dict[str, DataFrame]` — optional, additional fields
- `universe: DataFrame[date x asset, bool]` — time-varying eligibility (optional; defaults to non-NaN price)

### 4.2 PIT enforcement
- **Lag mode (v1):** data indexed by `event_time`; user declares per-field lag in bars. Engine shifts before exposing.
- Knowledge-time mode (vintaged data with restatements) is deferred. See §19.

Engine exposes only `DataView.as_of(t)` to strategies. No raw access.

### 4.3 Missing data policy (per field)
- `drop` (default), `ffill(max_staleness)`, or `value` (constant).

### 4.4 Universe filtering
- Eligibility evaluated as-of t. Delisted assets retained with their delist date as terminal `knowledge_time`. No survivorship leakage.

Maps to paper: Survivorship bias, Point-in-time, Look-ahead bias, Universe selection.

## 5. Strategy contract (`strategy/`)

```python
class Strategy:
    rebalance_frequency: str | Callable      # required: "daily","weekly","monthly", cron, or callable
    risk: RiskConfig = DEFAULT_RISK_CONFIG    # optional override

    def required_data(self) -> dict: ...
    def generate_weights(self, data: DataView, t) -> pd.Series: ...
    def apply_risk(self, proposed, state, data) -> pd.Series:
        # default delegates to RiskManager(self.risk); override for custom logic
        ...
```

Rules:
- `rebalance_frequency` must be set; engine raises if missing.
- Strategy never accesses data outside `DataView.as_of(t)`.
- Returns target weights summing to anything within risk bounds (-1 to +1 per asset by default).

## 6. Risk module (`risk/`)

Declarative `RiskConfig` consumed by a default `RiskManager`. Strategy may override `apply_risk` for fully custom logic.

### Default config (when strategy provides none)
```
max_position = 0.20
max_gross    = 1.0
max_net      = 1.0
target_vol   = None
```

### Available constraints
- Per-asset position cap
- Gross exposure cap (sum |w|)
- Net exposure cap (sum w)
- Leverage hard ceiling
- Volatility targeting (scale to hit target portfolio vol from rolling cov)
- Concentration limits by group (placeholder, requires asset metadata)

## 7. Costs module (`costs/`)

`CostModel` base; composable. Built-ins:
- Commission: per-share or bps of notional
- Spread: half-spread in bps (configurable per asset)
- Market impact: sqrt or linear (Kyle-style), parameterised
- Short-borrow: bps annualised, applied per holding day on negative positions

Liquidity cap: order capped at `cap_pct * ADV` (rolling N-day median volume).

Maps to paper: Transaction costs (Borkovec & Heidle), Short-sale constraints, Liquidity constraints.

## 8. Splitters (`splitters/`)

- `WalkForward(train_end | train_pct, embargo_bars=0)` — single path. Yields one `(train, test)` tuple.
- `CombinatorialPurgedCV(n_splits, n_test_groups, purge_bars=0, embargo_pct=0.01)` — multi-path. Yields `C(n_splits, n_test_groups)` train/test splits, reassembled into `C(n_splits-1, n_test_groups-1)` full-timeline paths.
- Purging: drops train bars within `purge_bars` on either side of any test run.
- Embargo: drops a further `int(n * embargo_pct)` train bars after any test run. Purge and embargo stack on the trailing side; only purge applies on the leading side.

`WalkForward` and `CombinatorialPurgedCV` are separate concrete classes behind a shared `Splitter` ABC. CPCV requires `n_splits >= 2`; the single-split case is `WalkForward`.

Maps to paper: Walk-forward, Resampling, Purging, Embargo (Lopez de Prado 2018).

## 9. Engine (`engine/`)

Per `(train_dates, test_dates)`:
1. Strategy fit/configured on train (if `fit` method present).
2. For each rebalance date `t` in test:
   - `view = data.as_of(t)`
   - `proposed = strategy.generate_weights(view, t)`
   - `final = strategy.apply_risk(proposed, state, view)`
   - `trades = compute_trades(current, final)`
   - `trades = liquidity_cap.apply(trades)`
   - `costs = cost_model.apply(trades, view)`
   - update positions, accrue PnL
3. On non-rebalance dates: weights drift with prices; no trading, no costs.
4. Output: equity curve, trade log, metrics.

Multi-path (CPCV): paths run in parallel via joblib.

## 10. Metrics (`metrics/`)

| Metric | Source |
|---|---|
| Sharpe (annualised) with skew/kurt-aware variance | Eq. (1)–(2) |
| PSR | Eq. (3) |
| MinTRL | Eq. (4) |
| Max drawdown | Eq. (5) |
| VaR | Eq. (6) |
| cVaR | Eq. (7) |
| Calmar, Sortino | Standard |
| Modified Sharpe (VaR-denom) | Favre & Galeano 2002 |
| Time-under-water | Holistic eval |

Returns a `MetricsReport` object with named fields plus `.to_dict()`.

## 11. Selection bias corrections (`selection/`)

- Sidak: `alpha_k = 1 - (1-alpha)^(1/K)`
- Bonferroni: `alpha_k = alpha/K`
- DSR (Eq. 21) using False Strategy Theorem expectation (Eq. 20)
- Effective-K estimator: hierarchical clustering on trial-return correlation matrix (Lopez de Prado & Lewis 2019)

Reads K from the trial registry — not researcher-reported.

## 12. Trial registry (`registry/`)

Per-trial record:
- Strategy class + config hash
- Data window
- Causal-graph image path (required for non-exploratory runs)
- Returns series
- Metrics
- Timestamp

Persistence: parquet files per trial + SQLite index for fast queries.

`engine.run()` requires a causal-graph image path; `skip_causal_graph=True` flags the run as exploratory and excludes it from DSR K computations.

Maps to paper: Causal graphs (Lopez de Prado 2023), SBuMT.

## 13. Simulation stub (`simulation/`)

Empty interface in v1. Two future entry points:
- Block / stationary bootstrap (low complexity)
- DGP-based (GARCH, regime-switching, neural — high complexity)

Engine accepts arbitrary data, so MC integration is feeding synthetic frames in. No engine changes anticipated.

## 14. Performance and optimization plan

| Layer | Approach | Package |
|---|---|---|
| Data view / PIT lookup | Sorted-index `loc` with cached masks | pandas, numpy |
| Rolling stats | NaN-aware reductions | bottleneck |
| Engine inner loop | JIT-compiled numerical kernel (numpy in/out, no Python objects) | numba `@njit` |
| Max drawdown / time-under-water | JIT loop | numba |
| CPCV multi-path | Embarrassingly parallel | joblib `Parallel(n_jobs=-1)` |
| Distribution math | Standard CDF/PPF | scipy.stats |
| Registry I/O | Columnar parquet | pyarrow |
| User-facing API | DataFrames | pandas |

Architecture rule: numba kernels stay pure-numpy; pandas wraps at the user boundary. No premature numexpr or polars; profile before adding.

## 15. Build rules

1. Lean, straight-to-the-point code. No emojis, no decorative comments, no docstring novels. One-line WHY-comment only when non-obvious.
2. Vectorize where vectorizable, JIT where loops are unavoidable, parallelize independent work.
3. Code in chunks. After each chunk, halt for user testing before proceeding.
4. No backwards-compat shims, no half-finished features. Each chunk merges fully or not at all.
5. Type hints on public interfaces.
6. Tests live alongside modules (`module_test.py`).

## 16. Build chunks

Each chunk is independently testable. Order is dependency-driven.

| # | Chunk | Modules | Deliverable |
|---|---|---|---|
| 1 | Skeleton + data layer | `data/` | Project layout, `DataView.as_of(t)`, lag mode, missing-data policy, universe filter |
| 2 | Strategy + risk | `strategy/`, `risk/` | Base class, RiskConfig, default RiskManager, override hook |
| 3 | Costs + liquidity | `costs/` | Commission, spread, impact, borrow, ADV cap |
| 4 | Single-path engine | `engine/` (walk-forward) | Run a strategy end-to-end, return equity curve + trade log |
| 5 | Metrics | `metrics/` | All metrics in section 10, MetricsReport |
| 6 | Splitters | `splitters/` | Walk-forward + CPCV with purge/embargo |
| 7 | Multi-path engine | `engine/` (CPCV) | Parallel CPCV via joblib, aggregate per-path metrics |
| 8 | Selection corrections | `selection/` | Sidak, Bonferroni, DSR, effective-K |
| 9 | Trial registry | `registry/` | Persistence, causal-graph requirement, K source |
| 10 | Integration | example strategy + end-to-end run | Smoke test wiring everything |

## 17. Defaults locked

- Asset scope: multi-asset (single-asset is a degenerate case)
- Direction: long-short
- Position units: fractional weights
- Rebalance: per-strategy declaration
- Risk config: permissive default (max_position 20%, max_gross 100%, max_net 100%)
- Data: BYOD pandas DataFrames, lag-mode PIT
- Storage: parquet + SQLite for registry

## 18. Flags / deferred (tracked, not in v1)

Items intentionally deferred. Each entry has a trigger that brings it back into scope.

- **Knowledge-time PIT mode** — long-form `(event_time, knowledge_time, asset, field, value)` input handling restatements (same `(asset, event_time)` with multiple `knowledge_time` versions). Lag mode cannot represent this. *Trigger:* user feeds vintaged fundamentals (earnings, balance-sheet, revised macro, retroactively-changed index constituents).
- **Monte Carlo / DGP simulation** — `simulation/` stub only. *Trigger:* user has read-up on the approach and wants synthetic-path generation; engine already accepts arbitrary data, so integration is feeding synthetic frames in.
- **Discrete-share fills** — currently fractional weights. *Trigger:* small-account or per-share commission strategies where the rounding drag is non-negligible.
- **Sector/region concentration limits** — `RiskConfig` placeholder only; needs asset metadata. *Trigger:* user provides a group-membership table.
- **Polars in `data/`** — sticking with pandas. *Trigger:* profiling shows `data/` is a bottleneck on multi-asset, long-history runs.

## 19. Out of scope explicitly

- Data fetching, vendor adapters
- Live trading, order routing
- Discrete-share execution
- Intraday-specific features (microstructure costs, etc.)
- Monte Carlo / DGP simulation
- Strategy optimisation / parameter search (registry tracks trials but does not search)

# Skill 07 — Multi-Path Engine

> **Read `PROTOCOL.md` and `skills/00-overview.md` first.** Cell-loop and announcement rules in PROTOCOL apply.

> Stage 6 splitter must already be constructed. This skill covers the `MultiPathEngine` that consumes it. Splits-vs-paths and CPCV mechanics live in `skills/06-splitters.md`.

---

## When to load

- **Stage 6, run cell**: wiring a constructed `Splitter` into `MultiPathEngine` and calling `.run()`.
- Any cell that imports `MultiPathEngine` or `MultiPathResult`.
- Any cell that reads `MultiPathResult` attributes (`split_returns`, `path_returns`, `path_equity`, etc.) or methods (`metrics_per_path`, `aggregate_metrics`, `summary`).

---

## What you'll find here

1. The `MultiPathEngine` constructor — args, validation, what it does **not** validate.
2. Strategy vs zero-arg factory — when each is right.
3. `run()` — parallelism, what happens per split.
4. The per-split execution path: fit-once, contiguous-segment splitting, fresh engine per segment.
5. `MultiPathResult` — fields, what's retained, what's discarded.
6. `metrics_per_path` / `aggregate_metrics` — the nine aggregated keys and how stats are computed.
7. `summary()` — what prints, what plots.
8. Common mismatches.
9. Minimal valid cell.
10. Validation checklist.
11. Anti-patterns.

---

## 1. `MultiPathEngine` — the constructor

```python
MultiPathEngine(
    data: DataPanel,
    strategy: Strategy | Callable[[], Strategy],
    splitter: Splitter,
    costs: CostModel | None = None,
    liquidity: LiquidityCap | None = None,
    initial_capital: float = 1_000_000.0,
    cash_rate: float = 0.0,
    borrow_rate: float = 0.0,
)
```
`source: backtest/engine/multi_path.py:135–157`

| Arg | Required? | Notes |
|---|---|---|
| `data` | Yes | The `DataPanel` from Stage 1. Shared across all splits (joblib pickles it once per worker). |
| `strategy` | Yes | A `Strategy` **instance** (deep-copied per split) or a zero-arg callable returning a fresh `Strategy` (called per split). See §2. |
| `splitter` | Yes | A `Splitter` subclass: `WalkForward` or `CombinatorialPurgedCV` from skill 06. |
| `costs` | Per PROTOCOL: **yes** | Same `CompositeCostModel` from Stage 3. `PROTOCOL.md §6` makes it mandatory. |
| `liquidity` | Optional | A `LiquidityCap` if the panel has volume. |
| `initial_capital` | No | Starting equity per split. Must be `> 0`. Each split starts fresh from this; equity does not carry across paths. |
| `cash_rate` | No | Annualized interest on idle positive cash. Default `0.0`. Forwarded to every inner `Engine`. `source: backtest/engine/multi_path.py:165, 226` |
| `borrow_rate` | No | Annualized rate paid on negative cash (overdraft). Default `0.0`. Forwarded to every inner `Engine`. `source: backtest/engine/multi_path.py:166, 227` |

### 1.1 Construction-time validation

Only **one** check fires at construction. `source: backtest/engine/multi_path.py:144–145`

1. **`initial_capital <= 0`** — `"initial_capital must be positive"`.

That's it. The other validations from single-path `Engine` (`required_data` fields, `liquidity-needs-volume`) **do not fire at `MultiPathEngine.__init__`**. They fire when `_run_one_split` constructs a fresh `Engine` per fold, which happens inside `.run()`. So a misconfigured panel surfaces as a `ValueError` from inside a joblib worker — readable but later than you'd expect. Catch it earlier by running a single `Engine(...)` in Stage 4 first.

---

## 2. Strategy vs factory — when to use which

The constructor argument is typed `StrategyOrFactory = Union[Strategy, Callable[[], Strategy]]`. `source: backtest/engine/multi_path.py:18`

The discrimination happens at `__init__`:
```python
self._is_factory = callable(strategy) and not isinstance(strategy, Strategy)
```
`source: backtest/engine/multi_path.py:149`

A `Strategy` subclass is itself `callable` (you can call it to instantiate), so the check needs both clauses. The second clause says: if the user passed an instance (not a class, not a lambda), treat it as an instance to deep-copy. If they passed a zero-arg callable that is **not** itself a `Strategy` instance, treat it as a factory.

| Pattern | Behavior per split |
|---|---|
| `MyStrategy(lookback=60)` (instance) | `copy.deepcopy(prototype)` runs in `_run_one_split:207`. State on `self` at `__init__` time is preserved through the copy. |
| `lambda: MyStrategy(lookback=60)` (factory) | Factory is called fresh per split: `strategy = strategy_or_factory()`. Each split gets a brand-new instance with `__init__`-clean state. |
| `_make_factory` (named function returning a Strategy) | Same as lambda — called per split. |

**When to prefer the factory pattern:**

- Strategy holds state that doesn't survive deep-copy cleanly (e.g. unpicklable objects, file handles).
- Strategy uses RNG and you want each split to see a different sequence (factory + per-call seed).
- Strategy lazily initializes heavy state in `__init__` and you want to control whether the cost is paid once-per-prototype-instance or once-per-split.
- `fit(...)` mutates `self` in a way you don't want carrying over between paths in CPCV (deep-copy handles this fine, but factory makes it explicit).

**When to prefer the instance pattern:**

- Strategy is parameterised with cheap-to-copy state (numbers, arrays).
- You want a single trial-registry entry with stable `__repr__` (the prototype's `__repr__` is what gets hashed).
- You want every split to start identically — deep-copy from a single prototype guarantees this; a factory using time-based RNG does not.

### 2.1 State isolation guarantee

Whether instance or factory, **the prototype is never mutated**. Tested at `multi_path_test.py:108–113`: `_StatefulCounter` with `fit_calls=0` retains `0` after a multi-path run, because each split mutated its own copy. `source: backtest/engine/multi_path.py:204–207`

---

## 3. `run()` — what it does

```python
run(
    dates: pd.DatetimeIndex | None = None,
    n_jobs: int = -1,
) -> MultiPathResult
```
`source: backtest/engine/multi_path.py:151–191`

### 3.1 Arguments

| Arg | Default | Meaning |
|---|---|---|
| `dates` | `self.data.dates` | The dates to split. Pass an explicit `pd.DatetimeIndex` to restrict to a subwindow. |
| `n_jobs` | `-1` | joblib's process pool size. `-1` = all cores. `1` = sequential (no joblib worker). |

`n_jobs=1` is the right choice for debugging — exceptions propagate immediately without joblib serialization. For production sweeps, `-1` is fastest. On Windows, joblib uses processes (not threads), so the panel and strategy pickle once per worker; large panels (>100k bars × 1k assets) may show non-trivial startup cost.

### 3.2 The orchestration

1. `dates = self.data.dates` if not supplied; otherwise sorted `DatetimeIndex`. `source: backtest/engine/multi_path.py:164`
2. `all_splits = list(self.splitter.split(dates))` — materialize the iterator once. `source: backtest/engine/multi_path.py:165`
3. Build per-split argument tuples including `cash_rate` and `borrow_rate` so workers can construct identically-configured inner engines. `source: backtest/engine/multi_path.py:167–181`
4. **Parallel dispatch**:
   - `n_jobs == 1`: sequential list comprehension. Exceptions propagate directly.
   - else: `joblib.Parallel(n_jobs=n_jobs)(delayed(_run_one_split)(*a) ...)` inside a `try/except`. **On any worker exception, MultiPathEngine emits a `UserWarning` and reruns sequentially** so the user sees the underlying traceback unwrapped, not buried in a joblib `_RemoteTraceback`. The fallback doubles the failed-run time but only fires when the run was failing anyway. `source: backtest/engine/multi_path.py:183–199`
5. Unpack per-split outputs into `split_returns` (one `pd.Series` per split) and `split_results` (one `list[BacktestResult]` per split — one entry per contiguous segment). `source: backtest/engine/multi_path.py:201–202`
6. `path_returns = splitter.assemble_paths(dates, split_returns)` — splitter stitches per-split returns into per-path series (skill 06 §3.6). `source: backtest/engine/multi_path.py:204`
7. `path_equity = (1.0 + path_returns.fillna(0.0)).cumprod() * self.initial_capital`. `source: backtest/engine/multi_path.py:205`
8. Wrap in `MultiPathResult`. `source: backtest/engine/multi_path.py:207–216`

**Note on `fillna(0.0)`:** per-segment leading bars are `NaN` (engine convention — see `skills/05-engine-single-path.md §3` / `08-metrics.md`). Compounding fills them with 0, so equity at those bars stays at the prior bar's value (or `initial_capital` for the very first bar of a path). The returns themselves retain `NaN`, which is what `metrics_per_path` consumes via `dropna()`.

---

## 4. The per-split execution

`source: backtest/engine/multi_path.py:194–228`

Per split (run by joblib worker, top-level function for picklability):

1. **Materialize a fresh strategy**:
   - Factory: `strategy = strategy_or_factory()`. `source: backtest/engine/multi_path.py:204–205`
   - Instance: `strategy = copy.deepcopy(strategy_or_factory)`. `source: backtest/engine/multi_path.py:206–207`
2. **Empty test → return empty Series**. No engine instantiated. `source: backtest/engine/multi_path.py:209–210`
3. **Fit on the train window** if non-empty: `strategy.fit(data.as_of(train[-1]))`. The fit sees a `DataView` at the last train date, not the panel itself — consistent with single-path Engine convention. `source: backtest/engine/multi_path.py:212–213`
4. **Split the test into contiguous segments**: `_contiguous_segments(test, data.dates)`. For walk-forward (one contiguous test region), this returns a single segment. For CPCV with `n_test_groups > 1`, the test may be non-contiguous and split into 2..k segments. `source: backtest/engine/multi_path.py:215, 231–247`
5. **Instantiate one `Engine`** with costs / liquidity / capital, then call `engine.run(test_dates=seg)` per segment and concat per-segment returns. `source: backtest/engine/multi_path.py:216–228`

### 4.1 Why per-segment runs

For CPCV with non-contiguous test sets, running the engine over the entire test set including the gap would have positions drift through the gap (no rebalancing in the gap, just mark-to-market). That drift is not a real strategy P&L — it's an artifact of the engine seeing dates the splitter excluded. Per-segment runs reset positions and cash to fresh state for each contiguous test region, so each segment's return series reflects only what the strategy did on that segment. `source: backtest/engine/multi_path.py:128–129` (docstring) and `backtest/engine/multi_path.py:215–228` (impl).

**Consequence:** between two contiguous segments of the same split, positions are zeroed and cash reset to `initial_capital`. A strategy that "holds" through one segment will see the next segment as a fresh start. This is correct for CPCV's semantics (the segments are non-adjacent in real time) but worth flagging to the user if they have a held-position interpretation.

### 4.2 `_contiguous_segments`

`source: backtest/engine/multi_path.py:231–247`

Maps each test date to its index in the full panel via `all_dates.get_indexer(test)`. If any date is missing from the panel, raises `ValueError("test contains dates not in data panel")` — guards against a splitter producing dates outside the panel. Otherwise finds index breaks (`np.diff(pos) > 1`) and slices the test index at those breaks.

For walk-forward: `n_test_groups=1, n_splits=1` → one contiguous segment, `_contiguous_segments` returns a 1-element list. For CPCV with `n_test_groups=2, n_splits=6`: up to two segments per split.

---

## 5. `MultiPathResult` — the fields

```python
@dataclass
class MultiPathResult:
    split_returns: list[pd.Series]
    path_returns:  pd.DataFrame
    path_equity:   pd.DataFrame
    splitter:      Splitter
    initial_capital: float
    n_splits:      int
    n_paths:       int
    split_results: list[list[BacktestResult]] = field(default_factory=list)
```
`source: backtest/engine/multi_path.py:21–32`

| Field | Type | Shape | Meaning |
|---|---|---|---|
| `split_returns` | `list[pd.Series]` | length = `n_splits` | Per-split returns Series (after segment concat). Indexed by that split's test dates. Each leads with `NaN` per contiguous segment. |
| `path_returns` | `pd.DataFrame` | `(n_dates, n_paths)` for CPCV; `(n_test, n_paths)` for WF | Stitched per-path returns from `splitter.assemble_paths`. Column names `path_0, path_1, …`. Has `NaN` at every segment start (skill 06 §3.6). |
| `path_equity` | `pd.DataFrame` | same as `path_returns` | Compounded from `path_returns` with `NaN`s filled to 0. Each path starts at `initial_capital` at its first bar. |
| `splitter` | `Splitter` | — | Reference to the splitter used. `MultiPathResult.summary()` and downstream consumers may inspect `n_paths`, `n_groups`, etc. |
| `initial_capital` | `float` | — | The capital each path was simulated against. |
| `n_splits` | `int` | — | `splitter.n_splits()` snapshot. |
| `n_paths` | `int` | — | `splitter.n_paths()` snapshot. |
| `split_results` | `list[list[BacktestResult]]` | outer length = `n_splits`; inner length = number of contiguous segments in that split (1 for WF; 1 or more for CPCV) | Full per-segment `BacktestResult` objects — `equity_curve`, `returns`, `weights`, `positions`, `trades`, `costs`, `unfilled`, `metadata`. Use this for per-path turnover, per-path cost analysis, position diagnostics, etc. |

### 5.1 Memory cost of `split_results`

Each `BacktestResult` holds five DataFrames at `(n_bars × n_assets)` plus three Series at `(n_bars,)` plus a small metadata dict. For a single split with 252 bars and 5 assets that's ~7 kB. For 15 splits of a 1260-bar / 500-asset panel: ~75 MB total. Acceptable on a workstation; worth pruning before pickling if the result is being serialized.

Quick prune pattern: `res.split_results = []` after extracting whatever you need.

---

## 6. `metrics_per_path()` and `aggregate_metrics()`

### 6.1 `metrics_per_path()`

```python
def metrics_per_path(self) -> list[MetricsReport]:
    ...
```
`source: backtest/engine/multi_path.py:31–39`

For each `path_returns` column, drops NaNs and calls `compute_metrics(ret, eq)` (skill 08). Returns a list of `MetricsReport` of length `n_paths`. Each report has 21 fields (skill 08 §X).

### 6.2 `aggregate_metrics()`

```python
def aggregate_metrics(self) -> dict:
    ...
```
`source: backtest/engine/multi_path.py:41–61`

Aggregates **fifteen** metrics across paths into a dict with `{metric}_{stat}` keys.

**Aggregated metrics** (the keys, exactly): `sharpe, sortino, calmar, modified_sharpe, psr, max_drawdown, time_under_water, var_5, cvar_5, ann_return, ann_vol, total_return, total_pnl, total_costs, longest_underwater`. `source: backtest/engine/multi_path.py:46–51`

**Stats per metric**: `mean, std, min, max`. So 15 × 4 = **60 keys** in the returned dict.

**Per-key arithmetic** (`source: backtest/engine/multi_path.py:53–67`):

- Filter to finite values: `arr = arr[np.isfinite(arr)]`.
- If `len(arr) == 0`: all four stats are `NaN`.
- Else: `mean`, `min`, `max` are the obvious numpy reductions. `std` uses `ddof=1` (sample standard deviation) when `len > 1`, else `0.0`.

Metrics **not** aggregated (present in `MetricsReport` but missing from the aggregate dict): `gross_pnl`, `sharpe_std`, `sharpe_ci_low`, `sharpe_ci_high`, `min_trl`, `n_obs`, `ann_factor`. These are either per-sample CIs (don't aggregate cleanly across paths), constants (`ann_factor`), or trial-counting (`n_obs`). Compute them from `metrics_per_path()` directly if needed.

---

## 7. `summary()` — what it does

```python
result.summary(ann_factor=252, figsize=(13, 4.5)) -> dict
```
`source: backtest/engine/multi_path.py:63–119`

Four things happen:

1. **Computes `aggregate_metrics()`** and `metrics_per_path()`.
2. **Displays a compact table** (HTML in Jupyter via `IPython.display.display`, else plain print): rows `sharpe, sortino, calmar, max_drawdown, psr`, columns `mean, std, min, max` — a 5×4 view, not the full 9-metric aggregate. `source: backtest/engine/multi_path.py:85–97`
3. **Plots a 1×2 dashboard**:
   - Left: per-path equity overlay (all paths drawn semi-transparent in `C0`), with `initial_capital` reference line.
   - Right: histogram of per-path annualized Sharpe with a mean line.
   `source: backtest/engine/multi_path.py:99–117`
4. **Returns the aggregate dict** (36 keys, all 9 metrics × 4 stats).

Requires `matplotlib`. Raises `ImportError` with install instructions if missing.

The summary table is a **subset** of the aggregate dict; if the user wants to see annualized vol or modified Sharpe in the table, they need to reach into the returned dict directly.

---

## 8. Common mismatches — surface these by default

Use the question shape from `PROTOCOL.md §3`. Engine-faithful default first.

| Spec says | Engine has | Question to ask |
|---|---|---|
| "Run with cash interest at 4% / borrow at SOFR + 50bps" | `MultiPathEngine(..., cash_rate=0.04, borrow_rate=0.055)` — forwarded to every inner `Engine`. | Confirm the rate basis (annualized, daily-compounded over 252 bars per `engine.py:218`). |
| "Show per-path turnover / per-path total costs" | `res.split_results[split_idx][segment_idx]` returns the full `BacktestResult` per segment, including `trades`, `costs`, `weights`. Derive turnover as e.g. `bt.trades.abs().sum().sum() / bt.equity_curve.mean()`. | Confirm aggregation: do they want one number per *split* (sum across segments) or per *path* (sum across the splits that contributed to that path)? Recommend per-split first. |
| "Each path should refit the strategy mid-window" | `strategy.fit(...)` is called once per split at `train[-1]`. No per-bar or rolling refits within a split. | Out of scope for `MultiPathEngine`. Suggest a custom strategy whose `generate_weights` retrains internally on the `DataView`, but flag that this is a strategy-side workaround. |
| "Compare CPCV vs walk-forward on the same strategy" | Two separate `MultiPathEngine` runs, one with each splitter. Both produce `MultiPathResult` you can `summary()`. | None — proceed, but log both runs as separate trials in the registry (skill 11) so DSR counts them. |
| "Use different RNG seeds per path" | Factory pattern: `lambda i=i: MyStrategy(seed=i)` per split — but the splitter doesn't expose `i` to the factory. | Workaround: factory closes over a global counter or uses `time.time_ns()`. Surface as a mismatch — the cleanest fix is a strategy that seeds itself from a known input (e.g. the panel's first date). |
| "Use joblib threads, not processes" | joblib defaults to `prefer="processes"`. `MultiPathEngine` doesn't expose `prefer`. | Run-time only: set `joblib` backend before `.run()`, or live with processes. Surface if startup overhead is a real concern. |
| "Persist per-split results to disk" | `MultiPathEngine` does not persist; the registry (skill 11) does, but for the aggregated trial. | (a) Use the registry on the aggregate. (b) Write a custom wrapper that pickles `split_returns` after `.run()`. Recommend (a). |
| "Show me equity over a custom time window" | `path_equity` covers `data.dates` (CPCV) or `test` (WF) only. Slicing to a sub-window is a `.loc` call by the user, not an engine arg. | None — slice in the visualization cell. |
| "Aggregate annualized return / vol across paths" | Not in `aggregate_metrics()` keys. | Compute from `metrics_per_path()` directly: `np.mean([r.ann_return for r in res.metrics_per_path()])`. Surface as a mismatch if the user expected it in the aggregate dict. |

---

## 9. Minimal valid cell — Stage 6, multi-path run

The shape of the second Stage 6 cell. Splitter must already be constructed (skill 06). Costs and liquidity carry over from Stage 3.

```python
from backtest.engine import MultiPathEngine

mp = MultiPathEngine(
    panel,
    strat,                       # the same instance (or a factory) from Stage 2
    splitter,                    # from skill 06's cell
    costs=costs,                 # from Stage 3 — REQUIRED
    liquidity=liq,               # from Stage 3
    initial_capital=1_000_000,
)
res = mp.run(n_jobs=-1)

# Visible artifact for validation (PROTOCOL §4 / §5).
print(f"Splits run:           {res.n_splits}")
print(f"Paths assembled:      {res.n_paths}")
print(f"path_returns shape:   {res.path_returns.shape}")
print(f"path_equity shape:    {res.path_equity.shape}")
print(f"Date range:           {res.path_returns.index[0].date()} → {res.path_returns.index[-1].date()}")
print(f"Per-path NaN count:   min={res.path_returns.isna().sum().min()}, max={res.path_returns.isna().sum().max()}")
print()
print("Final equity per path:")
print(res.path_equity.iloc[-1].rename("final_equity").map(lambda v: f"${v:,.0f}"))
```

Do **not** call `res.summary()` or `res.aggregate_metrics()` in this cell. Those are Stage 7 (skill 08) — measuring, not running.

---

## 10. Validation checklist (after the cell runs)

- [ ] **`res.n_splits`** matches what the splitter cell printed in skill 06.
- [ ] **`res.n_paths`** matches what the splitter cell printed.
- [ ] **`res.path_returns.shape[1] == res.n_paths`** — column count is paths, not splits.
- [ ] **`res.path_returns.shape[0]`** equals `len(panel.dates)` for CPCV with full coverage, or the test-window length for walk-forward.
- [ ] **`res.path_returns.isna().sum().max() <= n_groups`** (CPCV) or `== 1` (WF). One NaN per contiguous segment per path is the engine convention; more than that is a bug.
- [ ] **Final equity per path** is positive, finite, in a sane range. Wildly negative → catastrophic bug; do not proceed. Wildly positive (>10000%) → likely strategy lookahead.
- [ ] **All final equities differ from each other.** If `path_equity.iloc[-1]` has duplicates across paths, the splitter is degenerate (e.g. `n_test_groups=1` deliberately gives `n_paths=1`; otherwise check the splitter cell).
- [ ] No `ValueError` from inside a joblib worker. If you see one, rerun with `n_jobs=1` to surface the traceback inline.

If any fails: state which, do not proceed to Stage 7. Propose a fix-cell — usually a different splitter (skill 06), or a fix in Stage 2/3.

---

## 11. What NOT to do

- **Do not pass `costs=None`** to `MultiPathEngine` unless the user has explicitly asked for a gross-vs-net comparison run. `PROTOCOL.md §6`.
- **Do not call `.summary()`** in the run cell. That's Stage 7 (skill 08). Stage 6 only verifies the run completed cleanly.
- **Do not assume `MultiPathEngine.__init__`** catches the same errors as single-path `Engine.__init__`. Only `initial_capital <= 0` is caught at construction. Required-data and liquidity-needs-volume failures fire inside joblib workers at `.run()` time.
- **Do not assume `cash_rate` / `borrow_rate` default to non-zero.** They default to `0.0`. If the spec asks for cash interest, pass the constructor argument — the engine will not infer it.
- **Do not reach into `res.split_returns`** to reconstruct per-split equity curves and treat them as paths. Splits and paths are different (skill 06 §3.5); the canonical equity curves are in `res.path_equity`.
- **Do not iterate the splitter yourself to get per-split `BacktestResult`s.** They're already retained at `res.split_results[split_idx][segment_idx]`. Reaching back into `splitter.split(dates)` and running `Engine` manually re-does the work the engine just did.
- **Do not panic when a parallel run prints a `UserWarning` about retrying with `n_jobs=1`.** The engine caught a worker exception and fell back automatically so the real traceback can surface. The eventual `raise` is the actionable one — read it, fix the strategy, re-run.
- **Do not assume `aggregate_metrics()` includes annualized return / vol / total costs.** It only includes the nine listed in §6.2. Compute the rest from `metrics_per_path()`.
- **Do not treat `res.path_returns` rows as continuous returns across CPCV groups.** Between groups (where a fresh `Engine.run` started), positions were reset to zero and cash to `initial_capital`. The path is a *reconstructed* equity curve, not a real-time held-portfolio P&L.
- **Do not pass a `Strategy` class object** (e.g. `MyStrategy`) as `strategy` — pass either an instance (`MyStrategy()`) or a factory (`lambda: MyStrategy()`). A class is callable, so the `_is_factory` check accepts it, but `MyStrategy()` would then fail at the per-split call if `__init__` has required arguments.
- **Do not mutate the prototype strategy** between calls to `.run()`. Deep-copy isolates each split, but if you change the prototype's attributes between runs, subsequent runs see the changes.
- **Do not pass dates outside the panel** via `run(dates=...)`. `_contiguous_segments` will raise `ValueError("test contains dates not in data panel")` from inside a worker.
- **Do not use `MultiPathEngine` for a single-path backtest.** It works (`WalkForward(train_pct=...)` gives 1 split, 1 path), but it's heavier than `Engine` — joblib startup, deep-copy, segment splitting — for no gain. Stage 4 is for single-path; Stage 6 is for multi-path.

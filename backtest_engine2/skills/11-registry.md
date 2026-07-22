# Skill 11 — Trial Registry

> **Read `PROTOCOL.md` and `skills/00-overview.md` first.** Cell-loop and announcement rules in PROTOCOL apply.

> Stage 10 — required only when the user wants to commit results (PROTOCOL §8). Captures every trial in a persistent store so subsequent DSR / effective-K queries see the *real* trial count, not the user-reported one.

---

## When to load

- **Stage 10**: persisting a trial with `TrialRegistry.run(engine, causal_graph_path=...)` or logging a pre-computed result with `TrialRegistry.log(...)`.
- Any cell that imports `TrialRegistry`, `strategy_hash`, or `dataset_id_for`.
- Any cell where the user asks for "K from the registry" or "DSR pulling K from logged trials" — this is the file that owns K.

---

## What you'll find here

1. The two-file store layout (SQLite + parquet).
2. `TrialRegistry(directory)` constructor.
3. `register.log(result, strategy, data, causal_graph_path=..., family=...)`.
4. `register.run(engine, causal_graph_path=..., family=...)` — the convenience wrapper.
5. Querying: `list`, `get`, `returns`, `returns_df`.
6. `register.k(family=..., method="count" | "effective", threshold=...)`.
7. `register.dsr_for(trial_id, ...)` — DSR with registry-supplied K.
8. `strategy_hash` and `dataset_id_for` — what the registry uses to dedupe.
9. The causal-graph requirement — why it exists, how `skip_causal_graph=True` interacts with K.
10. Common mismatches.
11. Minimal valid cell.
12. Validation checklist.
13. Anti-patterns.

---

## 1. Store layout

```
<directory>/
├── registry.sqlite          # SQLite index of trials (one row per trial)
└── returns/
    ├── 00000001.parquet     # trial 1's returns series
    ├── 00000002.parquet
    └── ...
```
`source: backtest/registry/registry.py:52–57, 131`

The SQLite database holds metadata; the per-trial returns are stored as parquet for fast columnar reads when computing effective-K across many trials. `returns/<id:08d>.parquet` is the canonical filename pattern.

### 1.1 SQLite schema

```sql
CREATE TABLE trials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,         -- ISO 8601 UTC
    strategy_class TEXT NOT NULL,    -- "<module>.<ClassName>"
    strategy_hash TEXT NOT NULL,     -- strategy_hash(strategy)
    strategy_repr TEXT,              -- first 500 chars of repr(strategy)
    family TEXT,                     -- user-supplied group key
    causal_graph_path TEXT,
    exploratory INTEGER NOT NULL,    -- 1 if skip_causal_graph=True
    dataset_id TEXT NOT NULL,        -- dataset_id_for(panel) or user-supplied
    train_start TEXT, train_end TEXT,
    test_start TEXT,  test_end TEXT,
    n_obs INTEGER,
    sharpe REAL, psr REAL, max_drawdown REAL,
    returns_path TEXT NOT NULL       -- absolute path to the parquet file
);
CREATE INDEX idx_dataset ON trials(dataset_id);
CREATE INDEX idx_family ON trials(family);
```
`source: backtest/registry/registry.py:18–41`

Only a few metric columns are denormalized into the SQLite row (`sharpe`, `psr`, `max_drawdown`, `n_obs`). For anything else, load the parquet and re-`compute_metrics`. The schema is intentionally minimal — heavy stats live in the returns parquet.

---

## 2. `TrialRegistry(directory)`

```python
TrialRegistry(directory: str | Path)
```
`source: backtest/registry/registry.py:52–57`

Creates `directory/`, `directory/returns/`, and `directory/registry.sqlite` if they don't exist. Idempotent — re-opening an existing directory binds to the existing store.

```python
from backtest.registry import TrialRegistry
reg = TrialRegistry("./trial_store")
```

There is no destructor / `close()`. Each `log`, `list`, etc. opens and closes its own connection. Safe across processes (SQLite handles concurrent reads; writes serialize).

---

## 3. `register.log(...)`

```python
log(
    result: BacktestResult,
    strategy: Strategy,
    data: DataPanel,
    *,
    causal_graph_path: str | None = None,
    family: str | None = None,
    dataset_id: str | None = None,
    skip_causal_graph: bool = False,
    verify_path: bool = True,
) -> int                          # returns the new trial_id
```
`source: backtest/registry/registry.py:72–148`

### 3.1 What gets stored

1. The strategy's class (`module.Class`), `strategy_hash` (16-char SHA-256 of `module.Class | repr(strategy)`), and the first 500 chars of `repr(strategy)`.
2. `dataset_id_for(data)` (auto) or the user-supplied `dataset_id`.
3. The test window (`test_start`, `test_end`) from `result.returns.index`. `train_start` / `train_end` are always `NULL` from `log` (you'd have to fill them yourself by directly mutating the DB if you cared).
4. The denormalized metrics: `n_obs`, `sharpe`, `psr`, `max_drawdown` from `compute_metrics(result.returns, result.equity_curve)`.
5. The full `result.returns` series, written to `returns/<id:08d>.parquet`.
6. A `timestamp` (UTC, ISO 8601).
7. `causal_graph_path` and the `exploratory` flag.

### 3.2 The causal-graph contract

`source: backtest/registry/registry.py:84–93`

```
if not skip_causal_graph:
    if causal_graph_path is None:    → ValueError
    if verify_path and not Path(causal_graph_path).exists():
                                     → FileNotFoundError
```

The default flow **requires** a causal-graph image path that points to an existing file. Pass `skip_causal_graph=True` to log an exploratory trial that doesn't need the graph; that trial is marked `exploratory=1` in the DB and will be excluded from `k(...)` (see §6.2). `verify_path=False` lets you pass a path string without filesystem verification — rarely correct; usually a sign you're not actually committing to a hypothesis.

### 3.3 Atomicity

`source: backtest/registry/registry.py:111–146`

The SQLite INSERT and the parquet write happen inside a single `with conn:` transaction. If the parquet write fails (disk full, etc.) the partial file is cleaned up and the SQLite row is rolled back. No half-logged trials.

`returns_path` is set in two steps — initially empty string, then UPDATE'd to the resolved path once the trial_id is known. Both updates are inside the transaction. A reader who sees a SQLite row will always see a complete `returns_path` pointing to an existing parquet.

---

## 4. `register.run(engine, ...)` — convenience wrapper

```python
run(
    engine: Engine,
    *,
    causal_graph_path: str | None = None,
    family: str | None = None,
    dataset_id: str | None = None,
    skip_causal_graph: bool = False,
    verify_path: bool = True,
    start=None, end=None,
    train_dates=None, test_dates=None,
) -> BacktestResult
```
`source: backtest/registry/registry.py:150–176`

Calls `engine.run(start, end, train_dates, test_dates)` then `self.log(result, engine.strategy, engine.data, ...)`. Returns the `BacktestResult` so the caller can inspect it.

This is the canonical Stage 10 entry point. The user constructs an `Engine(panel, strat, costs=..., liquidity=...)`, then calls `reg.run(engine, causal_graph_path=..., family=...)` once. Don't call `engine.run(...)` separately first — it would produce a result you'd need to remember to pass to `reg.log`, and missing that step leaves the trial uncounted.

---

## 5. Querying

### 5.1 `list(family=..., dataset_id=..., include_exploratory=True) -> list[dict]`

`source: backtest/registry/registry.py:178–201`

Returns every matching trial as a dict (column→value). Order: ascending `id`. `family` and `dataset_id` are optional filters; `include_exploratory=False` drops trials logged with `skip_causal_graph=True`.

### 5.2 `get(trial_id: int) -> dict`

`source: backtest/registry/registry.py:203–213`

Single trial by id. Raises `KeyError("trial {id} not found")` on miss.

### 5.3 `returns(trial_id: int) -> pd.Series`

`source: backtest/registry/registry.py:215–220`

Loads the per-trial parquet and returns the `ret` column as a `pd.Series` named `"returns"`. Convenient when the user wants to re-`compute_metrics` with different `ann_factor` or pass to `dsr` manually.

### 5.4 `returns_df(family=..., dataset_id=..., include_exploratory=False) -> pd.DataFrame`

`source: backtest/registry/registry.py:222–237`

Loads **all** matching trials' returns as columns of one DataFrame, named `trial_{id}`. Indices are unioned by pandas. **Default `include_exploratory=False`** here, unlike `list()`'s default — exploratory trials are excluded by default from the returns matrix because the matrix is typically fed into effective-K, which should not see them.

If no trials match, returns an empty DataFrame.

---

## 6. K — `register.k(family=..., method=..., threshold=...)`

```python
k(
    *,
    family: str | None = None,
    dataset_id: str | None = None,
    method: str = "count",            # or "effective"
    threshold: float = 0.5,
) -> int
```
`source: backtest/registry/registry.py:239–264`

The canonical K for downstream DSR. Two methods:

### 6.1 `method="count"`

`source: backtest/registry/registry.py:247–254`

Returns `len(list(..., include_exploratory=False))`. The raw count of non-exploratory trials in the family / dataset. Fast, no parquet reads.

### 6.2 `method="effective"`

`source: backtest/registry/registry.py:255–263`

Loads `returns_df(...)` and calls `effective_k(df, threshold=threshold)` (skill 09 §4). Returns the correlation-corrected K. **More honest** when trials are correlated (parameter sweeps usually are). Slightly slower because it reads every trial's parquet.

Returns `0` if no trials match. Callers should guard if they intend to feed this into `dsr` (which raises on `K < 1`).

### 6.3 Exploratory trials are excluded from K

This is the design point that justifies the causal-graph requirement: only trials with a committed hypothesis (graph) count toward the family's K. Exploratory log-and-explore runs (with `skip_causal_graph=True`) live in the DB for traceability but don't inflate K. PRD §12 specifies this.

---

## 7. DSR with registry-supplied K — `register.dsr_for(trial_id, ...)`

```python
dsr_for(
    trial_id: int,
    *,
    family: str | None = None,
    dataset_id: str | None = None,
    method: str = "count",            # or "effective"
    threshold: float = 0.5,
) -> float
```
`source: backtest/registry/registry.py:266–284`

Loads `returns(trial_id)`, computes `K = self.k(...)`, calls `dsr(returns, K=K)`. If `K < 1` (no committed trials in the family), promotes K to 1 — DSR-at-K=1 equals PSR-vs-zero (skill 09 §7) and is the right floor.

Use this instead of `dsr(returns, K=user_supplied_K)` whenever the registry exists. The whole point of the registry is that K is not a user-reported number.

---

## 8. `strategy_hash` and `dataset_id_for`

`source: backtest/registry/hashing.py:11–33`

### 8.1 `strategy_hash(strategy: Strategy) -> str`

```python
rep = f"{module}.{ClassName}|{repr(strategy)}"
return hashlib.sha256(rep.encode()).hexdigest()[:16]
```

A 16-char hex digest. Used by SQLite for the `strategy_hash` column and by `FingerprintCache` (skill 10 §7.2) for cache lookups.

**Stability depends on `repr(strategy)`.** Default Python `repr` includes the object's memory address — two identical `MyStrategy(60)` instances will hash differently. **Override `__repr__` to depend only on `__init__` args.** `STRATEGY_GUIDE.md` (root) covers this; the rule is in skill 02. Without a stable repr, the registry can't dedupe and downstream caches behave erratically.

### 8.2 `dataset_id_for(panel: DataPanel) -> str`

```python
cols      = ",".join(panel.assets_all)
start, end = ISO timestamps of panel.dates[0], panel.dates[-1]
n         = len(panel.dates)
head_sum  = panel._prices.head(5).sum().sum()
tail_sum  = panel._prices.tail(5).sum().sum()
return sha256(f"{cols}|{start}|{end}|{n}|{head_sum:.4f}|{tail_sum:.4f}")[:16]
```

A 16-char hash that fingerprints the panel: asset list + date span + length + price head/tail sums. Two panels with the same assets, dates, and prices produce the same `dataset_id`. Adding even one row of prices, or perturbing one bar's price, changes the head/tail sum and produces a different ID.

This lets you filter `list` / `returns_df` / `k` by `dataset_id` so trials run on different datasets (different universes, different splits) don't pollute each other's K. The auto-derived ID is good enough for most cases; supply your own when you have a canonical reference (e.g. a particular vintage of data).

---

## 9. Common mismatches — surface these by default

| Spec says | Engine has | Question to ask |
|---|---|---|
| "Persist trials to a folder" | `TrialRegistry("./trial_store")`. | Confirm the folder name. Use a separate folder per project / experiment campaign. |
| "I don't have a causal graph yet" | `log(..., skip_causal_graph=True)` flags the trial exploratory; it's stored but excluded from K. | Confirm: is this trial actually exploratory, or is the graph just deferred? Recommend drawing even a rough graph rather than `skip_causal_graph` becoming the default. |
| "Use my own dataset ID" | `log(..., dataset_id="my_custom_id")` overrides the auto-derived one. | Confirm the user has a stable canonical ID system. Auto-derived is usually fine. |
| "Family of trials called `momentum_xs`" | `log(..., family="momentum_xs")` — string column for grouping. | Confirm the family naming convention; recommend snake_case strings that match across sessions. |
| "Use effective-K instead of raw count" | `register.k(family=..., method="effective", threshold=0.5)`. | Confirm threshold. Default 0.5 means "trials correlating > 0.5 collapse to one cluster". Recommend keeping the same threshold across DSR queries in a family. |
| "Run a parameter sweep with K logged per parameter" | Loop: for each parameter, construct `Engine(panel, MyStrategy(p), ...)` and call `reg.run(engine, family="my_family", ...)`. K updates automatically. | Confirm each parameter gets its own causal-graph path (or `skip_causal_graph=True` for exploration). Recommend logging each as its own trial — that's the whole point. |
| "Recompute trial metrics with different `ann_factor`" | Load `reg.returns(trial_id)`, call `compute_metrics(..., ann_factor=ann)`. Stored denormalized columns (sharpe, psr, max_drawdown) use default 252. | Confirm. Recommend that the user re-load from the parquet rather than trusting the SQLite-cached numbers. |
| "Drop a logged trial" | Not exposed. Manual SQL would work but the registry doesn't expose a `delete` method. | Surface as a mismatch. Deleting a trial after running is itself a research-integrity concern (it cherry-picks K). Recommend keeping the trial logged and treating it appropriately. |
| "K for an external trial set" | Registry only knows what's been logged. If trials were run outside the registry, K is incomplete. | Surface. Recommend logging every trial through the registry from the start; retroactively reconstructing K from notebooks is fragile. |
| "Trial returns are huge — don't store them" | The registry stores the full returns Series per trial in parquet. Compressed but not tiny — at ~5000 bars × float64 = 40 KB/trial. | Acceptable unless the user is doing 1M+ trials. Suggest a separate sweep storage strategy if it's a concern. |
| "I want to filter trials by `repr(strategy)` substring" | Not in the public API. The SQLite column `strategy_repr` stores up to 500 chars; query directly via `sqlite3.connect(...)`. | Surface as a mismatch. Recommend `family` for first-class grouping; `strategy_hash` for exact dedupe; raw SQL for substring searches. |

---

## 10. Minimal valid cell — Stage 10, register a trial

```python
from backtest.registry import TrialRegistry

reg = TrialRegistry("./trial_store")

result = reg.run(
    engine,                                          # the Engine from Stage 4
    causal_graph_path="diagrams/momentum_thesis.png", # must exist on disk
    family="momentum_xs",                            # group key — sweep variants under same family
    start=panel.dates[252],                          # same warmup as Stage 4
)

# Visible artifact for validation.
trials = reg.list(family="momentum_xs", include_exploratory=False)
K_count = reg.k(family="momentum_xs", method="count")
K_eff = reg.k(family="momentum_xs", method="effective", threshold=0.5)
deflated = reg.dsr_for(trials[-1]["id"], family="momentum_xs", method="effective", threshold=0.5)

print(f"Family:           momentum_xs")
print(f"Trials logged:    {len(trials)}")
print(f"K (count):        {K_count}")
print(f"K (effective@0.5):{K_eff}")
print(f"This trial's DSR: {deflated:.3f}  (K_eff = {K_eff})")
print(f"Trial id:         {trials[-1]['id']}")
print(f"Storage dir:      {reg.directory}")
```

For an exploratory log (no causal graph):

```python
result = reg.run(
    engine,
    causal_graph_path=None,
    skip_causal_graph=True,                          # marks exploratory; excluded from K
    family="momentum_xs",
)
```

Do **not** persist a trial that you haven't run with proper costs (`PROTOCOL.md §6`). Do **not** persist a trial without DSR being checked first (Stage 9 is required before Stage 10).

---

## 11. Validation checklist (after the cell runs)

- [ ] **`trials`** list contains the newly logged trial (id is the highest).
- [ ] **`K_count`** matches `len(trials)`.
- [ ] **`K_eff <= K_count`** (effective-K never exceeds raw count by construction).
- [ ] **`reg.directory / "registry.sqlite"`** exists and `reg.directory / "returns" / "<id:08d>.parquet"` exists for the new trial.
- [ ] **`reg.get(trial_id)`** returns a populated dict (no NULL columns except `train_start`, `train_end` which `log` doesn't fill).
- [ ] **`reg.returns(trial_id).notna().sum()`** equals `result.returns.notna().sum()` — the parquet round-trip preserved all observations.
- [ ] **`exploratory` column** is `0` if `causal_graph_path` was supplied, `1` if `skip_causal_graph=True`.
- [ ] **DSR is < PSR for K > 1** (DSR is more conservative; if equal, K=1).
- [ ] **`strategy_hash`** is a 16-char hex string (not `None`, not empty). If the user has a default `__repr__`, the hash includes a memory address and will differ across re-instantiations — flag that as a stable-`__repr__` requirement.

If any fails: state which. The registry should be the source of truth for the K downstream consumers see; a broken trial entry there breaks DSR everywhere.

---

## 12. What NOT to do

- **Do not log a trial without a causal-graph path** unless the user has explicitly accepted that it's exploratory (and therefore excluded from K). The causal-graph requirement is the whole point — pressing past it dilutes the registry's value.
- **Do not call `engine.run(...)` followed by `reg.log(...)`** if `reg.run(engine, ...)` does both. Two-step is fine for explicit control; just don't forget the `log` call — an un-logged trial is invisible to K and DSR.
- **Do not pass `skip_causal_graph=True` by default.** Use it deliberately for exploration. Default to providing a path.
- **Do not assume `register.k()` always returns the same number across sessions.** It reflects the registry's current state. Re-running queries after new trials are logged returns a higher K. This is the desired behavior — DSR should deflate against the actual trial count.
- **Do not delete trials from `registry.sqlite`** to lower K. Cherry-picking K is the bias the registry was built to prevent. If a trial was wrong (e.g., used the wrong costs), re-log it correctly with a note in `family` — don't erase the bad one.
- **Do not write to `registry.sqlite` from multiple processes simultaneously** without coordination. SQLite serializes writes, but a long parquet write blocks other inserts. For parameter sweeps, run sequentially or use one registry per worker and merge later.
- **Do not assume `dataset_id_for(panel)` is stable across pandas / numpy versions.** It uses `panel._prices.head(5).sum()` rounded to 4 decimals — robust against small float jitter but not invariant to data reload from different sources. Use a user-supplied `dataset_id` for canonical pinning.
- **Do not use the registry as a backup of `BacktestResult`.** It stores only the returns Series and a few denormalized columns. `weights`, `positions`, `trades`, `costs`, `unfilled`, full `equity_curve` are **not** persisted. If you need those, save them separately.
- **Do not load `returns_df` for a giant family** and pass it to `effective_k` without checking the column count first. At 5000+ trials and 5000-bar returns, that's a 200MB DataFrame and a `O(n²)` correlation matrix.
- **Do not rely on `strategy_repr`** column for round-tripping the strategy. It's truncated to 500 chars and stored as text — not a serialization format. To reproduce a trial's strategy, you need the class and the original `__init__` args.
- **Do not store credentials, paths, or any sensitive data in `family` or `dataset_id`.** They're plain text in the SQLite column and the parquet filenames. Use opaque strings.
- **Do not log a trial whose DSR is `NaN`.** A NaN DSR means the metrics are degenerate (constant returns, < 2 obs); persisting it as a "trial" inflates K with junk. Fix Stage 4 / 6 first.
- **Do not mix `family` across different evaluation procedures** (CPCV vs walk-forward, different splitters, different cost models). DSR's K only makes sense within a comparable evaluation family. Surface this if the user is about to log two procedurally-different runs under the same family name.

# Skill 06 — Splitters

> **Read `PROTOCOL.md` and `skills/00-overview.md` first.** Cell-loop and announcement rules in PROTOCOL apply.

> Stages 1–4 must already be done. Splitters are consumed by `MultiPathEngine` (skill `07-multipath.md`), not by `Engine`. This file covers the splitter object itself; the run lives in skill 07.

---

## When to load

- **Stage 6, splitter cell**: constructing a `WalkForward` or `CombinatorialPurgedCV` and inspecting `n_splits()` / `n_paths()` before wiring it into `MultiPathEngine`.
- Any cell that imports from `backtest.splitters`.
- Any cell that needs to reason about purge / embargo / fold counts.

---

## What you'll find here

1. The `Splitter` ABC — the four-method contract `MultiPathEngine` relies on.
2. `WalkForward` — single-path train/test split with optional embargo.
3. `CombinatorialPurgedCV` — group construction, combination iteration, purge vs embargo (and what the trailing buffer **actually** is in code).
4. **Splits vs paths** — they are not the same number, and confusing them changes the meaning of every downstream metric.
5. `assemble_paths` — how `MultiPathEngine` stitches per-split returns into per-path series.
6. How splitters feed `MultiPathEngine` downstream.
7. Common mismatches.
8. Minimal valid cell.
9. Validation checklist.
10. Anti-patterns.

---

## 1. The `Splitter` ABC

```python
class Splitter(ABC):
    @abstractmethod
    def split(self, dates: pd.DatetimeIndex) -> Iterator[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]: ...
    @abstractmethod
    def n_splits(self) -> int: ...
    @abstractmethod
    def n_paths(self) -> int: ...
    @abstractmethod
    def assemble_paths(
        self,
        all_dates: pd.DatetimeIndex,
        split_returns: list[pd.Series],
    ) -> pd.DataFrame: ...
```
`source: backtest/splitters/splitters.py:14–31`

| Method | What it returns | Who calls it |
|---|---|---|
| `split(dates)` | An iterator of `(train_dates, test_dates)` tuples — one per train/test run. | `MultiPathEngine.run` materializes the list and runs one `Engine` per pair. `source: backtest/engine/multi_path.py:156–157` |
| `n_splits()` | The number of `(train, test)` tuples `split` will yield. | Sanity check; equals `len(list(split(dates)))`. |
| `n_paths()` | The number of independent full-timeline equity paths reconstructable from the splits. **Not always equal to `n_splits()`.** | Sizes `MultiPathResult.path_returns` / `path_equity`. `source: backtest/engine/multi_path.py:180–191` |
| `assemble_paths(all_dates, split_returns)` | A `DataFrame` indexed by `all_dates`, with one column per path (`path_0, path_1, …`), built by concatenating test-segment returns from different splits. | `MultiPathEngine.run` after collecting all per-split returns. `source: backtest/engine/multi_path.py:180` |

Only two concrete implementations exist: `WalkForward` and `CombinatorialPurgedCV`. `source: backtest/splitters/__init__.py:1–7`

---

## 2. `WalkForward`

```python
WalkForward(
    train_end: pd.Timestamp | None = None,
    train_pct: float | None = None,
    embargo_bars: int = 0,
)
```
`source: backtest/splitters/splitters.py:42–56`

Single-path train/test split: train on the past, test on the future. One iteration, one path.

### 2.1 Construction-time validation

`ValueError` at construction in three cases. `source: backtest/splitters/splitters.py:48–53`

1. **Neither or both of `train_end` / `train_pct` set** — `"specify exactly one of train_end or train_pct"`. The check is the XOR `(train_end is None) == (train_pct is None)`.
2. **`train_pct` not in `(0, 1)` exclusive** — `"train_pct must be in (0, 1)"`. So `0`, `1`, and negatives all fail.
3. **`embargo_bars < 0`** — `"embargo_bars must be >= 0"`.

### 2.2 Behavior

`split(dates)` yields exactly one `(train, test)` tuple. `source: backtest/splitters/splitters.py:58–67`

- If `train_end` is set, the split index is `dates.searchsorted(train_end, side="right")`. **`train_end` is inclusive** — the last train bar *is* `train_end`. First test bar is `dates[split_idx]`. `source: backtest/splitters/splitters.py:60–61, splitters/splitters_test.py:24–30`
- If `train_pct` is set, the split index is `int(len(dates) * train_pct)` (floored).
- `embargo_bars` shortens train: `train = dates[:split_idx - embargo_bars]`. Test is `dates[split_idx:]` regardless. So embargo eats train, not test. `source: backtest/splitters/splitters.py:64–66, splitters/splitters_test.py:33–37`

`n_splits() == 1` and `n_paths() == 1`. `source: backtest/splitters/splitters.py:69–73`

`assemble_paths` returns a `DataFrame` with one column `path_0` indexed by the test dates. Raises `ValueError` if you pass anything other than exactly one `split_returns` series. `source: backtest/splitters/splitters.py:75–82`

### 2.3 Typical call

```python
from backtest.splitters import WalkForward

splitter = WalkForward(train_pct=0.7, embargo_bars=5)
```

70% of bars in train, 5-bar embargo. `MultiPathEngine` fits the strategy once on train, then runs the test window. One path out.

---

## 3. `CombinatorialPurgedCV`

```python
CombinatorialPurgedCV(
    n_splits: int = 6,
    n_test_groups: int = 2,
    purge_bars: int = 0,
    embargo_pct: float = 0.01,
)
```
`source: backtest/splitters/splitters.py:96–116`

Combinatorial Purged Cross-Validation (López de Prado 2018, *Advances in Financial Machine Learning*, Ch. 12).

### 3.1 Construction-time validation

`ValueError` at construction in five cases. `source: backtest/splitters/splitters.py:103–112`

1. **`n_splits < 2`** — `"n_splits must be >= 2"`. There is **no** `CPCV(n_splits=1)`. Use `WalkForward` for the single-split case.
2. **`n_test_groups < 1`** — `"n_test_groups must be >= 1"`.
3. **`n_test_groups >= n_splits`** — `"n_test_groups must be < n_splits"`.
4. **`purge_bars < 0`** — `"purge_bars must be >= 0"`.
5. **`embargo_pct` outside `[0, 1)`** — `"embargo_pct must be in [0, 1)"`. So `0.0` is allowed; `1.0` is not.

Note the naming inconsistency: the constructor argument is `n_splits`, but it is stored on the instance as `self.n_groups`. The method `n_splits()` returns `C(n_groups, n_test_groups)`, not the constructor argument. `source: backtest/splitters/splitters.py:113, 136–137`

### 3.2 Group construction

The timeline is sliced into `n_splits` (= `n_groups`) contiguous groups by `np.linspace`:

```python
bounds = np.linspace(0, n, self.n_groups + 1, dtype=int)
groups = [(bounds[i], bounds[i + 1]) for i in range(self.n_groups)]
```
`source: backtest/splitters/splitters.py:125–126`

With `n = 120, n_splits = 6`: each group is 20 bars. Group sizes can differ by ±1 when `n` doesn't divide evenly. With `n_test_groups=2`, each test set is the union of two groups = ~40 bars. `source: backtest/splitters/splitters_test.py:154–158`

### 3.3 Combination iteration

For every combination of `n_test_groups` groups out of `n_groups`, the test mask is the union of those groups; train is the complement, minus purge/embargo. `source: backtest/splitters/splitters.py:128–134`

```python
for combo in itertools.combinations(range(self.n_groups), self.n_test_groups):
    ...
```

- Number of train/test runs: `C(n_groups, n_test_groups)`. With defaults: `C(6, 2) = 15` splits. `source: backtest/splitters/splitters.py:136–137, splitters/splitters_test.py:59–62`
- Iteration order is lexicographic and deterministic. Don't rely on a specific ordering downstream — it changes with `n_splits` / `n_test_groups`.

### 3.4 Purge and embargo — exact behavior in code

This matches the López de Prado textbook recipe: purge on **both** sides of each test run, embargo as an additional buffer **after** each test run. They stack on the trailing side. `source: backtest/splitters/splitters.py:177–192`

`embargo_bars = int(n * embargo_pct)` — floored via `int(...)`. With `n = 119, embargo_pct = 0.05`: `int(5.95) = 5`, not 6. `source: backtest/splitters/splitters.py:123`

For each contiguous test run `[s, e)`, the train mask is zeroed out over the half-open interval:

```python
lo = max(0, s - purge_bars)
hi = min(n, e + purge_bars + embargo_bars)
train[lo:hi] = False
```
`source: backtest/splitters/splitters.py:187–191`

So:

- **Leading side (before test):** drop `purge_bars` train bars. Embargo does **not** apply on the leading side.
- **Trailing side (after test):** drop `purge_bars + embargo_bars` train bars. Purge and embargo stack.

Consequences:

| `purge_bars` | `embargo_bars` (= int(n·embargo_pct)) | Leading | Trailing | Notes |
|---|---|---|---|---|
| 0 | 5 | 0 | 5 | Embargo-only (no purge). `source: splitters_test.py:102–114` |
| 2 | 0 | 2 | 2 | Pure purge, symmetric. `source: splitters_test.py:86–99` |
| 2 | 5 | 2 | 7 | Standard López de Prado: leading purge + trailing purge + embargo. `source: splitters_test.py:221–246` |
| 5 | 1 | 5 | 6 | Both stack: trailing = 5 + 1. |

The early-return path skips this loop entirely when `purge_bars == 0 and embargo_bars == 0`. `source: backtest/splitters/splitters.py:184–185`

### 3.5 `n_splits` vs `n_paths` — the conceptual hump

```python
def n_splits(self):  return comb(n_groups, n_test_groups)
def n_paths(self):   return comb(n_groups - 1, n_test_groups - 1)
```
`source: backtest/splitters/splitters.py:136–140`

| `n_splits` (param) | `n_test_groups` | Splits (trainings, runs) | Paths (equity curves) |
|---|---|---|---|
| 6 | 2 | 15 | 5 |
| 10 | 2 | 45 | 9 |
| 10 | 3 | 120 | 36 |
| 6 | 1 | 6 | 1 |
| 4 | 1 | 4 | 1 |

**Intuition.** Each of the `n_groups` timeline groups appears in the test set of `C(n_groups - 1, n_test_groups - 1)` of the total `C(n_groups, n_test_groups)` splits. By cycling through which split provides the test return for each group, you stitch together `C(n_groups - 1, n_test_groups - 1)` independent full-timeline equity series. Those are the paths. `source: backtest/splitters/splitters_test.py:117–128, splitters/splitters.py:160–175`

So **`n_splits` is the cost** (how many fits and engine runs you pay for) and **`n_paths` is the deliverable** (how many equity curves you measure). `MultiPathResult.path_returns` has `n_paths` columns, not `n_splits`. `source: backtest/engine/multi_path.py:180–191`

With `n_test_groups == 1`: `n_paths == 1`. Each group is tested in exactly one split; there's nothing to stitch.

### 3.6 `assemble_paths` mechanics

You will not call this directly — `MultiPathEngine.run` calls it after collecting all per-split returns. But it's part of the public API and worth understanding.

The algorithm: `source: backtest/splitters/splitters.py:142–175`

1. Recompute the same `bounds = np.linspace(...)` to recover the `n_groups` timeline groups.
2. For each group `g`, enumerate the `n_paths` splits whose test set included `g`. Index this list with the path number `p`.
3. For path `p` and group `g`: take the `p`-th of those splits' return series, reindex to that group's dates.
4. Concatenate per-group segments to form a full-timeline return series for path `p`.

`assemble_paths` raises `ValueError` if `len(split_returns) != C(n_groups, n_test_groups)`. `source: backtest/splitters/splitters.py:155–158, splitters/splitters_test.py:215–218`

---

## 4. How splitters feed `MultiPathEngine`

The splitter is one of three required constructor arguments to `MultiPathEngine`. The engine:

1. Materializes the list of `(train, test)` tuples via `splitter.split(dates)`. `source: backtest/engine/multi_path.py:156–157`
2. Runs one `Engine` per split in parallel via joblib (`n_jobs=-1` by default). Each fold gets a fresh strategy (deep-copied from the prototype, or freshly instantiated if a zero-arg factory was passed). `source: backtest/engine/multi_path.py:159–178, 204–207`
3. For non-contiguous CPCV test sets (`n_test_groups > 1`), splits the test into contiguous segments and runs the engine per segment to avoid cross-segment drift artifacts. The per-segment returns are concatenated. `source: backtest/engine/multi_path.py:215–228, 231–247`
4. Calls `splitter.assemble_paths(dates, split_returns)` to stitch per-split returns into per-path series. `source: backtest/engine/multi_path.py:180`
5. Compounds equity: `path_equity = (1 + path_returns.fillna(0)).cumprod() * initial_capital`. `source: backtest/engine/multi_path.py:181`

Full `MultiPathResult` construction and downstream usage live in skill `07-multipath.md`.

---

## 5. Common mismatches — surface these by default

Use the question shape from `PROTOCOL.md §3`. Engine-faithful default first.

| Spec says | Engine has | Question to ask |
|---|---|---|
| "5-fold time-series cross-validation" | `WalkForward` is one split; `CombinatorialPurgedCV` is combinatorial, not classic k-fold. No time-series k-fold class exists. | (a) `CPCV(n_splits=5, n_test_groups=1)` — 5 splits, 1 path (each group tested once; no stitching). Closest existing primitive. (b) `CPCV(n_splits=5, n_test_groups=2)` if they want multiple paths — 10 splits, 4 paths. (c) Extend the engine. Recommend (a) and ask. |
| "Walk-forward with 1-year window expanding monthly" | The engine has no rolling/expanding walk-forward. `WalkForward` is one split, one fit. | (a) `WalkForward(train_pct=...)` — closest existing primitive, single split. (b) Extend the engine — separate task. (c) Use CPCV instead, accept that each path is a stitch, not a forward roll. Recommend (a). |
| "20% embargo" | `embargo_pct=0.20` is accepted (must be `< 1.0`) but with `n_splits=6, n_test_groups=2` you'll lose huge chunks of train per split. | Confirm the embargo size makes sense for the dataset length. With small `n`, train can shrink to zero. |
| "Purge ±5 bars on each side and embargo 10 bars after" | Engine stacks purge + embargo on the trailing side: `purge_bars=5, embargo_pct = 10/n` gives leading=5, trailing=15. | Confirm the `embargo_pct` translation given `n = len(panel.dates)`. |
| "Test on the most recent year, train on everything before" | `WalkForward(train_end=date_one_year_before_end)`. | Confirm `train_end` is **inclusive** (the last train bar is `train_end`). First test bar is the next date in the panel. |
| "Each path should use a different train window" | Each *split* has its own train window. Each *path* is a stitch of test segments from multiple splits, each with its own train window. Paths are not in 1-1 with train windows. | Surface the splits-vs-paths distinction (§3.5). The user may actually want walk-forward with refitting, not CPCV. |
| "Use sklearn's `TimeSeriesSplit`" | Not supported. `MultiPathEngine` only consumes `backtest.splitters.Splitter` subclasses. Sklearn time-series split lacks purge and embargo. | Recommend `WalkForward` or `CPCV`. |
| "Embargo of 5 days" with daily data | Engine takes `embargo_pct = embargo_bars / n` (computed by the user). | Convert: `embargo_pct = 5 / len(panel.dates)`. Confirm rounding via `int(n * embargo_pct)`. |
| "I want to see each fold's train window" | `splitter.split(dates)` yields them. Materialize as `list(splitter.split(panel.dates))` and inspect the `(train, test)` pairs. | This is fine for inspection but should be a separate diagnostic cell — do not bake it into the splitter cell. |

---

## 6. Minimal valid cell — Stage 6, splitter construction

The shape of the first Stage 6 cell. Construct the splitter and verify counts. The engine run lives in skill 07.

```python
from backtest.splitters import CombinatorialPurgedCV

splitter = CombinatorialPurgedCV(
    n_splits=6,
    n_test_groups=2,
    purge_bars=5,
    embargo_pct=0.01,
)

# Visible artifact for validation (PROTOCOL §4 / §5).
n = len(panel.dates)
embargo_bars = int(n * splitter.embargo_pct)
print(f"Timeline:           {n} bars  ({panel.dates[0].date()} → {panel.dates[-1].date()})")
print(f"Groups:             {splitter.n_groups} contiguous groups of ~{n // splitter.n_groups} bars each")
print(f"Splits (trainings): {splitter.n_splits()}")
print(f"Paths (curves):     {splitter.n_paths()}")
print(f"Purge:              {splitter.purge_bars} bars on each side of every test run")
print(f"Embargo:            {embargo_bars} bars after each test run (stacks with purge → trailing buffer = {splitter.purge_bars + embargo_bars})")

# Peek at the first split to confirm shapes.
first_train, first_test = next(iter(splitter.split(panel.dates)))
print(f"First split:        train={len(first_train)}, test={len(first_test)}, disjoint={len(first_train.intersection(first_test)) == 0}")
```

Do **not** instantiate `MultiPathEngine` or call `.run()` in this cell. That's a separate decision, a separate cell announcement, and lives in skill 07.

For walk-forward, the analogous cell is shorter:

```python
from backtest.splitters import WalkForward

splitter = WalkForward(train_pct=0.7, embargo_bars=5)
train, test = next(iter(splitter.split(panel.dates)))
print(f"Train: {len(train)} bars  ({train[0].date()} → {train[-1].date()})")
print(f"Test:  {len(test)} bars  ({test[0].date()} → {test[-1].date()})")
print(f"Gap between train end and test start: {(test[0] - train[-1]).days} days")
```

---

## 7. Validation checklist (after the cell runs)

- [ ] **`n_splits()`** equals `C(n_splits_param, n_test_groups)` (CPCV) or `1` (WF). This is the count of fits / engine runs the user is about to pay for.
- [ ] **`n_paths()`** equals `C(n_splits_param - 1, n_test_groups - 1)` (CPCV) or `1` (WF). This is the count of equity curves they will measure.
- [ ] **First split's train/test are disjoint** — printed verification.
- [ ] **Test size per split** (CPCV) ≈ `int(n / n_splits) * n_test_groups`, give or take group-boundary rounding.
- [ ] **`embargo_bars`** equals what they expected. `int(n * embargo_pct)` floors — surprise here if they expected ceil or round.
- [ ] **Train size per split** (CPCV) is large enough for a meaningful fit. With small `n` or aggressive purge/embargo, train can shrink dramatically per split.
- [ ] No `ValueError` from construction: that means all the constructor validations passed.

If any fails: state which. Propose a fix to the splitter args or to the panel date range. Do not proceed to skill 07 (the run) until the counts and sizes look right.

---

## 8. What NOT to do

- **Do not use `sklearn.model_selection.TimeSeriesSplit`, `KFold`, or any other sklearn splitter.** They lack purge / embargo and don't satisfy the `Splitter` ABC; `MultiPathEngine` won't accept them.
- **Do not assume `n_splits == n_paths`.** With `n_test_groups > 1` they differ. The number of *runs* you pay for is `n_splits`; the number of *paths* you get is `n_paths`. Downstream metric counts (DSR effective-K, per-path stats) use `n_paths`.
- **Do not pass `n_splits=1`** to `CombinatorialPurgedCV` — it raises. Use `WalkForward` for the single-split case.
- **Do not refer to `splitter.n_splits` as an attribute** — it's a method (`splitter.n_splits()`). The attribute is `splitter.n_groups` (storing the constructor argument). Confusing the two raises `TypeError` or silently returns a bound method.
- **Do not shuffle** the iterator order returned by `split` and assume `assemble_paths` still works. The internal mapping from splits to paths assumes the lexicographic combination order from `itertools.combinations`.
- **Do not call `splitter.split(dates)` twice** thinking the splits will differ — they're deterministic for the same `dates`.
- **Do not set `embargo_pct` close to 1.0** — accepted (must be `< 1.0`) but train shrinks to zero or near-zero per split. The engine will not warn.
- **Do not call `assemble_paths` yourself** in user code. `MultiPathEngine.run` calls it internally. Passing the wrong number of split-return Series raises `ValueError`; passing returns indexed outside the test segments produces NaN-laden paths.
- **Do not assume embargo replaces purge on the trailing side.** Purge wraps the test run on both sides; embargo adds *more* buffer after. Trailing total = `purge_bars + embargo_bars`. Surface both numbers in the announcement when either is non-zero.
- **Do not interpret `n_paths` as "k-fold validation".** Paths are independent reconstructions of the full timeline, each a valid equity curve. CPCV is not k-fold.
- **Do not switch from WF to CPCV (or vice versa) mid-experiment** without re-running everything. The two produce different distributions of metrics, and DSR (`09-selection.md`) requires that all trials in a family come from a comparable evaluation procedure.
- **Do not change `n_splits` / `n_test_groups` between trials in the same family.** Each combination is a distinct experiment; metrics across them aren't directly comparable and inflate K artificially.

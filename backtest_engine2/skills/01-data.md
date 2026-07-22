# Skill 01 — Data

> **Read `PROTOCOL.md` and `skills/00-overview.md` first.** This file assumes you have. Cell-loop and announcement rules in PROTOCOL apply.

---

## When to load

- **Stage 1** of every backtest (setup + data load).
- Any later cell that adds a new field, changes a `FieldSpec`, modifies the universe, or changes the lag of an existing field.

---

## What you'll find here

1. The `DataPanel` constructor — every argument, every default, every validation rule.
2. `FieldSpec` — the per-field PIT + missing-data policy.
3. `DataView` — exactly what a strategy sees at time `t`, attribute by attribute.
4. PIT enforcement (lag mode) — how lag is applied and when you'd set it.
5. Universe semantics — implicit (NaN-based) vs explicit (bool DataFrame).
6. Outlier check — what triggers it, how to interpret, how to silence.
7. Common mismatches and the questions to ask the user.
8. The minimal valid data-load cell.
9. Validation checklist.
10. Concrete anti-patterns.

---

## 1. `DataPanel` — the constructor

```python
DataPanel(
    prices: pd.DataFrame,                              # required
    volume: pd.DataFrame | None = None,
    features: dict[str, pd.DataFrame] | None = None,
    universe: pd.DataFrame | None = None,              # bool, time-varying eligibility
    specs: dict[str, FieldSpec] | None = None,
    check_outliers: bool = True,
    outlier_mad_threshold: float = 10.0,
)
```
`source: backtest/data/panel.py:73–82`

### 1.1 What each argument is for

| Argument | Type | Meaning |
|---|---|---|
| `prices` | `DataFrame[date × asset]` | Adjusted prices. **Required.** |
| `volume` | `DataFrame[date × asset]` | Optional. **Required if you plan to use `LiquidityCap` or `MarketImpact`** — both rely on ADV. |
| `features` | `dict[str, DataFrame]` | Optional named panels (each `DataFrame[date × asset]`). Accessed in strategies via `data.feature(name)`. |
| `universe` | `DataFrame[date × asset]` of bool | Optional. Per-bar asset eligibility. If omitted, eligibility falls back to "price is not NaN at `t`". |
| `specs` | `dict[str, FieldSpec]` | Per-field PIT + missing-data policy. Keyed by field name (`"prices"`, `"volume"`, or any feature name). Missing keys get `FieldSpec()` defaults. |
| `check_outliers` | `bool` (default `True`) | Run a MAD-based outlier scan at construction; warn if any bar exceeds the threshold. |
| `outlier_mad_threshold` | `float` (default `10.0`) | Threshold for the MAD scan. |

### 1.2 Validation rules (all enforced at construction)

`source: backtest/data/panel.py:103–122`

- `prices` must be a `DataFrame`. Else `TypeError`.
- `prices.index` must be a `DatetimeIndex`. Else `TypeError`.
- `prices.index` must be **monotonically increasing**. Else `ValueError`.
- `prices.index` must have **no duplicates**. Else `ValueError`.
- `volume`, `universe`, and every `features[k]` must have:
  - **identical index** to `prices`, and
  - **identical columns** to `prices`.
  Otherwise `ValueError`.

If the user's data has, e.g., volume for fewer assets than prices: that is a mismatch — surface it. Do not silently `reindex` without asking.

### 1.3 What the constructor does (one-time work)

1. Validates inputs (above).
2. For every field, applies its `FieldSpec`:
   - `shift(spec.lag)` if `lag > 0`,
   - then either `ffill(limit=max_staleness)`, `fillna(fill_value)`, or no-op (`"drop"`).
3. Stores universe as `bool` DataFrame.
4. Caches `self.dates` (= `prices.index`) and `self.assets_all` (= `prices.columns`).
5. If `check_outliers=True` **and** `len(prices) >= 30`, runs `outlier_report` and warns with the top 5 flagged bars.

`source: backtest/data/panel.py:83–101, 124–135, 172–192`

After construction, `as_of(t)` is a cheap slice — no further data manipulation happens at strategy time.

### 1.4 Field introspection (`has_field`, `available_fields`)

Two small public methods for asking the panel what it exposes:

```python
panel.has_field("prices")           # True
panel.has_field("volume")           # True iff a volume DataFrame was supplied
panel.has_field("eps")              # True iff features={"eps": ...} was passed
panel.available_fields()            # ["prices", "volume", "eps", ...] (sorted features)
```

`source: backtest/data/panel.py` (methods `has_field` and `available_fields` on `DataPanel`).

You usually don't call these directly — they're consumed automatically by `Engine.__init__` to validate `strategy.required_data()` against the panel (`02-strategy.md §8`). They're also useful when debugging a "strategy required X but panel doesn't have it" error: print `panel.available_fields()` to confirm what's actually registered.

---

## 2. `FieldSpec` — per-field policy

```python
@dataclass(frozen=True)
class FieldSpec:
    lag: int = 0
    missing: str = "drop"            # "drop" | "ffill" | "value"
    max_staleness: int | None = None # None = unlimited (only used when missing=="ffill")
    fill_value: float = 0.0          # only used when missing=="value"
```
`source: backtest/data/panel.py:13–25`

| Field | Default | When you change it |
|---|---|---|
| `lag` | `0` | Set `>0` for any field whose value at row `t` reflects information **published after `t`** in real life. Earnings, ratings, fundamentals: typically 1–5 bars. Adjusted prices: usually `0`. |
| `missing` | `"drop"` | `"ffill"` for slow-moving fields where staleness is OK (fundamentals, ratings). `"value"` for indicator-style fields where missing means "off" (use `fill_value=0.0`). |
| `max_staleness` | `None` | Cap on `ffill` propagation. E.g., `max_staleness=21` lets a quarterly rating ffill for ~1 month then go back to NaN. |
| `fill_value` | `0.0` | Constant for `missing="value"`. |

Unknown `missing` value → `ValueError` at construction. `source: backtest/data/panel.py:133–134`

---

## 3. `DataView` — what strategies see

```python
class DataView:
    __slots__ = ("_panel", "t")

    @property
    def prices(self) -> pd.DataFrame: ...      # all prices up to and including t
    @property
    def volume(self) -> pd.DataFrame | None: ...
    @property
    def assets(self) -> pd.Index: ...          # eligible asset ids at t
    def feature(self, name: str) -> pd.DataFrame: ...
    def has_feature(self, name: str) -> bool: ...

    # also: view.t  (the snap-to-prior Timestamp)
```
`source: backtest/data/panel.py:28–62`

Exact slicing semantics (verbatim from source):

- **`view.prices`** → `panel._prices.loc[:view.t]` — DatetimeIndex × asset, **inclusive of `view.t`**. `source: backtest/data/panel.py:38–39`
- **`view.volume`** → same, or `None` if no volume was supplied. `source: backtest/data/panel.py:42–44`
- **`view.assets`**: `source: backtest/data/panel.py:46–56`
  - If `universe is None`: returns `prices.loc[t].index[notna()]` — the assets whose price at `t` is not NaN.
  - Else: looks up the universe row at `t`; if `t` isn't in the universe index, falls back to the **last available row strictly before `t`**.
- **`view.feature(name)`** → `panel._features[name].loc[:view.t]`. Raises `KeyError` if `name` not registered. `source: backtest/data/panel.py:58–59`
- **`view.has_feature(name)`** → bool guard before `feature(name)`. `source: backtest/data/panel.py:61–62`

`view.t` is **the timestamp the panel actually has data for**, not necessarily the `t` you passed to `as_of`. See §4.2.

---

## 4. PIT enforcement (lag mode, v1)

The engine implements only **lag mode** in v1: every field is shifted forward by `spec.lag` bars at construction. Knowledge-time mode (vintaged data with restatements) is **deferred** — see `prd.md §19`.

### 4.1 What lag does

`out = df.shift(spec.lag)` `source: backtest/data/panel.py:127–128`

Concretely: a value originally at row `t` is moved to row `t + lag`. The first `lag` rows become NaN. So when a strategy asks for `view.feature("eps")` at `t`, it sees the EPS that would have been published at most `lag` bars ago, never anything more recent.

### 4.2 `as_of(t)` — snap to prior bar

```python
def as_of(self, t) -> DataView:
    t = pd.Timestamp(t)
    if t < self.dates[0]:
        raise ValueError(...)
    if t not in self.dates:
        pos = self.dates.searchsorted(t, side="right") - 1
        t = self.dates[pos]
    return DataView(self, t)
```
`source: backtest/data/panel.py:137–144`

Calling `panel.as_of(t)` with a `t` that isn't in `panel.dates` snaps **to the latest date < `t`** (via `searchsorted(side="right") - 1`). This is silent — no warning. If you need to detect non-trading dates, do it explicitly before calling.

`t` strictly before `panel.dates[0]` raises `ValueError`.

### 4.3 When to set `lag > 0`

| Field type | Typical lag (bars) | Why |
|---|---|---|
| Adjusted prices | `0` | Closing prices are knowable at the close of `t`. |
| Volume | `0` | Same. |
| Earnings, ratings, fundamentals | `1`+ | Reported on day `t` is typically only actionable at the next bar. |
| Analyst signals, broker outputs | `1`+ | Same. |
| Anything user "computes from external sources" | **ask the user** | If the spec says nothing about lag for a feature that obviously needs one, surface it as a mismatch. |

---

## 5. Universe semantics

| User input | What `view.assets` returns |
|---|---|
| `universe=None` | `prices.loc[t].index[notna()]`. Assets with NaN prices at `t` are excluded. |
| `universe=DataFrame[date×asset, bool]` | The True-cells of the universe row at `t` (or the last row before `t` if `t` isn't in the universe index). |

`source: backtest/data/panel.py:46–56`

### Survivorship — what the engine does and doesn't do

The engine **does not** dedup, drop, or align asset columns. It exposes whatever the user provided. The way to avoid survivorship bias is one of:

- Provide an explicit `universe` DataFrame with True/False per `(date, asset)`.
- Or include delisted assets in `prices` with NaN values after their delist date — `view.assets` will then exclude them automatically.

If the user provides "current S&P 500 members backtested 20 years": that is a survivorship bias. **Surface it.** The engine cannot save them from this.

---

## 6. Outlier check

Run automatically at construction when `check_outliers=True` (default) and `len(prices) >= 30` (`_OUTLIER_MIN_BARS`). `source: backtest/data/panel.py:10, 100–101, 172–174`

### Formula

For each `(date, asset)`:
```
return     = pct_change(price)
median     = median over time, per asset, of return
MAD        = median over time, per asset, of |return - median|
mad_score  = |return - median| / MAD       (NaN if MAD == 0)
flagged    = mad_score > outlier_mad_threshold     (default 10.0)
```
`source: backtest/data/panel.py:146–170`

Output is a `DataFrame[date, asset, return, mad_score]` sorted by `mad_score` descending. The warning prints the top 5.

### What to do when it warns

1. Read the warning. It names the bars and assets.
2. State to the user: *"DataPanel flagged N bars as outliers. Top examples are X. Are these real moves (e.g. corporate actions, vol events) or bad ticks?"*
3. Wait for the user's call:
   - Real moves → keep going.
   - Bad ticks → user fixes the data and reruns the cell. Do not "fix" by passing `check_outliers=False`.
4. Only silence with `check_outliers=False` if the user explicitly asks to.

You can also call `panel.outlier_report(threshold=...)` to inspect manually — same formula, returns the full sorted DataFrame. `source: backtest/data/panel.py:146–170`

---

## 7. Common mismatches — surface these by default

Use the question shape from `PROTOCOL.md §3`. Engine-faithful default first.

| Spec says | Engine has | Question to ask |
|---|---|---|
| "Use vintaged / knowledge-time data" | Only lag mode in v1. Knowledge-time deferred (`prd.md §4.2, §19`). | Lag mode with a `lag` you and the user agree on, **or** drop the field. |
| "Intraday" data | Architecture compatible but not exercised (`prd.md §2`). | Daily proxy, **or** flag as out-of-scope. |
| Sector / region concentration limits | **Not implemented** (`prd.md §2`, placeholder only). | Pass sector as a feature and apply the limit yourself in `apply_risk`, **or** drop the constraint. |
| "Drop assets that are NaN at the start" | Engine does not "clean" data. | Build an explicit `universe` DataFrame, **or** keep the NaNs (they're handled per-bar by `view.assets`). |
| "Use the current S&P 500 going back 20 years" | Survivorship bias — engine will not catch. | Use a point-in-time membership universe, **or** acknowledge the bias in the registry. |
| "Discrete shares only" | v1 uses fractional weights (`prd.md §2`). | Approximate with fractional weights, **or** flag as out-of-scope. |
| "Multiple data sources, different indexes" | Constructor requires identical index/columns across all panels (`backtest/data/panel.py:119–122`). | Reindex before constructing — but ask the user how to handle date misalignment (`reindex` + `ffill`? union of dates?), don't pick silently. |

---

## 8. The minimal valid data-load cell

Use this as the **shape** of Cell 1, not as a copy-paste. Adapt paths and field names to the user's spec.

```python
import pandas as pd
from backtest.data import DataPanel, FieldSpec

prices = pd.read_parquet("path/to/prices.parquet")
volume = pd.read_parquet("path/to/volume.parquet")  # required if costs/liquidity use ADV

panel = DataPanel(
    prices,
    volume=volume,
    # Add features=... and a matching specs={...} entry per field if the spec needs them.
    # Add universe=... if the spec requires explicit eligibility (survivorship-aware).
    check_outliers=True,
)

print(f"Bars: {len(panel.dates)}")
print(f"Assets: {len(panel.assets_all)}")
print(f"Range: {panel.dates[0].date()} → {panel.dates[-1].date()}")
print(f"NaN fraction in prices: {prices.isna().mean().mean():.3%}")
prices.tail(3)
```

Output the user must paste back so you can validate (§9). End the cell with a small visible artifact (`tail(3)`, `head()`, `print`).

---

## 9. Validation checklist (after the cell runs)

When the user pastes the output, check **all** of these. State which passed and which failed in your reply.

- [ ] **Bars count** is plausible for the date range (≈252 trading days/year for daily US equities).
- [ ] **Asset count** matches what the spec implied.
- [ ] **Date range** matches the spec window (or a superset of it).
- [ ] **NaN fraction** is small. If >5%, ask the user whether that's expected (early-history pre-listing, late-history delisting, or a data quality problem).
- [ ] **No outlier warning** — or, if there is one, you've raised it explicitly with the user.
- [ ] **No silent index/column mismatch error** raised.

If any fails: state the failure. Propose either a fix-cell or a panel re-construction with different `specs`. Do not proceed to Stage 2 without resolution.

---

## 10. What NOT to do

- **Do not call `DataPanel.as_of(t)` from the strategy.** That's the engine's job. The strategy receives the resulting `DataView`.
- **Do not access `panel._prices`, `panel._volume`, `panel._features`, or `panel._universe`** from any user code. They are private (leading underscore). The notebook example uses `panel._prices.pct_change()` once for MC fitting — that is a Monte-Carlo-only escape hatch (`10-simulation.md`), not a pattern for strategies.
- **Do not "clean" or "reindex" the user's prices** before passing to `DataPanel` without asking. If the user's data has misaligned indexes, surface it as a mismatch (§7) — do not silently align.
- **Do not pass `check_outliers=False`** to silence a warning. Surface the warning to the user first.
- **Do not invent a `FieldSpec.lag`** for a feature whose lag the user hasn't specified. Ask.
- **Do not introduce knowledge-time mode** by hand-vintaging features. Lag mode is the only supported PIT mechanism in v1 (`prd.md §4.2`).
- **Do not assume `universe is None` is fine.** If the spec mentions S&P membership, sector rotation, listing/delisting, or any time-varying eligibility, that requires an explicit `universe`. Surface it.

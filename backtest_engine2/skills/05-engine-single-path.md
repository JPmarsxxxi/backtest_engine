# Skill 05 — Single-Path Engine

> **Read `PROTOCOL.md` and `skills/00-overview.md` first.** Cell-loop and announcement rules in PROTOCOL apply.

> Stage 3 (costs + liquidity) must already be done. Costs are mandatory before any run.

---

## When to load

- **Stage 4** of every backtest: running one single-path backtest with `Engine.run()` and inspecting `BacktestResult`.
- Any cell that calls `Engine(...)` directly. (CPCV / multi-path is a different orchestrator — see `07-multipath.md`.)
- Any cell that reads `BacktestResult` attributes (`equity_curve`, `returns`, `weights`, etc.) — to confirm what each one actually contains.

---

## What you'll find here

1. The `Engine` constructor — every argument, every validation.
2. `Engine.run` — every argument and how `start` / `end` / `train_dates` / `test_dates` interact.
3. The per-bar loop, in order — exactly what the engine does on every bar and on rebalance bars.
4. `BacktestResult` — every field with its dtype and meaning.
5. `BacktestResult.pnl` and `.gross_pnl` properties.
6. `BacktestResult.metadata` — the six keys, verbatim.
7. `BacktestResult.summary()` — what it prints, what it plots, what it returns.
8. Common mismatches.
9. Minimal valid cell.
10. Validation checklist.
11. Anti-patterns.

---

## 1. `Engine` — the constructor

```python
Engine(
    data: DataPanel,
    strategy: Strategy,
    costs: CostModel | None = None,
    liquidity: LiquidityCap | None = None,
    initial_capital: float = 1_000_000.0,
)
```
`source: backtest/engine/engine.py:118–139`

| Arg | Required? | Notes |
|---|---|---|
| `data` | Yes | The `DataPanel` from Stage 1. Engine holds an immutable reference. |
| `strategy` | Yes | The `Strategy` instance from Stage 2. |
| `costs` | Per PROTOCOL: **yes** | A `CostModel` (usually `CompositeCostModel`). Engine accepts `None`, but `PROTOCOL.md §6` forbids it. |
| `liquidity` | Optional | A `LiquidityCap`. If supplied, the panel **must** have volume — see §1.1. |
| `initial_capital` | No | Starting equity in dollars. Must be `> 0`. |

### 1.1 Construction-time validation

The engine raises `ValueError` at construction (before any work) in three cases. `source: backtest/engine/engine.py:126–139`

1. **`initial_capital <= 0`** — `"initial_capital must be positive"`.
2. **Strategy declares a `required_data` field the panel doesn't have** — message lists the missing fields and what the panel does have. See `02-strategy.md §8`.
3. **`liquidity is not None` and panel has no volume** — message: `"LiquidityCap was supplied but the panel has no volume. Without volume the cap silently does nothing. Either pass volume to DataPanel or drop the LiquidityCap."` See `04-costs.md §8`.

These are construction-time, not run-time. So a clean `Engine(...)` line means every basic mismatch has been caught.

---

## 2. `Engine.run` — the arguments

```python
def run(
    start: pd.Timestamp | None = None,
    end:   pd.Timestamp | None = None,
    train_dates: pd.DatetimeIndex | None = None,
    test_dates:  pd.DatetimeIndex | None = None,
) -> BacktestResult
```
`source: backtest/engine/engine.py:134–140`

### 2.1 Date resolution (`_resolve_dates`)

```python
if test_dates is not None:
    return pd.DatetimeIndex(test_dates).sort_values()
dates = self.data.dates
if start is not None:
    dates = dates[dates >= pd.Timestamp(start)]
if end is not None:
    dates = dates[dates <= pd.Timestamp(end)]
return dates
```
`source: backtest/engine/engine.py:284–292`

Precedence:

- **If `test_dates` is supplied, it wins.** `start` and `end` are ignored. The backtest runs over `test_dates` sorted ascending.
- **Otherwise**, the engine starts from `panel.dates`, then filters by `start` (inclusive lower bound) and `end` (inclusive upper bound).
- After filtering, if `len(dates) == 0`, the engine raises `ValueError("no dates in range")`. `source: backtest/engine/engine.py:142–143`

### 2.2 The `train_dates` argument

If `train_dates` is supplied and non-empty, the engine calls
```python
strategy.fit(self.data.as_of(train_dates[-1]))
```
exactly once, before the per-bar loop. `source: backtest/engine/engine.py:145–147`

The contents of `train_dates` beyond `train_dates[-1]` are not used by `Engine` — the only purpose of the index is to set the `as_of` cutoff. (For per-fold fit/test splits, use `MultiPathEngine` — `07-multipath.md`.)

If `train_dates` is `None` or empty, `fit` is not called. The `metadata["fit_called"]` key reflects this.

### 2.3 Typical usage

```python
result = Engine(panel, strat, costs=costs, liquidity=liq).run(
    start=panel.dates[252],     # 1-year warmup
)
```

This is the canonical Stage 4 call. Skip the first `lookback` (e.g. 252) bars so the strategy has a full history to compute features at the first decision point. The engine does **not** automatically apply a warmup — if you don't pass `start`, the loop runs from `panel.dates[0]`, and a strategy that needs `>0` history may produce all-zero weights for the first bars (which is fine numerically but pollutes the metrics).

---

## 3. The per-bar loop — exactly what happens

`source: backtest/engine/engine.py:180–262`

For each bar `i` over the resolved `dates`:

1. **Mark to market.** `positions = positions * ratios[i-1]`, where `ratios = prices[1:] / prices[:-1]`, NaN→1.0 (asset price-of-record stays put). `source: backtest/engine/engine.py:183–184, 294–298`
2. **Compute `equity_pre`** = `positions.sum() + cash`. Update `peak` and `dd`.
3. **Build `view = data.as_of(t)`** if costs are present OR this is a rebalance bar; else skip.
4. **Charge `holding_cost`** (e.g. short borrow) — `cash -= hc`. Every bar.
5. **If this is a rebalance bar**:
   - Build `state = {drawdown, equity, positions, cash, peak}` (per `02-strategy.md §4`).
   - Call `proposed = strategy.generate_weights(view, t)`, reindex to all assets, fill NaN with 0, coerce to float.
   - **Eligibility filter**: zero out any asset not in `view.assets`.
   - Call `final_w = strategy.apply_risk(proposed, state, view)`, reindex + fill + coerce again.
   - `target_dollars = final_w * equity`.
   - `proposed_trades = target_dollars - positions`.
   - Zero out trades for any asset with non-finite price on this bar (`price_invalid`).
   - If `liquidity` is set: `(executed, unfilled) = liquidity.apply(...)`. Else `executed = proposed_trades`, `unfilled = 0`.
   - Charge `trade_cost(executed, view)` — `tc`.
   - `positions += executed`. `cash -= executed.sum() + tc`.
6. **Record this bar**: `eq_curve[i] = positions.sum() + cash`, `weights[i] = positions / eq_curve[i]` (only if equity != 0), `returns[i] = (eq[i] - eq[i-1]) / eq[i-1]` (for `i > 0`).

Important consequences for the LLM:

- **Weights in the result are *realized* (positions / equity), not the strategy's proposed weights.** They reflect everything the engine did — risk overlay, liquidity throttling, mark-to-market drift between rebalances.
- **`returns[0]` is `NaN`** (not 0) because there's no previous bar to diff against — this matches `pandas.Series.pct_change()` convention. `compute_metrics` drops the NaN via `dropna()` before measuring, so `MetricsReport.n_obs == len(returns) - 1`. `source: backtest/engine/engine.py:181, 277–279; backtest/metrics/report.py:138`
- **Between rebalance bars, weights drift** with prices — `weights[i]` will not exactly match `weights[rebal_bar]` unless prices haven't moved.
- **Cash from shorts** adds to `cash`; the engine accounts for this correctly via `cash -= executed.sum()` (executed is signed, shorts are negative trades on the position side and positive cash).

---

## 4. `BacktestResult` — the fields

```python
@dataclass
class BacktestResult:
    equity_curve: pd.Series       # equity over time
    returns:      pd.Series       # bar-to-bar return
    weights:      pd.DataFrame    # realized weights (positions / equity_now)
    positions:    pd.DataFrame    # dollar positions per asset
    trades:       pd.DataFrame    # executed trades in dollars per asset
    costs:        pd.Series       # total cost per bar (holding + trade)
    unfilled:     pd.DataFrame    # liquidity-throttled unfilled portion
    metadata:     dict            # bookkeeping; see §6
```
`source: backtest/engine/engine.py:15–24`

| Field | Type | Shape | Meaning |
|---|---|---|---|
| `equity_curve` | `pd.Series` | `(n_bars,)` | Total equity at each bar (positions + cash), post-trade. Indexed by date. |
| `returns` | `pd.Series` | `(n_bars,)` | `(eq[i] - eq[i-1]) / eq[i-1]`. `returns[0] = NaN` (no prior bar to diff). |
| `weights` | `pd.DataFrame` | `(n_bars, n_assets)` | `positions[i] / equity[i]` per asset per bar. Realized, post-everything. |
| `positions` | `pd.DataFrame` | `(n_bars, n_assets)` | Dollar position per asset per bar, post-trade. |
| `trades` | `pd.DataFrame` | `(n_bars, n_assets)` | **Executed** trades (after liquidity throttling). 0 on non-rebalance bars. |
| `costs` | `pd.Series` | `(n_bars,)` | `holding_cost + trade_cost` per bar. Always ≥ 0 for normal models (negative possible only for negative-bps tricks like rebate — see `04-costs.md §6`). |
| `unfilled` | `pd.DataFrame` | `(n_bars, n_assets)` | `proposed_trades - executed` from the liquidity step. 0 if no `LiquidityCap` or no throttling occurred. |
| `metadata` | `dict` | — | See §6. |

All Series have `.name` set (e.g. `equity_curve.name = "equity"`). All DataFrames are column-indexed by `data.assets_all`, **not** the per-bar eligible subset.

---

## 5. Properties — `pnl` and `gross_pnl`

```python
@property
def pnl(self) -> pd.Series:        # net dollar PnL per bar, after costs
    return self.equity_curve.diff().fillna(0.0).rename("pnl")

@property
def gross_pnl(self) -> pd.Series:  # pre-cost dollar PnL per bar
    return (self.pnl + self.costs).rename("gross_pnl")
```
`source: backtest/engine/engine.py:26–34`

- `pnl` is the first difference of `equity_curve`, with the leading NaN replaced by 0. Sum over the full series equals `equity_curve.iloc[-1] - initial_capital`.
- `gross_pnl = pnl + costs`. By construction, `gross_pnl - pnl - costs == 0` exactly (modulo floating point). Tested at `engine_test.py` (gross-pnl test).

---

## 6. `BacktestResult.metadata` — the six keys

Verbatim from `Engine.run`:
```python
meta = {
    "start":          dates[0],
    "end":            dates[-1],
    "initial_capital": self.initial_capital,
    "n_bars":         n,
    "n_rebalances":   int(rebalance_mask.sum()),
    "fit_called":     train_dates is not None and len(train_dates) > 0,
}
```
`source: backtest/engine/engine.py:264–271`

That's the complete set. Any other key you reach for doesn't exist. Use `.get("key", default)` if you want safe access, but the canonical six are guaranteed.

---

## 7. `BacktestResult.summary()` — what it does

```python
result.summary(ann_factor=252, ci_level=0.95, rolling_window=60, figsize=(13, 8))
```
`source: backtest/engine/engine.py:36–107`

Three things happen:

1. **Computes the full `MetricsReport`** via `compute_metrics(returns, equity_curve, costs=costs, ann_factor=ann_factor, ci_level=ci_level)`. See `08-metrics.md`.
2. **Displays the report** — uses `IPython.display.display(rep)` if available (HTML in Jupyter), else `print(rep)`.
3. **Plots a 2×2 dashboard**:
   - Top-left: equity curve with horizontal line at `initial_capital`.
   - Top-right: drawdown (filled area, **negative** sign — see `03-risk.md §5`), titled with max DD.
   - Bottom-left: rolling Sharpe over `rolling_window` bars (annualized), with horizontal lines at 0 and the full-sample Sharpe.
   - Bottom-right: asymptotic SR distribution via `plot_sharpe_distribution`.
4. **Returns the `MetricsReport`**.

Requirements:

- `matplotlib` must be installed. If not, raises `ImportError` with install instructions. `source: backtest/engine/engine.py:50–56`
- In a non-Jupyter environment, the metrics display falls back to `print`.

In a notebook cell, the typical usage is `report = result.summary()` — you get the plots inline and the report object back.

---

## 8. Common mismatches — surface these by default

Use the question shape from `PROTOCOL.md §3`. Engine-faithful default first.

| Spec says | Engine has | Question to ask |
|---|---|---|
| "Train on 2015–2019, test on 2020–2023" | Pass `train_dates=panel.dates[(panel.dates >= '2015-01-01') & (panel.dates <= '2019-12-31')]` and either `start='2020-01-01'`, `end='2023-12-31'` or `test_dates=...`. | Confirm whether the spec wants `fit()` called on train then run on test (use `train_dates` + `start`/`end`), or full walk-forward with rolling fits (use `MultiPathEngine` with `WalkForward` splitter — `06-splitters.md`). |
| "Parameter sweep over lookback ∈ {30, 60, 90, 120}" | Multiple `Engine.run()` calls, one per strategy instance. | One cell per parameter is too many; one cell with a loop is OK but each trial should go to the trial registry (`11-registry.md`) for DSR deflation. Surface the trade-off. |
| "Backtest starts 2015-01-01" but panel starts 2010-01-01 | `start=pd.Timestamp("2015-01-01")` — the engine drops bars before. | Confirm: do they want a 5-year warmup (then run from 2015) or do they want the engine to also drop the 2010–2014 data from the panel (it won't — the panel keeps everything; only the run window is filtered)? |
| "No warmup" | Engine has no built-in warmup. Strategy gets `view.prices.loc[:t]` from `panel.dates[0]`. | Confirm: strategies needing lookback will produce zero weights early. Acceptable or do we add an explicit `start=` cutoff? |
| Returns should be log returns | Engine returns are simple (`(eq[i]-eq[i-1])/eq[i-1]`). | Convert with `np.log1p(result.returns)` post-hoc, **or** accept simple returns (which is the convention in `08-metrics.md`). |
| Account for the warmup in metrics | Metrics use the full `result.returns`, including warmup zeros if `start` wasn't set. | Always pass an explicit `start=` that respects the strategy's lookback. |
| "Show me the proposed weights, not realized" | `BacktestResult.weights` is realized (positions/equity). The proposed weights are not retained. | Custom strategy that stores its own proposed weights on `self`, **or** accept realized weights. |
| Compounding turned off (always size against initial capital) | Engine compounds — every rebalance sizes against current equity. | Custom risk override that rescales weights by `initial_capital / equity`, **or** accept compounding (the standard). |
| Multi-currency / FX overlay | Single-currency only. | Out of scope. |
| Intra-day fills / slippage from VWAP | Single price per bar (the panel's price at `t`). | Out of scope (`prd.md §2`). |
| Per-fold fit and walk-forward retraining | `Engine.run(train_dates=...)` fits once. For per-fold fits use `MultiPathEngine` (`07-multipath.md`). | Switch to `MultiPathEngine` with `WalkForward`. |

---

## 9. Minimal valid cell — Stage 4

The shape of Stage 4. Adapt the warmup, start date, and capital to the user's spec.

```python
from backtest.engine import Engine

WARMUP_BARS = 252   # 1y for strategies with a 60-252 bar lookback; adjust to spec

engine = Engine(
    panel,
    strat,
    costs=costs,                                 # from Stage 3 — REQUIRED
    liquidity=liq,                               # from Stage 3
    initial_capital=1_000_000,
)
result = engine.run(start=panel.dates[WARMUP_BARS])

# Visible artifact for validation (PROTOCOL §4 / §5).
print(f"Bars:          {len(result.equity_curve)}")
print(f"Rebalances:    {result.metadata['n_rebalances']}")
print(f"Date range:    {result.metadata['start'].date()} → {result.metadata['end'].date()}")
print(f"Initial:       ${result.metadata['initial_capital']:>14,.0f}")
print(f"Final equity:  ${result.equity_curve.iloc[-1]:>14,.0f}")
print(f"Total return:  {(result.equity_curve.iloc[-1] / result.metadata['initial_capital'] - 1):+.2%}")
print(f"Total costs:   ${result.costs.sum():>14,.0f}")
print(f"Max position weight:  {result.weights.abs().max().max():.2%}")
print(f"Max gross exposure:   {result.weights.abs().sum(axis=1).max():.2%}")
```

Do **not** call `result.summary()` in this same cell. `.summary()` does plotting and metrics; that's Stage 7 (`08-metrics.md`). Stage 4 only verifies the engine ran cleanly.

---

## 10. Validation checklist (after the cell runs)

After the user pastes the output:

- [ ] **`Bars`** equals `len(panel.dates) - WARMUP_BARS` (or whatever the `start=` implies).
- [ ] **`Rebalances`** matches the strategy's `rebalance_frequency`. Daily ≈ Bars, monthly ≈ Bars/21, etc.
- [ ] **Final equity** is positive and finite. Negative or `NaN` → catastrophic bug; do not proceed.
- [ ] **Total return** is in a plausible range. Wildly negative (-90%) with no obvious cost reason → check `apply_risk` and `LiquidityCap`. Wildly positive (>10000%) → very likely a lookahead bug in the strategy.
- [ ] **Total costs** is positive (or zero if no trading). Negative only if `ShortBorrow` was given a negative `annual_bps` for rebate modelling (`04-costs.md §6`).
- [ ] **Max position weight** is ≤ `RiskConfig.max_position + 1e-9`. Violations point to a bug in `apply_risk`.
- [ ] **Max gross exposure** is ≤ `RiskConfig.max_gross` (or `max_leverage`) + 1e-9.
- [ ] No `ValueError` from the engine: that means initial capital, required_data, and liquidity-needs-volume all passed.

If any fails: state which, do not proceed to Stage 5/6/7. Propose a fix-cell (modify Stage 1, 2, or 3 — never patch the engine).

---

## 11. What NOT to do

- **Do not pass `costs=None`** unless the user has explicitly asked for a gross comparison cell (`PROTOCOL.md §6`).
- **Do not call `Engine.run()` multiple times with different parameters** in the same cell. Each trial is a separate decision the user must approve; multiple trials also need DSR deflation (`09-selection.md`). Surface this as a mismatch before sweeping.
- **Do not assume `result.weights` are the strategy's proposed weights.** They're realized. To inspect proposed weights, you have to instrument the strategy itself.
- **Do not skip the explicit `start=`** if the strategy has any lookback. Without it, the early bars produce zero (or NaN) weights and pollute every downstream metric.
- **Do not pass `test_dates` and `start` together** thinking they combine. `test_dates` wins; `start` is ignored. If you want both behaviors, intersect them before passing.
- **Do not assume metadata has extra keys** beyond the six in §6. `metadata.get("foo")` returns `None`, not an error — be defensive when reading.
- **Do not call `result.summary()` inside Stage 4.** That's Stage 7's job (`08-metrics.md`). Stage 4 just checks the engine ran; Stage 7 does the metrics analysis.
- **Do not recompute returns from `equity_curve.pct_change()`** — the engine already provides `result.returns` with `returns[0] = NaN` (same as `pct_change`) and a guard against zero prior equity that `pct_change` lacks. Use `result.returns` directly. `source: backtest/engine/engine.py:277–278`
- **Do not reach into `engine.data._prices` or any panel private** from user code. The engine itself does that for performance; user code uses `view.prices` (`01-data.md`).
- **Do not assume `BacktestResult.trades` is the strategy's intent.** Trades are *executed* dollars, post-throttling. The unfilled portion is in `result.unfilled`.
- **Do not silently regenerate a `BacktestResult`** by re-running with different costs or risk if you want to keep both for comparison — that's a deliberate two-cell decision and each result is a separate trial.

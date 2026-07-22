# Skill 08 — Metrics

> **Read `PROTOCOL.md` and `skills/00-overview.md` first.** Cell-loop and announcement rules in PROTOCOL apply.

> Stage 4 (single-path engine) or Stage 6 (multi-path) must have produced returns and an equity curve. This skill is about measuring them. DSR / multiple-testing correction is Stage 9 (`skills/09-selection.md`); this skill stops at the per-trial report.

---

## When to load

- **Stage 7**: building `MetricsReport` via `compute_metrics(returns, equity)` and inspecting the 22 fields.
- Any cell that calls `result.summary()` and needs to explain what each printed field means.
- Any cell that uses one of the scalar metrics (`sharpe_ratio`, `psr`, `max_drawdown`, etc.) directly.

---

## What you'll find here

1. The 21 fields of `MetricsReport`.
2. `compute_metrics` — args, NaN handling, what's optional.
3. Sharpe and the asymptotic SR distribution (paper Eq. 1–2).
4. PSR and MinTRL (paper Eq. 3–4) — what "this Sharpe is real" means quantitatively.
5. Drawdown and time-under-water (path-based, numba-jitted).
6. Secondary metrics: Sortino, Calmar, VaR, cVaR, modified Sharpe.
7. Annualization — what `ann_factor` does and when to override the default.
8. Display: `__repr__`, `_repr_html_`, `plot_sharpe_distribution`, `psr_at`, `min_trl_at`.
9. Common mismatches.
10. Minimal valid cell.
11. Validation checklist.
12. Anti-patterns.

---

## 1. `MetricsReport` — the 28 fields

```python
@dataclass
class MetricsReport:
    total_return:        float
    ann_return:          float   # CAGR
    ann_vol:             float
    hit_rate:            float   # fraction of bars with strictly positive return
    total_pnl:           float   # dollars
    total_costs:         float   # dollars; NaN if costs not provided
    gross_pnl:           float   # pnl + costs; NaN if costs not provided
    turnover:            float   # total two-way turns = sum(|trades|)/mean(equity); NaN if trades not provided
    margin:              float   # net PnL per dollar traded = total_pnl/sum(|trades|); NaN if trades not provided
    sharpe:              float   # annualized
    sharpe_std:          float   # annualized SE of the Sharpe estimator
    sharpe_ci_low:       float
    sharpe_ci_high:      float
    ci_level:            float   # the confidence level used for the CI (default 0.95)
    sortino:             float
    calmar:              float
    information_ratio:   float   # ann. Sharpe of (returns - benchmark); NaN if no benchmark
    beta:                float   # OLS slope vs benchmark; NaN if no benchmark
    modified_sharpe:     float
    psr:                 float   # vs sr_star=0
    min_trl:             float   # vs sr_star=0, bars
    var_5:               float
    cvar_5:              float
    max_drawdown:        float   # positive fraction
    time_under_water:    float   # fraction of bars
    longest_underwater:  int     # bars
    n_obs:               int     # = len(returns.dropna())
    ann_factor:          int     # the annualization factor used
    median_cal_year_return: float  # median calendar-year compound; NaN if no dated index
    median_arith_ann_return: float # median of within-year mean*ann_factor; NaN if no dated index
    returns:             np.ndarray = field(default=None, repr=False, compare=False)
    yearly:              pd.DataFrame | None  # per-year table; use yearly_table()
```
`source: backtest/metrics/report.py:36–68`

28 user-facing fields (26 metrics + 2 metadata: `n_obs`, `ann_factor`) plus the hidden `returns` array and `yearly` DataFrame (both excluded from `repr` and `to_dict`). The `returns` array is retained on the report so `psr_at(sr_star)` and `min_trl_at(sr_star, alpha)` can be called later without recomputing — see §8.4. `source: backtest/metrics/report.py:69–76`

Five of these fields — `hit_rate`, `turnover`, `margin`, `information_ratio`, `beta` — are **conditional**: `hit_rate` is always computed from returns; `turnover` and `margin` need `trades=`; `information_ratio` and `beta` need `benchmark=`. When the inputs aren't passed, the field is `NaN` and the corresponding row is **hidden** from `__repr__` / `_repr_html_` output. See §6.5 and §2.

---

## 2. `compute_metrics` — the entry point

```python
compute_metrics(
    returns,                          # pd.Series | array-like
    equity,                           # pd.Series | array-like
    costs=None,                       # pd.Series | array-like | None
    trades=None,                      # pd.DataFrame | array-like | None
    benchmark=None,                   # pd.Series | array-like | None
    ann_factor: int = 252,
    rf: float = 0.0,                  # annualized
    var_alpha: float = 0.05,
    ci_level: float = 0.95,
) -> MetricsReport
```
`source: backtest/metrics/report.py:127–228`

| Arg | Default | Meaning |
|---|---|---|
| `returns` | required | Bar-to-bar returns. Engine convention: index 0 is `NaN`. `compute_metrics` drops it via `dropna()`. `source: backtest/metrics/report.py:139–140` |
| `equity` | required | Equity curve over time, same length as `returns`. Used for max-drawdown, time-under-water, total return, total PnL, turnover (if trades passed). |
| `costs` | `None` | Per-bar cost series (positive). If supplied, `total_costs` and `gross_pnl` are computed; otherwise both are `NaN`. `source: backtest/metrics/report.py:182–188` |
| `trades` | `None` | `DataFrame[date × asset]` of executed dollar trades (e.g. `result.trades`). Enables `turnover`. Otherwise `turnover = NaN`. `source: backtest/metrics/report.py:190–194` |
| `benchmark` | `None` | Returns series of a reference asset/portfolio. Enables `information_ratio` and `beta`. Aligned by index intersection (Series) or truncation (array). Otherwise both are `NaN`. `source: backtest/metrics/report.py:195–201` |
| `ann_factor` | `252` | Daily-bar default (252 trading days/year). Override for non-daily data: weekly = 52, monthly = 12, hourly = 252×6.5, minute-bar = 252×390. |
| `rf` | `0.0` | Annualized risk-free rate. Per-bar excess return is `r - rf/ann_factor`. Affects Sharpe and Sortino only. `source: backtest/metrics/core.py:28, 43` |
| `var_alpha` | `0.05` | Quantile for VaR/cVaR. `0.05` → 5% worst-case tail. `0.01` → 1%. |
| `ci_level` | `0.95` | Confidence level for the Sharpe CI band. Must be in `(0, 1)`. Stored on the report; downstream plotting uses it as default. `source: backtest/metrics/report.py:136–137, 156` |

`Engine.summary()` now forwards `trades=self.trades` automatically (`backtest/engine/engine.py:60–64`), so `result.summary()` reports turnover out of the box. To get `information_ratio` and `beta`, pass `benchmark=` to `compute_metrics(...)` directly — `summary()` does not currently accept a benchmark argument.

### 2.1 NaN handling

`returns` and `costs` are coerced to a 1-D float array. NaN entries are **dropped** (Series) or **filtered to finite values** (array). `source: backtest/metrics/core.py:11–17, report.py:137–141`

So `n_obs = len(returns) - 1` for engine output (one leading NaN per the convention in skill 05 §3) and `n_obs = len(returns) - n_NaN` more generally.

### 2.2 Years for annualized return

`source: backtest/metrics/report.py:158–172`

```python
if isinstance(equity, pd.Series) and isinstance(equity.index, pd.DatetimeIndex):
    years = (equity.index[-1] - equity.index[0]).days / 365.25     # calendar time
else:
    years = len(r_arr) / ann_factor                                # bar-count fallback
ann_return = (1 + total_return) ** (1 / years) - 1
```

When `equity` is a `pd.Series` with a `DatetimeIndex` (the engine's normal output), **CAGR uses calendar time** — the actual span from `equity.index[0]` to `equity.index[-1]`. For array or `RangeIndex` input, falls back to `len(r_arr) / ann_factor`.

This means a 252-bar daily backtest spanning a real calendar year reports `ann_return ≈ total_return`. The earlier `len(r_arr) / ann_factor` formula misreported it as `total_return^(252/251) ≈ total_return × 1.004` due to the engine's leading-NaN convention. The fallback is preserved for arrays so legacy callers without dates still get a sane answer.

### 2.3 What's `NaN` and why

| Field | NaN when |
|---|---|
| `sharpe`, `sortino` | `< 2` finite obs, or `std < 1e-12` (constant returns). |
| `sortino` only | `< 2` *downside* obs (e.g., the strategy is always up). |
| `calmar` | `< 2` obs, or `max_drawdown == 0` (pure uptrend), or `mdd` non-finite. |
| `modified_sharpe` | `< 2` obs, or VaR not finite, or VaR `>= 0` (no negative tail in sample). |
| `psr`, `min_trl` | SR not computable, or `var_term <= 0`. |
| `min_trl` only | Observed SR ≤ `sr_star` (would never reach significance against itself). |
| `sharpe_ci_low/high` | `sharpe_std` is NaN (no asymptotic SE computable). |
| `total_costs`, `gross_pnl` | `costs is None`. |
| `total_return`, `ann_return`, `total_pnl` | First or last equity bar non-positive / non-finite. |
| `ann_vol` | `< 2` obs. |
| `hit_rate` | No finite returns. |
| `turnover` | `trades is None`, no bars, or mean equity non-positive. |
| `margin` | `trades is None`, total traded is zero, or `total_pnl` not finite. |
| `information_ratio` | `benchmark is None`, `< 2` aligned obs, or excess returns are constant. |
| `beta` | `benchmark is None`, `< 2` aligned obs, or benchmark variance ≈ 0 (constant benchmark). |

NaN is **information**, not failure. Report it; do not silently coerce to 0.

### 2.4 Calendar-year breakdown (OOS / regime checks)

When `returns` is a `pd.Series` with `DatetimeIndex` or `PeriodIndex`, `compute_metrics` also fills:

| Field / method | Meaning |
|---|---|
| `median_cal_year_return` | Median of **calendar-year compound** returns (decimal). Robust summary for choppy OOS windows. |
| `median_arith_ann_return` | Median of within-year **mean × ann_factor** (decimal). Comparable to paper Table 1 style. |
| `yearly` | `DataFrame` indexed by year: `n`, `total_return`, `arith_ann`, `sharpe`, `max_drawdown`. Hidden from `repr`; use `rep.yearly_table()`. |
| `rep.yearly_table()` | Copy of the per-year table for display / export. |

Standalone helpers (`backtest.metrics.period`, re-exported from `backtest.metrics`):

```python
from backtest.metrics import (
    calendar_year_returns,
    median_calendar_year_return,
    median_arith_annual_return,
    yearly_metrics,
)

yr = calendar_year_returns(result.returns)          # Series[year -> compound ret]
tbl = yearly_metrics(result.returns, result.equity_curve, ann_factor=12)
med = median_calendar_year_return(result.returns)
```

`source: backtest/metrics/period.py`, wired in `report.py:compute_metrics`.

**Requires a dated index.** Array-only returns skip yearly fields (`NaN` / `yearly=None`) without error.

**Do not confuse:**

- `ann_return` on `MetricsReport` = **CAGR** over the whole window (equity calendar span).
- `total_return` in `yearly_table` = **compound return within that calendar year only**.
- `arith_ann` in `yearly_table` = mean monthly × `ann_factor` within that year (partial years OK but noisy).

For post-paper OOS, report **both** headline `ann_return` (CAGR on the OOS slice) and `median_cal_year_return` (typical calendar year).

---

## 3. Sharpe — the central metric and its CI

### 3.1 Point estimate (paper Eq. 1)

`source: backtest/metrics/core.py:20–32`

```python
sharpe = excess.mean() / excess.std(ddof=1) * sqrt(ann_factor)
```

where `excess = r - rf/ann_factor`. Sample standard deviation (`ddof=1`). The `sqrt(ann_factor)` annualizes a per-bar Sharpe. Default `ann_factor=252` for daily bars.

### 3.2 Asymptotic distribution (paper Eq. 2)

`source: backtest/metrics/core.py:114–143`

The variance term, in the form the code uses (with `g4` = non-Fisher kurtosis):
```
var_term = 1 - g3 · SR_pb + (g4 - 1)/4 · SR_pb²
```

Equivalently, with `g4` in Fisher (excess) form `k = g4 - 3`:
```
var_term = 1 + 0.5 · SR_pb² - g3 · SR_pb + k/4 · SR_pb²
```
Algebraic identity tested at `metrics_test.py:221–229`.

Per-bar SE: `sqrt(var_term / (T - 1))`. Annualized SE: multiply by `sqrt(ann_factor)`.

`sharpe_distribution(returns, ann_factor) -> (mean, std)` returns the annualized point estimate and SE. `compute_metrics` then turns this into a `(ci_low, ci_high)` band using `norm.ppf(0.5 + ci_level/2)` as the half-width multiplier. `source: backtest/metrics/report.py:151–156`

**Interpretation.** The CI says: under the asymptotic normal approximation, the true SR lies within `[ci_low, ci_high]` with probability `ci_level`. It is **not** a frequentist CI on the *next* sample — it's on the population SR given this sample. Skew and kurtosis widen the band: a heavy-tailed strategy has a wider CI for the same observed Sharpe.

### 3.3 The `sharpe_var_term` export

`sharpe_var_term(r, sr_pb)` is exposed publicly. Useful when you want to compute PSR/MinTRL against a different threshold without re-running the full report. `source: backtest/metrics/__init__.py:11, core.py:114–122`

---

## 4. PSR and MinTRL — "is this Sharpe real?"

### 4.1 PSR (paper Eq. 3)

`source: backtest/metrics/core.py:146–162`

```python
psr = Φ( (SR_pb - sr_star_pb) · √(T - 1) / √var_term )
```

Probability that the **true** Sharpe exceeds `sr_star` (annualized), given the observed sample. `sr_star_pb = sr_star / √ann_factor`. Returns a value in `[0, 1]`.

Properties locked in by tests:

- At `sr_star = SR̂`: `PSR ≈ 0.5` (you're equally likely to be above or below your own estimate). `metrics_test.py:87–91`
- Monotone decreasing in `sr_star`. `metrics_test.py:94–98`
- Monotone increasing in `T` (more data → more confidence). `metrics_test.py:101–104`

`compute_metrics` records `psr` against `sr_star=0` by default. To check against a non-zero threshold without recomputing the whole report, use `rep.psr_at(sr_star)` (§8.4).

### 4.2 MinTRL (paper Eq. 4)

`source: backtest/metrics/core.py:165–183`

```python
min_trl = 1 + var_term · (z_α / (SR_pb - sr_star_pb))²
```

Minimum number of **bars** required to be `1 - alpha` confident that true SR > `sr_star`. With `alpha=0.05` and `sr_star=0`: the bars-to-significance for an SR-vs-zero test.

NaN if observed `SR_pb <= sr_star_pb` — you can't establish significance against a threshold you didn't beat. `source: backtest/metrics/core.py:177–178`

Properties:
- Decreases as observed SR increases (better strategies need less data). `metrics_test.py:107–110`
- Increases as `alpha` decreases (tighter confidence demands more data). `metrics_test.py:113–117`

To compare to your actual sample size: `rep.min_trl < rep.n_obs` → significant; `rep.min_trl > rep.n_obs` → underpowered.

### 4.3 Both annualize via the same `√ann_factor` mapping

`sr_star` is passed in as annualized; the code converts to per-bar with `sr_star / √ann_factor`. Do not pre-convert. `source: backtest/metrics/core.py:156, 176`

---

## 5. Drawdown and time-under-water

Both are **path-based** (operate on the equity curve, not returns) and **numba-jitted** for speed. `source: backtest/metrics/path.py:7, 25`

### 5.1 `max_drawdown(equity) -> float`

`source: backtest/metrics/path.py:7–22`

Walks the equity curve, tracks rolling peak, returns the maximum `(peak - equity) / peak` as a **positive fraction**. With a monotone-up curve: returns `0.0`. `metrics_test.py:126–139` lock in the formula.

### 5.2 `time_under_water(equity) -> (frac, longest)`

`source: backtest/metrics/path.py:25–44`

Returns a tuple:
- `frac` = fraction of bars strictly below the running peak.
- `longest` = the longest consecutive run of underwater bars (in bars).

A bar at the peak (`equity[i] >= peak`) resets the underwater counter. `metrics_test.py:142–160` cover the simple, all-underwater, and monotone-up cases.

### 5.3 In the `MetricsReport`

`max_drawdown` becomes `rep.max_drawdown` (positive fraction). `time_under_water` is split into `rep.time_under_water` (the fraction) and `rep.longest_underwater` (the count of bars, as `int`). `source: backtest/metrics/report.py:148–149, 199–200`

---

## 6. Secondary metrics

### 6.1 Sortino

`source: backtest/metrics/core.py:35–50`

Like Sharpe, but denominator is **downside semi-deviation**: `√(mean(r⁻²))` where `r⁻ = r[r < 0]` (excess returns, downside only). Annualized by `√ann_factor`. NaN if fewer than 2 downside obs.

### 6.2 Calmar

`source: backtest/metrics/core.py:53–70`

```python
calmar = r.mean() · ann_factor / max_drawdown(equity)
```

Ratio of annualized mean return to max drawdown. NaN if no drawdown or `< 2` obs.

### 6.3 VaR / cVaR

`source: backtest/metrics/core.py:73–88`

- `value_at_risk(returns, alpha=0.05)` = `np.quantile(returns, alpha)`. Returns the `alpha`-quantile, typically **negative** for a sensible return series.
- `conditional_value_at_risk` = mean of returns at or below the VaR threshold. The tail expectation. Always `<= var`. `metrics_test.py:67–72`.

Both are **per-bar**, not annualized. Daily VaR at α=0.05 is "the worst-5% daily return".

### 6.4 Modified Sharpe

`source: backtest/metrics/core.py:91–102`

```python
modified_sharpe = r.mean() · ann_factor / |VaR|
```

Replaces the denominator with `|VaR|` (the worst-5% tail) instead of standard deviation. Penalizes left-tail risk more than Sharpe does. NaN if VaR ≥ 0 (no negative tail). Favre & Galeano 2002.

### 6.5 Hit rate, turnover, information ratio, beta

The four practitioner metrics. All are exported as top-level functions in `backtest.metrics` and as fields on `MetricsReport`. NaN-when conditions are in §2.3.

**`hit_rate(returns) -> float`.** `source: backtest/metrics/core.py:186–191`. Fraction of bars with strictly positive return. Always computed (always populated on the report). NaN only if no finite observations.

**`turnover(trades, equity) -> float`.** `source: backtest/metrics/core.py:194–214`. Total **two-way** book turns over the backtest period. Formula:
```
turnover = Σ |trades| / mean(equity)
```
A turnover of `2.0` means the portfolio traded twice its average book value in total over the period. Not annualized — divide by the number of years externally if you want an annual rate. Halve externally for one-way. `trades` is the engine's `result.trades` (a DataFrame of dollar trades per asset per bar); NaN entries are ignored.

**`margin(total_pnl, trades) -> float`.** `source: backtest/metrics/core.py:217–232`. Net PnL per dollar traded. Formula:
```
margin = total_pnl / Σ |trades|
```
A margin of `0.01` means the strategy earned 1 cent of net PnL per dollar of two-way volume. NaN if trades total is zero or `total_pnl` is not finite.

**`information_ratio(returns, benchmark, ann_factor=252) -> float`.** `source: backtest/metrics/core.py:244–259`. Annualized Sharpe of `(returns - benchmark)`. Series inputs are aligned by **index intersection** (then drop-NaN); array inputs are truncated to the shorter and joint-NaN-filtered. NaN if `< 2` aligned obs or excess returns are constant (e.g. IR against self is NaN, not 0).

**`beta(returns, benchmark) -> float`.** `source: backtest/metrics/core.py:262–270`. OLS slope of `returns` regressed on `benchmark`: `cov(r, b) / var(b)`. Alignment same as IR. `beta(r, r) == 1.0` exactly. NaN if benchmark is constant.

These metrics fill in the practitioner gap. The engine still doesn't include "trade-level" metrics (win/loss per round-trip trade) because the engine does not track round-trip trades — `result.trades` is per-bar dollar deltas, not a trade log.

---

## 7. Annualization

The default `ann_factor = 252` assumes **daily business-day bars**. For other frequencies:

| Frequency | `ann_factor` |
|---|---|
| Daily (business) | 252 |
| Daily (calendar) | 365 |
| Weekly | 52 |
| Monthly | 12 |
| Quarterly | 4 |
| Hourly (NYSE) | 252 × 6.5 = 1638 |
| 1-minute (NYSE) | 252 × 390 = 98 280 |

Pass it to `compute_metrics(ann_factor=...)`. Also pass it to `sharpe_ratio`, `sortino_ratio`, etc. when calling them directly.

**What does `ann_factor` affect?** Sharpe, Sortino, Calmar, modified Sharpe (all the annualized ratios), PSR / MinTRL (the per-bar / annualized SR conversion), ann_return, ann_vol, sharpe_std and the CI bounds. It does **not** affect VaR / cVaR, max DD, time-under-water, total return, total PnL, total costs.

If `result.returns` has been resampled to a different bar size (e.g. `returns.resample("W").apply(...)`), use the matching `ann_factor`.

---

## 8. Display and follow-up methods

### 8.1 `__repr__` — plain text (`source: backtest/metrics/report.py:102–108`)

Used outside Jupyter (or when an LLM prints the report). Produces a name-aligned monospaced block headed `MetricsReport:`. Always shows the 21 fields; "Total costs" and "Gross PnL" rows are hidden when `total_costs` is NaN. `source: backtest/metrics/report.py:81–83`

### 8.2 `_repr_html_` — HTML table (`source: backtest/metrics/report.py:110–122`)

Auto-invoked by Jupyter when a `MetricsReport` is the last expression in a cell. Returns a styled `<table>`. Useful when you want a nicely-formatted report inside a notebook.

### 8.3 `plot_sharpe_distribution` — the asymptotic SR plot

```python
rep.plot_sharpe_distribution(ax=None, ci_level=None) -> matplotlib.axes.Axes
```
`source: backtest/metrics/report.py:61–68; backtest/metrics/plot.py:11–75`

Plots the asymptotic Normal `N(SR̂, sharpe_std²)` curve, with:
- Filled CI band at the report's `ci_level` (or override).
- Red vertical line at the estimate.
- Dashed gray line at `sr_null` (default `0`).

`ValueError("insufficient data for Sharpe distribution")` if `sharpe_std` is NaN or zero. `source: backtest/metrics/plot.py:38–40`

### 8.4 `psr_at` and `min_trl_at` — re-evaluate against different thresholds

```python
rep.psr_at(sr_star=1.0)       # PSR against annualized SR = 1
rep.min_trl_at(sr_star=1.0, alpha=0.05)
```
`source: backtest/metrics/report.py:53–59`

Both reuse the cached `returns` array on the report, so they don't recompute the full set of metrics. Useful for the iterative "what SR threshold could I beat?" inquiry.

---

## 9. Common mismatches — surface these by default

Use the question shape from `PROTOCOL.md §3`. Engine-faithful default first.

| Spec says | Engine has | Question to ask |
|---|---|---|
| "Sharpe with annualization factor √260" | `ann_factor=252` default. | Confirm calendar convention. 252 (business days), 260 (business-day approximation), 365 (calendar)? Pass `ann_factor=...` explicitly. |
| "Sharpe assuming 5% risk-free rate" | `compute_metrics(rf=0.05)` — annualized. | Confirm `rf` is annualized; the engine converts to per-bar. |
| "Show me the Sharpe confidence interval" | `rep.sharpe_ci_low`, `rep.sharpe_ci_high`. Default 95%. | Adjust via `compute_metrics(ci_level=0.99)`. |
| "Information ratio" | `compute_metrics(..., benchmark=bench)`. Stored at `rep.information_ratio`. Aligned by index intersection. | Confirm benchmark series and its alignment with `result.returns`. |
| "Hit rate / win rate" | `rep.hit_rate` = `(returns > 0).mean()`. Always populated. | Confirm definition: strictly positive bar = win. Bars with return = 0 are *not* counted as wins. |
| "Total trades / per-trade P&L" | Not in the report. The engine does not track round-trip trades — `result.trades` is per-bar dollar deltas, not a trade log. | Out of scope for the metrics layer. To compute round-trip statistics, post-process `result.positions` (a bar with `sign(positions) ≠ sign(prev_positions)` is a flip). |
| "Turnover" | `rep.turnover` if `result.trades` is passed. `Engine.summary()` forwards it automatically. | Confirm two-way vs one-way (engine reports two-way). |
| "Beta vs SPY" | `compute_metrics(..., benchmark=spy_returns)`. Stored at `rep.beta`. OLS slope. | Confirm benchmark series and that it's already aligned to the strategy's bar frequency. |
| "I want Sharpe with `ddof=0`" | Engine uses `ddof=1` (sample) per Eq. 1. | Surface as a mismatch. Engine-faithful is `ddof=1`. |
| "Negative max drawdown convention" | Engine returns `max_drawdown` as a **positive fraction**. Plots show negative for visual convention. | Confirm. Sign flip is in the plotting code (`skills/05-engine-single-path.md §7`), not in the report. |
| "Monthly Sharpe" | Resample returns to monthly, then pass `ann_factor=12`. | Recommend resample-then-compute, not divide-by-21. Surface the choice. |
| "Information ratio vs benchmark with bootstrap CI" | Not supported. Engine's Sharpe CI is asymptotic Normal, not bootstrap. | Out of scope. Recommend external computation. |
| "Show me drawdown duration in calendar days, not bars" | `longest_underwater` is bars. Conversion requires `panel.dates`. | Convert: `panel.dates[rep.longest_underwater] - panel.dates[0]` (approximation). Recommend keeping bars and noting the bar-to-day ratio in the cell announcement. |
| "Skewness / kurtosis as separate fields" | Not in the report. The variance term uses them but doesn't expose them. | (a) Derive from `result.returns.dropna().skew()` and `.kurt()` in a separate cell. (b) Extend `MetricsReport`. Recommend (a). |

---

## 10. Minimal valid cell — Stage 7, metrics

For a single-path run (Stage 4 result):

```python
from backtest.metrics import compute_metrics

rep = compute_metrics(
    result.returns,
    result.equity_curve,
    costs=result.costs,            # enables total_costs / gross_pnl
    trades=result.trades,          # enables turnover
    benchmark=benchmark_returns,   # optional; enables information_ratio + beta
    ann_factor=252,                # daily bars; adjust if resampled
    ci_level=0.95,
)
print(rep)                          # plain-text table; HTML in Jupyter
print()
print(f"Sharpe:    {rep.sharpe:.3f}  ± {rep.sharpe_std:.3f}  "
      f"(95% CI: [{rep.sharpe_ci_low:.3f}, {rep.sharpe_ci_high:.3f}])")
print(f"PSR(0):    {rep.psr:.3f}    {'significant' if rep.psr > 0.95 else 'not significant'} at 95%")
print(f"MinTRL(0): {rep.min_trl:.0f} bars  (sample: {rep.n_obs} bars  "
      f"→ {'powered' if rep.min_trl < rep.n_obs else 'underpowered'})")
```

For a multi-path run (Stage 6 result), use the engine's helpers:

```python
agg = res.aggregate_metrics()
per_path = res.metrics_per_path()    # list[MetricsReport], length = res.n_paths
print(f"Sharpe across {res.n_paths} paths:  "
      f"mean={agg['sharpe_mean']:.3f}, std={agg['sharpe_std']:.3f}, "
      f"min={agg['sharpe_min']:.3f}, max={agg['sharpe_max']:.3f}")
```

Do **not** combine the single-path inspection and the multi-path inspection in one cell — they are separate decisions.

Do **not** run `selection.dsr(...)` in this cell. DSR is Stage 9 (`skills/09-selection.md`) and requires knowing K — even at K=1, that's its own announcement.

---

## 11. Validation checklist (after the cell runs)

- [ ] **`rep.n_obs`** equals `result.returns.notna().sum()`. Engine convention is one leading NaN; if more than one, the strategy is dropping bars somewhere.
- [ ] **`rep.sharpe`** is finite and within `±5`. Sharpe outside that range usually means: too few obs, constant returns (NaN), or a degenerate equity curve.
- [ ] **`rep.sharpe_ci_low < rep.sharpe < rep.sharpe_ci_high`** strictly. If the CI excludes the point estimate, something is off.
- [ ] **`rep.psr`** is in `[0, 1]`. NaN means the SR couldn't be computed; surface as a problem.
- [ ] **`rep.max_drawdown`** is in `[0, 1]`. Above 1 → equity went negative → catastrophic; do not proceed.
- [ ] **`rep.time_under_water`** is in `[0, 1]`. If `1.0`, the strategy is monotone-down — likely a bug.
- [ ] **`rep.longest_underwater <= rep.n_obs`** — basic sanity.
- [ ] **`rep.total_costs`** is finite if `costs` was passed; NaN if not. If you intended to track costs and see NaN, the result didn't carry them.
- [ ] **`rep.min_trl`** is finite if `rep.sharpe > 0`. If NaN here, observed SR ≤ `sr_star` (likely the strategy is not actually profitable).

If any fails: state which. Do not proceed to Stage 8/9 with a broken report.

---

## 12. What NOT to do

- **Do not interpret a `NaN` field as 0.** NaN means undefined or insufficient data. Treating NaN Sharpe as "zero Sharpe" is wrong and will bias any DSR follow-up.
- **Do not pass the unfiltered `result.returns` to `sharpe_ratio` and expect the leading NaN to count.** All `core.py` metrics dropna first (`source: backtest/metrics/core.py:11–17`). `n_obs` is the *finite* count.
- **Do not change `ann_factor` mid-comparison.** All metrics in a `MetricsReport` use the same factor; compare reports only if their `ann_factor` matches.
- **Do not treat `psr` as a p-value.** PSR is `P(true SR > sr_star | sample)`. A `p-value` would be `P(observed SR ≥ X | true SR = sr_star)`. They're not the same; the conversion is `pval ≈ 1 - psr` only when `sr_star = SR̂`.
- **Do not skip DSR after computing PSR.** PSR is a single-trial significance. DSR (Stage 9) deflates for multiple-testing inflation. Even at K=1 the deflation factor is non-trivial because of the False Strategy Theorem; do not report PSR alone as evidence of edge.
- **Do not compute Sharpe by dividing `total_return / ann_vol`.** That's a different quantity. Use `rep.sharpe`.
- **Do not recompute the report after every parameter sweep.** Each report is a trial; log each one in the trial registry (Stage 10, `skills/11-registry.md`) so DSR knows the trial count.
- **Do not include warmup bars in the input returns.** If your strategy has a `lookback` and you didn't pass `start=panel.dates[lookback]` to the engine, the early bars are zeros (or NaN) and they pollute every metric. Re-run Stage 4 with an explicit `start=`.
- **Do not assume `compute_metrics` accepts a `DataFrame` of multiple paths.** It accepts one Series / one array per call. For multi-path, use `MultiPathResult.metrics_per_path()` (`skills/07-multipath.md §6.1`).
- **Do not call `result.summary()` and `compute_metrics()` in the same cell** — `summary` already calls `compute_metrics` internally (`skills/05-engine-single-path.md §7`). Pick one entry point.
- **Do not forget to pass `trades=` / `benchmark=`** if the spec needs `turnover`, `information_ratio`, or `beta`. They default to NaN and the rows are silently hidden from `__repr__` output. `Engine.summary()` auto-forwards `trades`; you must pass `benchmark=` yourself.
- **Do not infer the Sharpe CI from `± 1.96 × ann_vol / √n`.** That formula assumes iid Normal returns; the engine's CI uses the asymptotic distribution with skew and kurtosis corrections (paper Eq. 2). For non-Normal returns, the engine's CI is wider and more honest.
- **Do not treat the `returns` array on the report as the engine's full returns.** It's been `dropna`'d — `len(rep.returns) == rep.n_obs`, not `len(result.returns)`. Use `result.returns` directly when you need the original index.

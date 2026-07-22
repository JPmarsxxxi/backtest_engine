# Skill 14b — Data Clamp (alpha pipeline Stage 2.5)

> **Read `PROTOCOL.md`, `00-overview.md`, `12-alpha-overview.md` first.** Cell-loop and announcement rules apply.

> Stage 2.5. Suppress raw-data feed anomalies (bad ticks, exchange glitches, corporate-action discontinuities) by clipping each asset's time series to a rolling `mu ± k·sigma` band. Runs **after frequency** (so the resample is done first) and **before signal_construction** (so the cleaned series is what the signal sees).

---

## When to load

- **Stage 2.5**: between `frequency` and `signal_construction`. Whenever the raw data source can produce single-bar anomalies that would corrupt the signal (Binance API hiccups, yfinance unadjusted splits, etc.).
- Skip if: you trust the data source completely, OR your signal logic is robust to outliers by construction (e.g. uses median-based statistics).

---

## API

```python
from backtest.alpha_pipeline import data_clamp

cleaned = data_clamp.run(prices)                                # default: k=5, window=24, trailing=True
cleaned = data_clamp.run(prices, k=3.0, window=48)              # tighter band, longer context
cleaned = data_clamp.run(prices, trailing=False)                # match turnover_control._clamp semantics
```
`run(data, k=5.0, window=24, trailing=True) -> pd.DataFrame`. `source: backtest/alpha_pipeline/data_clamp.py:run`

Apply once per field — if you have separate prices, volumes, returns, call it three times.

---

## Math

```
out_t = clip(data_t, mu_{t-1} - k*sd_{t-1}, mu_{t-1} + k*sd_{t-1})       # trailing=True (default)
out_t = clip(data_t, mu_t    - k*sd_t,     mu_t    + k*sd_t)             # trailing=False
```

- `mu`, `sd`: rolling mean and std over `window` bars, per asset, along the time axis.
- `trailing=True` shifts the band by 1 bar so the current value cannot inflate its own band. **This is the right default for data hygiene** — without it, a single huge outlier blows up its own `sd`, the band widens, and the outlier escapes clipping.
- `trailing=False` matches the existing `turnover_control._clamp`; use it only if you need that exact behaviour (e.g. signal-stability use).
- `k=5` (default) keeps the band loose enough to leave normal moves alone; only ≥5σ events get caught. Tighten with `k=3` if you want a stricter filter.
- Warmup bars (where `mu`/`sd` is NaN) are left **unclipped** by pandas's `clip` behaviour.

---

## Why a separate module from turnover_control's clamp?

| | `data_clamp.run` (Stage 2.5) | `turnover_control.run(.., "clamp", ..)` (Stage 6) |
|---|---|---|
| Applied to | raw data (prices, returns, volume) | signal vector (after rank) |
| Purpose | data quality — kill bad ticks | turnover control — cap signal jumps |
| Default trailing | `True` (don't let outlier inflate its own band) | `False` (matches Ch. 7 Finding Alphas math) |
| Pipeline position | after `frequency`, before `signal_construction` | after `rank` and `neutralisation` |
| Math | rolling `mu ± k·sd` clip | rolling `mu ± k·sd` clip |

**Same math, different intent, different stage.** Both can be applied in the same pipeline run — they don't conflict.

---

## Engine fit

Pure DataFrame transform. No engine API. Returns the same shape it received.

---

## Mismatches with the spec

| Idea says | Pipeline has | Question to ask |
|---|---|---|
| "Drop outliers" | clip to band, doesn't drop rows. | Confirm clipping is acceptable (preserves the bar with a corrected value). |
| "Catch corporate actions" | this is a per-bar clip; a clean adjusted-price series is still required. | Surface: should be handled upstream by the data fetch (split-adjusted prices). |
| "Use MAD instead of std" | not implemented; only rolling mean/std. | TODO: extend if MAD-robust is needed. |
| "Apply to multiple fields" | call once per field. | Confirm which fields need cleaning (prices yes; volume maybe; returns rarely). |
| "Same band as turnover_control" | set `trailing=False`. | Confirm intent; trailing=True is the data-hygiene default. |

---

## Minimal valid cell

```python
from backtest.alpha_pipeline import data_clamp

cleaned = data_clamp.run(prices, k=5.0, window=24, trailing=True)

n_clipped = (cleaned != prices).sum().sum()
print(f"clipped {n_clipped} bars across {prices.shape[1]} assets")
print(f"max raw / cleaned ratio: {(prices / cleaned).abs().max().max():.2f}")
print(cleaned.tail(3).round(3))
```

---

## Validation checklist

- [ ] Output shape == input shape; column / index identical.
- [ ] Warmup bars (first ~`window` rows) untouched.
- [ ] Bars that were not outliers are bit-for-bit identical.
- [ ] Bars that were clipped sit inside the trailing `mu ± k·sd` band.
- [ ] No new NaNs introduced outside the warmup region.

---

## What NOT to do

- **Do not** apply this after signal construction — it will operate on the signal, not the data. That's what `turnover_control` is for.
- **Do not** use `trailing=False` with `k` small and a feed that has occasional huge ticks — the outlier inflates its own band and the clip is a no-op on the worst bar.
- **Do not** apply on already-adjusted returns when the underlying source needs price adjustment first — fix the data fetch instead.
- **Do not** tune `k` to remove ordinary volatility — `k=5` is the standard data-hygiene default; tighten only when you have a documented source of moderate anomalies.

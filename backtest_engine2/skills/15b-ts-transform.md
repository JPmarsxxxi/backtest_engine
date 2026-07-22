# Skill 15b — Time-Series Transforms (alpha pipeline Stage 3b)

> **Read `PROTOCOL.md`, `00-overview.md`, `12-alpha-overview.md` first.** Cell-loop and announcement rules apply.

> Stage 3b (optional, after signal_construction, before rank/neutralise). Standardise a signal against its **own trailing history** per asset — the time-series counterpart to Skill 5 (cross-sectional rank) and Skill 4 (cross-sectional neutralise).

---

## When to load

- The raw signal's *level* is only meaningful relative to each asset's own recent distribution (most mean-reversion signals): e.g. "price vs its own 60-bar mean", not "price vs peers".
- You want a self-normalised feature (z-score / rolling rank) before ranking or neutralising cross-sectionally.

---

## API

```python
from backtest.alpha_pipeline import ts_transform as tt

out = tt.run(sig, "zscore", d=60)                    # (x - mean_d) / std_d, per asset
out = tt.run(sig, "rank", d=60)                       # rolling percentile of current bar, (0,1]
out = tt.run(sig, "scale", d=60)                      # (x - min_d) / (max_d - min_d), [0,1]
out = tt.run(sig, "quantile", d=60, driver="gaussian")# ts_rank -> inverse CDF (gaussian/uniform/cauchy)
```
`run(signal, method, d, **params) -> pd.DataFrame`. `source: backtest/alpha_pipeline/ts_transform.py:run`

---

## Math (per asset, rolling window `d`, current bar `t`)

- **zscore** `_zscore(d)`: `(x_t − rolling_mean(d)) / rolling_std(d)`, `std` ddof=1; zero-std bars → NaN. `source: …/ts_transform.py:_zscore`
- **rank** `_rank(d)`: `#{k in window : x_k ≤ x_t} / d`, in `(0, 1]`. Outlier-robust, distribution-free. `source: …:_rank`
- **scale** `_scale(d)`: `(x_t − rolling_min) / (rolling_max − rolling_min)`, in `[0, 1]`; flat window → NaN. `source: …:_scale`
- **quantile** `_quantile(d, driver)`: take `ts_rank`, clip to `(eps, 1−eps)`, map through an inverse CDF — `gaussian` (`NormalDist.inv_cdf`), `uniform` (`2p−1`), or `cauchy` (`tan(π(p−0.5))`). Reshapes/So spreads the tails. `source: …:_quantile`

First `d−1` rows are NaN (warmup); NaNs preserved throughout.

---

## Engine fit

Pure per-asset transform, same date × asset shape in/out. Typically applied to the raw signal from Stage 3 **before** cross-sectional rank (Stage 5) / neutralise (Stage 4). It does **not** touch turnover directly — it reshapes the signal, not the trade schedule.

---

## Mismatches with the spec

| Idea says | Pipeline has | Question to ask |
|---|---|---|
| "z-score the signal" | `zscore` (per-asset, time-series) vs Stage 5 `rank`/`zscore` (cross-sectional) | Which axis: standardise vs its **own history** (ts_transform) or vs **peers this bar** (rank/neutralise)? |
| "normalise over a lookback" | `zscore` / `scale` / `rank` over `d` bars | Which: SD units (zscore), 0–1 min-max (scale), or percentile (rank)? |
| "reshape to normal / tame tails" | `quantile` with gaussian/uniform/cauchy driver | Which driver? gaussian for near-normal, cauchy for heavy tails, uniform for flat. |
| "which window?" | `d` (bars); must be ≥ 2 | Confirm `d` matches the signal's intended horizon (don't standardise over a window longer than the edge lives). |

---

## Minimal valid cell

```python
from backtest.alpha_pipeline import ts_transform as tt

zsig = tt.run(sig, "zscore", d=60)
print("warmup NaN rows:", int(zsig.isna().all(axis=1).sum()))
print(zsig.dropna(how="all").head(3).round(4).to_string())
```
Expect the first ~`d−1` rows all-NaN, then per-asset z-scores centred near 0.

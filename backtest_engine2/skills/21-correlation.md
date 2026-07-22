# Skill 21 — Alpha Correlation (alpha pipeline Stage 9)

> **Read `PROTOCOL.md`, `00-overview.md`, `12-alpha-overview.md` first.** Cell-loop and announcement rules apply.

> Stage 9. Before keeping a new alpha, check it isn't a near-duplicate of one already in the winners pool. Low correlation to the pool = genuinely new edge (Ch. 4, Ch. 8).

---

## When to load

- **Stage 9**: a new alpha scored well (Skill 8) and you're deciding whether it adds anything beyond the existing `winners/` pool.

---

## API

```python
from backtest.alpha_pipeline import alpha_correlation as ac

pool = ac.load_pool("winners", kind="pnl")          # or kind="positions"
res  = ac.run(pool, new_pnl, method="pearson")
res  = ac.run(pool, new_positions, method="position", d=20)
```
`run(pool, new, method="pearson", d=20) -> dict`; `load_pool(winners_dir, kind="pnl") -> ...`. `source: backtest/alpha_pipeline/alpha_correlation.py:run`

Returns `{max_corr, t_corr, avg_score, pairwise, histogram, bin_edges}`.

---

## Math (PDF Ch. 8 "Alpha Correlations"; Ch. 4 thresholds)

- **pearson** `_pearson_1d`: standard correlation of PnL series.
- **temporal** `_temporal`: time-weighted cosine, weights `w = arange(n)/n` (recent bars weighted more) — PDF Eq. 6.
- **weekly**: correlation of 5-bar (weekly) mean PnL.
- **sign**: agreement of PnL signs.
- **position / trading**: stack the last `d` bars of positions / trades, intersect on common assets, correlate.
- **max_corr** = worst-case overlap to any pool member; **t_corr** = temporal version; **avg_score** = weighted blend; plus a `histogram` of pairwise correlations.
- Thresholds (Ch. 4): `> 0.7` too correlated (reject), `< 0.3` good diversifier.

---

## Engine fit

Reads PnL/positions parquet written when an alpha is promoted (Skill 22). No engine API — operates on the saved winners pool.

---

## Mismatches with the spec

| Idea says | Pipeline has | Question to ask |
|---|---|---|
| "Is it correlated?" | several flavours (pearson/temporal/weekly/sign/position/trading). | Correlate **what** — PnL series, or actual positions/trades? |
| "Against what?" | the `winners/` pool only (curated kept alphas), not failed notebooks. | Confirm the pool dir; empty pool → first alpha trivially passes. |
| "One number" | `max_corr` (worst overlap) is the gate; `avg_score` is a blend. | Use `max_corr` against the threshold. |

---

## Minimal valid cell

```python
from backtest.alpha_pipeline import alpha_correlation as ac

pool = ac.load_pool("winners", kind="pnl")
res  = ac.run(pool, new_pnl, method="pearson")
print("max_corr:", round(res["max_corr"], 3), "| verdict:",
      "REJECT" if res["max_corr"] > 0.7 else "KEEP" if res["max_corr"] < 0.3 else "BORDERLINE")
```

---

## Validation checklist

- [ ] Pool loaded from `winners/` (not the experiment notebooks).
- [ ] New alpha aligned to the pool's index/assets before correlating.
- [ ] `max_corr` compared to the Ch. 4 thresholds (0.7 / 0.3).
- [ ] Empty-pool case handled (first alpha auto-passes).

---

## What NOT to do

- **Do not** correlate against failed/experiment notebooks — only the curated `winners/` pool.
- **Do not** keep an alpha with `max_corr > 0.7` no matter how high its IR.
- **Do not** compare misaligned series — intersect dates/assets first.

# Skill 16 — Neutralisation (alpha pipeline Stage 4)

> **Read `PROTOCOL.md`, `00-overview.md`, `12-alpha-overview.md` first.** Cell-loop and announcement rules apply.

> Stage 4. Remove group-level exposure so the book is balanced within each group. **Execution order: this runs *after* rank (Skill 5), per Ch. 5.**

---

## When to load

- **Stage 4 (runtime: after rank)**: making the signal market/industry/sector-neutral.

---

## API

```python
from backtest.alpha_pipeline import neutralisation

neut = neutralisation.run(ranked, "market")                 # whole-row demean (dollar-neutral)
neut = neutralisation.run(ranked, "sector", metadata)        # demean within a STATIC column
neut = neutralisation.run(ranked, "subindustry", metadata)   # ANY static metadata column works
# dynamic / custom groups that change each bar (e.g. volatility deciles):
labels = neutralisation.bucket(vol_field, n_buckets=5)       # date x asset int labels 0..4
neut   = neutralisation.run(ranked, labels=labels)           # demean within per-bar buckets
```
`run(signal, by=None, metadata=None, labels=None) -> pd.DataFrame`: `by='market'`, OR any static `metadata` column, OR `labels=` (date × asset) for dynamic groups. `bucket(field, n_buckets) -> date × asset int labels`. `source: backtest/alpha_pipeline/neutralisation.py:run`, `source: …/neutralisation.py:bucket`

---

## Math (PDF Ch. 5; Ch. 13)

- `neutralised(i,t) = signal(i,t) − mean_{j∈group(i)} signal(j,t)`, per bar. Subtracting the group mean forces each group's row-sum to 0 (Ch. 5: `Sum(alpha within same industry) = 0`).
- `market` = one group = all instruments (dollar-neutral book; no metadata). **Any static metadata column** (`industry`, `sector`, `subindustry`, `country`, …) = group by that column. **`labels=`** (date × asset) = dynamic per-bar groups, so membership can change over time; build them with `bucket()` (cross-sectional quantile binning of any numeric field). `source: …/neutralisation.py:bucket`
- For categorical groups this equals OLS residualisation on group dummies — i.e. it *is* the Ch. 5 neutralisation and a special case of Ch. 13's regression neutralisation. (Ch. 13's *continuous*-factor regression is a different, heavier tool needing factor-return data.)

---

## Engine fit

Same raw date × asset vector shape; no engine API.

---

## Mismatches with the spec

| Idea says | Pipeline has | Question to ask |
|---|---|---|
| "Market-neutral" | `by="market"` (no metadata). | Confirm dollar-neutral is the intent. |
| "Sector/industry-neutral" | needs static `metadata` with that column. | Do you have the labels? Missing label → **raises** (by design). |
| "Neutralise by country / subindustry / any label" | any static `metadata` column via `by="<col>"`. | Is that column present in metadata? |
| "Neutralise within vol / liquidity buckets (time-varying)" | `bucket(field, n)` → `labels=`; per-bar dynamic groups. | Which numeric field, and how many buckets? |
| "Neutralise to momentum/size factor" | not this skill — that's Ch. 13 continuous-factor regression. | Surface: needs factor returns; out of scope here. |

---

## Minimal valid cell

```python
from backtest.alpha_pipeline import neutralisation

neut = neutralisation.run(ranked, "market")
print("per-bar mean (≈0):", neut.mean(axis=1).abs().max().round(8))
print(neut.tail(3).round(4))
```

---

## Validation checklist

- [ ] For `market`: each bar's cross-sectional mean ≈ 0.
- [ ] For `sector`/`industry`/any static column: each group's per-bar sum ≈ 0; single-asset groups become exactly 0.
- [ ] For `labels=` (dynamic buckets): each per-bar bucket group sums ≈ 0.
- [ ] No asset silently un-neutralised (the module raises on a missing group label).

---

## What NOT to do

- **Do not** run neutralisation *before* rank — the runtime order is rank → neutralise (Ch. 5).
- **Do not** leave assets without group labels and expect them dropped — it raises; fix the metadata.
- **Do not** call this "factor neutralisation" in the Ch. 13 regression sense — it's group-mean demeaning.

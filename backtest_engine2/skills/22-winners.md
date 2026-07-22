# Skill 22 — Winners (promoting a kept alpha)

> **Read `PROTOCOL.md`, `00-overview.md`, `12-alpha-overview.md` first.**

> Not a `run()` module — a **procedure**. When an alpha survives Skill 8 (good metrics) and Skill 9 (low correlation to the pool) *and* the user says "promote/keep it", copy the notebook into the `winners/` pool with a card and its PnL/positions, so future correlation checks (Skill 9) can read it.

---

## When to load

- The user explicitly says a notebook is a keeper ("promote this", "add to winners").
- **Never** auto-promote — promotion is a user decision (PROTOCOL: ask, wait).

---

## What `winners/` is

A curated pool of kept alphas. Each lives in its own folder:

```
winners/
  2026-06-01_btc-5d-reversion/
    notebook.ipynb        # the exact experiment that produced it
    alpha_card.md         # idea + recipe + data fetch + metrics (the spec to reproduce)
    pnl.parquet           # net PnL series  -> alpha_correlation.load_pool(kind="pnl")
    positions.parquet     # date × asset positions -> load_pool(kind="positions")
```

`alpha_correlation.load_pool("winners", kind=...)` reads every member's parquet. `source: backtest/alpha_pipeline/alpha_correlation.py:load_pool`

---

## Promotion procedure (one cell)

1. Make `winners/<YYYY-MM-DD>_<slug>/` (date = today, slug = short idea name).
2. Copy the source notebook into it.
3. Write `alpha_card.md` (template below).
4. Save `pnl.parquet` and `positions.parquet` **from the Skill 23 engine result** (`BacktestResult.pnl` is net dollar PnL per bar; `.positions` is the date × asset frame — `source: backtest/engine/engine.py:29,23`). The pool correlation (Skill 9) should use the real engine PnL, not the Skill 8 quick score, which doesn't return a series.

```python
from pathlib import Path
dst = Path("winners") / "2026-06-01_btc-5d-reversion"
dst.mkdir(parents=True, exist_ok=True)
result.pnl.to_frame("pnl").to_parquet(dst / "pnl.parquet")        # result = Engine(...).run()
result.positions.to_parquet(dst / "positions.parquet")
# copy the notebook + write alpha_card.md
print("promoted ->", dst)
```

---

## `alpha_card.md` template (the reproduce-spec)

```markdown
# <Alpha name>
- **Idea:** one-line hypothesis (what edge, why it should exist).
- **Universe:** top_n, dollar-volume window, any category filter.       (Skill 1)
- **Frequency / delay:** e.g. daily, delay=1.                            (Skill 2)
- **Signal:** type + lookback (or the custom fn expression).             (Skill 3)
- **Rank / neutralise:** rank → neutralise(by=...).                      (Skills 5,4)
- **Turnover / decay:** method + params.                                 (Skills 6,7)
- **Data fetch:** exact source + query Claude ran to get the raw data.
- **Metrics (net of cost):** IR, annual_return, daily_turnover, margin, max_drawdown, cost_bps used.
- **Correlation to pool at promotion:** max_corr vs winners (Skill 9).
- **Verdict:** engine CPCV + DSR result (Skill 23) — DSR, K trials.
```

---

## Validation checklist

- [ ] User explicitly approved promotion.
- [ ] `pnl.parquet` + `positions.parquet` written and re-loadable via `load_pool`.
- [ ] `alpha_card.md` records the full recipe **and the data-fetch step** (so it's reproducible).
- [ ] Metrics on the card are net-of-cost and include the cost assumption.
- [ ] Folder slug + date unique (no overwrite of an existing winner).

---

## What NOT to do

- **Do not** promote without the user's go-ahead.
- **Do not** copy failed/experiment notebooks into `winners/` — the pool is curated; Skill 9 correlates only against it.
- **Do not** save zero-cost metrics on the card.
- **Do not** omit the data-fetch step — the card must let you rebuild the alpha from scratch.

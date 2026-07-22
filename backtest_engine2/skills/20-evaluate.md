# Skill 20 — Evaluate (alpha pipeline Stage 8)

> **Read `PROTOCOL.md`, `00-overview.md`, `12-alpha-overview.md` first.** Cell-loop, **mandatory costs**, and DSR rules apply.

> Stage 8. Convert the finished raw signal into positions, simulate PnL net of cost, and score it. This is the **adapter** where the raw vector becomes an engine-shaped result. For the real verdict (CPCV + DSR) hand off to Skill 23.

---

## When to load

- **Stage 8**: a signal has passed through Stages 1–7 and you need its IR / return / turnover / margin / drawdown.

---

## API

```python
from backtest.alpha_pipeline import evaluate

# quick PDF-style scorecard
rpt = evaluate.run(sig, prices, book_size=1_000_000, cost_bps=5.0, mode="pdf")

# route through the engine's metrics (CPCV/DSR handled in Skill 23)
rpt = evaluate.run(sig, prices, book_size=1_000_000, cost_bps=spread_bps, mode="engine")
```
`run(signal, prices, book_size, cost_bps, mode="engine", ann_factor=252) -> dict`. `source: backtest/alpha_pipeline/evaluate.py:run`

Both modes return a `dict`. `mode="pdf"` → `{IR, annual_return, daily_turnover, margin, max_drawdown, pct_profitable_days}`; `mode="engine"` → `MetricsReport.to_dict()` (`sharpe, ann_return, turnover, margin, max_drawdown, …`). It does **not** return the PnL/positions series — for the winners pool (Skill 22) take those from the engine result in Skill 23.

---

## Math (PDF Ch. 6 metrics; Ch. 7 cost)

- **Positions** `_positions`: scale the signal so **gross book = book_size** each bar (dollar exposure per WebSim convention). `source: …/evaluate.py:_positions`
- **Simulate** `_simulate`: `pnl_t = pos_{t−1}·ret_t`; `trades_t = pos_t − pos_{t−1}` (row 0 = initial `pos_0`). `source: …:_simulate`
- **Cost** `_cost_series`: `cost_bps` is **per-instrument half-spread** (Ch. 7) — accepts scalar / per-asset Series / date×asset DataFrame; charged on `|trades|`. `source: …:_cost_series`
- **PDF metrics** `_pdf_metrics`: `IR = mean(pnl)/std(pnl)·√ann_factor`; `annual_return = annualised_pnl / book_size`; `daily_turnover = traded / book`; `margin = pnl / traded`; `max_drawdown` on `book/2` (long–short half-book). `source: …:_pdf_metrics`
- **engine mode** wraps the engine's `compute_metrics` so the scorecard matches the rest of the repo. `source: backtest/metrics/report.py`

---

## Engine fit — this is the touchpoint

This is the only stage that crosses from raw-vector land into engine metrics. `mode="engine"` returns the same `MetricsReport` (with the new total `turnover` and `margin = total_pnl / total_traded`) the engine produces elsewhere. **It is a quick scorer, not the verdict** — it does single-path PnL, no purging, no DSR deflation. Promote only after Skill 23.

---

## Mismatches with the spec

| Idea says | Pipeline has | Question to ask |
|---|---|---|
| "What's the Sharpe?" | `IR` with `ann_factor=252` (set 365 for 24/7 crypto). | Which annualisation factor? |
| "Use a flat 5 bps cost" | scalar `cost_bps` works, but per-instrument half-spread is the accurate Ch. 7 model. | Do you have per-asset spreads? Flat is a simplification. |
| "Drawdown on full book" | PDF normalises on `book/2` (half-book long–short). | Confirm `book/2` denominator. |
| "Is it tradable?" | this is a quick score; tradability = CPCV + DSR. | Route to Skill 23 before trusting it. |

**Costs are mandatory** (PROTOCOL): never report a zero-cost number as the headline.

---

## Minimal valid cell

```python
from backtest.alpha_pipeline import evaluate

rpt = evaluate.run(final_signal, prices, book_size=1_000_000, cost_bps=5.0, mode="pdf")
for k in ("IR", "annual_return", "daily_turnover", "margin", "max_drawdown"):
    print(f"{k:>15}: {rpt[k]:.4f}")
```

---

## Validation checklist

- [ ] Cost is non-zero and stated (half-spread bps or per-asset series).
- [ ] Gross book ≈ `book_size` each bar (positions scaled correctly).
- [ ] `pnl` uses `pos.shift(1)` — no same-bar lookahead.
- [ ] Annualisation factor matches the data cadence (252 vs 365).
- [ ] Result flagged as *quick score*; DSR/CPCV deferred to Skill 23.

---

## What NOT to do

- **Do not** present a quick-eval IR as a verdict — it has no purging or DSR.
- **Do not** report zero-cost metrics as headline numbers.
- **Do not** use `pos_t` (not shifted) in PnL — that's lookahead.
- **Do not** normalise drawdown on the full book when the alpha is long–short (use `book/2`).

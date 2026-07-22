# Skill 12 — Alpha Pipeline Overview

> **Read `PROTOCOL.md` and `skills/00-overview.md` first.** The cell-loop and announcement rules in PROTOCOL apply unconditionally to every alpha-pipeline cell, exactly as they do to the engine stages.

---

## When to load this skill

At the start of an **alpha-generation hunt** — i.e. when the user states an *idea* (or asks you to propose one) rather than handing you a finished strategy spec. Load it once, immediately after `PROTOCOL.md` and `00-overview.md`, then read the matching per-stage guide as you reach each cell.

---

## The two on-ramps to the engine

There are two distinct ways to reach the same backtest engine. Do not confuse them:

1. **Spec flow (`skills/01–11`).** User hands you a strategy spec → you wire it into the engine via the cell-loop. Unchanged.
2. **Alpha pipeline (`skills/12–21`, this flow).** User states an idea → you *construct* a raw alpha vector through 9 stages → the adapter feeds it to the **same** engine for the verdict.

The pipeline is the **front-end that generates** an alpha; the engine is the **back-end that judges** it. The pipeline never replaces the engine — costs, CPCV, and DSR are still mandatory before any alpha is believed.

---

## The pipeline (canonical order)

```
prices + volume  (fetched by you, never pasted)
        │
        ▼
1 universe ──► boolean date×asset mask (top-N liquidity + sector/region filter)
        │
        ▼
2 frequency ─► resample (tick/intraday passthrough; D/W/M) + delay PIT shift
        │
        ▼
2.5 data_clamp ─► rolling mu±k·σ clip on RAW data (bad-tick / feed-glitch hygiene)
        │            optional but recommended on noisy feeds
        ▼
2.7 EDA ─────► hypothesis-driven exploration (skill-only, no module)
        │       decompose the idea → test each sub-claim → confirm/refute/refine
        │       before signal_construction. See `14c-eda.md`.
        ▼
3 signal_construction ─► raw signal vector (mean_rev / momentum / fundamental / custom)
        │
        ▼
3b ts_transform ─► per-asset rolling zscore/rank/scale/quantile (optional; vs OWN history)
        │
        ▼
5 rank ──────► winsor+scale to [-1,1] (preserves interior magnitudes)
        │                                  (EXECUTION ORDER: rank BEFORE
        ▼                                   neutralise, per Ch. 5)
4 neutralisation ─► demean within market / any static column / dynamic buckets
        │
        ▼
6 turnover_control ─► clamp / hump / ema / linear / trade_when  (on SIGNAL — distinct from 2.5)
        │
        ▼
7 decay ─────► linear / ema / sma smoothing
        │
        ▼
8 evaluate ──► positions → PnL → metrics (engine mode | pdf mode)
        │
        ▼
9 alpha_correlation ─► uniqueness vs the winners/ pool
```

**Execution order note.** The modules are numbered per spec (4 = neutralisation, 5 = rank), but the *runtime* order applies **rank before neutralise** (Ch. 5: `Alpha3 = rank(Alpha1)`, then `Sum(Alpha3 within industry) = 0`). `source: backtest/alpha_pipeline/__init__.py` docstring.

---

## Module map

Every stage is one pure module under `backtest/alpha_pipeline/`, each with `run()` + `quick_test()`. The "Guide" column is the per-stage skill to read before writing that cell.

| Stage | Module | Entry point | Guide |
|---|---|---|---|
| 1 | `universe` | `run(prices, volume, top_n, ...)` | `13-universe.md` |
| 2 | `frequency` | `run(data, freq, delay, agg)` | `14-frequency.md` |
| 2.5 | `data_clamp` | `run(data, k, window, trailing)` | `14b-data-clamp.md` |
| 2.7 | **EDA (methodology)** | no module — hypothesis-driven exploration in cells | `14c-eda.md` |
| 3 | `signal_construction` | `run(data, signal_type, lookback, fn, ...)` | `15-signal-construction.md` |
| 3b | `ts_transform` | `run(signal, method, d, ...)` (zscore/rank/scale/quantile) | `15b-ts-transform.md` |
| 4 | `neutralisation` | `run(signal, by=None, metadata=None, labels=None)` + `bucket(field, n)` | `16-neutralisation.md` |
| 5 | `rank` | `run(signal, low_pct, high_pct)` (winsor+scale) | `17-rank.md` |
| 6 | `turnover_control` | `run(signal, method, **params)` (clamp/hump/ema/linear/trade_when) | `18-turnover-control.md` |
| 7 | `decay` | `run(signal, method, window)` | `19-decay.md` |
| 8 | `evaluate` | `run(signal, prices, book_size, cost_bps, mode)` | `20-evaluate.md` |
| 9 | `alpha_correlation` | `run(pool, new, method)` ; `load_pool(winners_dir)` | `21-correlation.md` |

**On `data_clamp` (2.5) vs `turnover_control`'s `clamp` (Stage 6):** same math (rolling `mu ± k·σ` clip), different intent. `data_clamp` operates on **raw data** to suppress feed anomalies; `turnover_control`'s clamp operates on the **signal** to cap signal-level outliers (Ch. 7). Both can be applied in the same run; they are not redundant. See `14b-data-clamp.md` for the contrast table.

Two procedures sit alongside the 9 modules (no `run()` of their own):

| — | promote a keeper to the pool | move notebook + card + parquet to `winners/` | `22-winners.md` |
| — | the **real verdict** (Strategy wrapper → Engine + CPCV + DSR) | `23-engine-onramp.md` |

`source: backtest/alpha_pipeline/__init__.py:1`

---

## The cell loop applies here too

One stage = **one announced notebook cell** = one turn. Before each cell, post the PROTOCOL §2 announcement (what it does, stage, guide consulted, API used, mismatches, decisions needed), then **wait**. Never chain two stages in one cell. There is **no** "run all 9" call — that would bulldoze the loop.

Each cell must end in output you can inspect (`.head()`, a shape, a metric print).

---

## What flows between cells

Skills 1–7 pass **raw alpha vectors** (per-instrument signal proportional to dollars to hold; WebSim convention). The conversion to engine positions happens once, at Skill 8's adapter — *not* in the intermediate stages. Do not normalise to weights early.

- Stage 1 output (boolean mask) feeds `DataPanel(universe=...)` directly, and gates which assets the later stages act on.
- Stages 2–7 output date × asset signal frames.
- Stage 8 turns the final signal into positions/PnL/metrics.
- Stage 9 consumes Stage 8's PnL (or positions) and the `winners/` pool.

---

## The winners/ loop

When an alpha clears the bar, the user says "promote it". You then:
1. move the experiment notebook to `winners/<YYYY-MM-DD>_<slug>/`,
2. write `alpha_card.md` (idea + pipeline recipe + data-fetch + metrics),
3. save `pnl.parquet` (and optionally `positions.parquet`).

`alpha_correlation.load_pool("winners/")` reads every winner's `pnl.parquet`/`positions.parquet` as the correlation pool, so each new candidate is automatically checked against the accumulated keepers (Ch. 8). See `22-winners.md`.

---

## Always-on rules (repeated from PROTOCOL)

1. **Cite or don't claim.** Every API/formula/default about a primitive ends with `source: backtest/alpha_pipeline/<module>.py:<fn>`; every math claim cites the PDF chapter.
2. **Costs are mandatory.** `evaluate` charges costs (per-instrument half-spread is the accurate model); a no-cost score is never the verdict.
3. **DSR is mandatory** before believing an alpha — even at K=1, via the engine on-ramp (`23-engine-onramp.md`). Skill 8's score is a quick filter, never the verdict.
4. **One cell per turn.**
5. **Surface mismatches; never paper over.** If the idea needs something the pipeline can't express cleanly, that's the user's decision.

---

## Where to go next

After this file, read `13-universe.md` and announce Cell 1 (the universe cell). Cell 1 of any alpha hunt is always the data fetch + universe construction.

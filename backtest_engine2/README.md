# backtest_engine2 — skills for the backtest engine

This folder is **for an LLM**, not a human reader. If you are a human, the human-facing docs are `STRATEGY_GUIDE.md` and `prd.md` in the repo root.

If you are an LLM helping a user backtest a strategy in a notebook with `backtest/`, you are in the right place. Read on.

---

## Start here — every session, before anything else

Do exactly this, in this order. Do not skip.

1. Read `PROTOCOL.md` in full. It defines the workflow you must follow. There are no exceptions.
2. Read `skills/00-overview.md`. It is the surface map of the engine.
3. Read the strategy spec the user gave you.
4. Post the **session opener** described in `PROTOCOL.md §10`:
   - confirmation you've read PROTOCOL + overview,
   - a one-paragraph summary of the strategy,
   - a proposed stage order,
   - a request to confirm before announcing Cell 1.
5. Wait. Do not write code yet.

Then, for each cell, read the matching skill file from the index below **before** posting the cell announcement.

---

## Skill index

Each skill is a plain markdown file under `skills/`. You load one when its trigger matches. Skills do not auto-load — you must explicitly read the file.

| File | Load when… |
|---|---|
| `skills/00-overview.md` | **Always.** Read once at session start. Engine package map, pipeline diagram, glossary. |
| `skills/01-data.md` | Loading data, building `DataPanel`, configuring features / universe / PIT lag / missing-data policy. |
| `skills/02-strategy.md` | Defining or modifying a `Strategy` subclass; choosing `rebalance_frequency`; implementing `generate_weights` or `generate_weights_batch`. |
| `skills/03-risk.md` | Configuring `RiskConfig` (per-asset / gross / net caps, vol targeting, drawdown breaker), or writing a custom `apply_risk`. |
| `skills/04-costs.md` | Configuring `Commission`, `Spread`, `MarketImpact`, `ShortBorrow`, `CompositeCostModel`, `LiquidityCap`. **Required before any `Engine.run()`.** |
| `skills/05-engine-single-path.md` | Running one backtest via `Engine.run()` and inspecting `BacktestResult`. |
| `skills/06-splitters.md` | Choosing or configuring `WalkForward` or `CombinatorialPurgedCV`; setting purge bars / embargo. |
| `skills/07-multipath.md` | Running `MultiPathEngine` to produce multiple equity paths from CPCV folds. |
| `skills/08-metrics.md` | Computing or interpreting `MetricsReport`, `sharpe_ratio`, `psr`, drawdown stats, or any individual metric. |
| `skills/09-selection.md` | Multiple-testing correction — `dsr`, PSR vs DSR, effective-K via clustering. **Required even for a single trial, because the user will iterate.** |
| `skills/10-simulation.md` | Monte Carlo — generators, adapters, `GeneratorValidator`, `ProbeBattery`, `fingerprint`, `select_generators`, `MonteCarloEngine`, `PathTensorCache`, `run_until_converged`, `run_batch`. |
| `skills/11-registry.md` | Persisting trials via `TrialRegistry` (requires causal-graph artifact). |

---

## Stage → skill map (canonical order)

You build the notebook in this order. The skill column lists what to read **before** announcing the cell for that stage. Detailed required/optional rules are in `PROTOCOL.md §8`.

| # | Stage | Skill(s) |
|---|---|---|
| 1 | Setup + data load | `01-data` |
| 2 | Strategy class | `02-strategy` |
| 3 | Costs + liquidity | `04-costs` |
| 4 | Single-path engine | `05-engine-single-path` |
| 5 | Risk overlay (optional) | `03-risk` |
| 6 | CPCV / multi-path | `06-splitters`, `07-multipath` |
| 7 | Metrics report | `08-metrics` |
| 8 | MC simulation (optional) | `10-simulation` |
| 9 | Selection / DSR | `09-selection` |
| 10 | Trial registry (if committing) | `11-registry` |

---

## Tool notes

These skills are designed to work in any LLM coding tool the user is in (Claude Code, Cursor, others). They are plain markdown — no auto-discovery is required.

- **You cannot assume auto-load.** Every claim above about "read X before Y" is your responsibility to honour. Do not rely on tool-level frontmatter or rules to attach files for you.
- **Citations are mandatory.** Every formula, default value, or API claim in any skill file ends with a `source: backtest/<path>:<line>` citation. When you copy a fact from a skill into a cell announcement or code, do not strip the citation in your reasoning — it is your proof you didn't invent the fact. If a citation looks stale (line numbers off, function gone), say so and re-grep before using the fact.
- **`examples/end_to_end.ipynb` is a fallback, not a source.** Everything you need for a normal session is in the skills — that is the whole point of this folder. Do not read the notebook by default; do not copy from it. The only time you may consult it is when a specific skill leaves you genuinely uncertain about how a primitive is used in practice, and then you read only the matching section, never the full file. If you find yourself wishing you could just paste from the notebook, the right move is to surface the gap in the skill back to the user, not to bypass the skill.

---

## What this folder is **not**

- Not a guide for humans. The audience is the LLM.
- Not a rewrite of the engine docs. It's a translation layer between a strategy spec and the engine's public API.
- Not a place to add new engine features. If the user wants something the engine doesn't have, `PROTOCOL.md §7` is the procedure.
- Not aspirational. Every code example, signature, and equation is grep-verified against the current `backtest/` source at the time the skill was written. If you find drift, surface it; do not silently update.

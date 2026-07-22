# PROTOCOL — How to help the user backtest a strategy

> **Read this file before doing anything else.** Every file in `skills/` assumes you have read this. These rules are not suggestions. They exist because less structured guides produced repeated integration failures.

---

## 0. What you are doing

The user has a backtesting engine (`backtest/`) and a strategy spec (usually a notebook, sometimes a prose description). Your job is to translate that spec into a sequence of notebook cells that exercise the engine **correctly**, **one cell at a time**, with explicit confirmation at every step.

You are not writing a fresh implementation. You are wiring the user's strategy into an engine that already exists. When the spec and the engine disagree, you stop and ask — you do not paper over the disagreement.

---

## 1. The cell loop

Every cell you add to the notebook follows this loop. No exceptions.

```
┌─────────────────────────────────────────────────────────────────┐
│  1. ANNOUNCE  what this cell does, which stage, what it needs   │
│  2. LIST      every mismatch between spec and engine             │
│  3. ASK       the user how to resolve each mismatch              │
│  4. WAIT      for explicit go-ahead. Do not write anything yet.  │
│  5. WRITE     exactly ONE cell. Append to the notebook.          │
│  6. RUN       inline if your environment supports it,            │
│               otherwise ask the user to run it and paste output  │
│  7. INSPECT   validate the output against expected shape/values  │
│  8. ASK       for go-ahead to proceed to the next stage          │
│  9. WAIT      for explicit go-ahead.                             │
└─────────────────────────────────────────────────────────────────┘
```

If you ever find yourself about to write a second cell in the same turn, **stop**. That is a violation.

---

## 2. Before writing any cell — the announcement

Before any code, post a short structured announcement. Use this exact shape:

```
### Cell <N> — <stage name>

**What this cell will do:**
- <one line per concrete action>

**Stage:** <e.g. "3. Single-path Engine">  •  **Skill:** <path to skills/*.md you consulted>

**Depends on cells:** <N-1, N-2, …>  (must be present and run cleanly)

**Engine API used:**
- `<ClassName>(<args>)`  — from `backtest/<path>`
- `<.method(...)>`

**Mismatches with the spec:**
- [if any] <see Section 3 below for categories>
- [if none] None.

**Decisions I need from you:**
1. <question with my recommended default first>
2. <…>

Ready to write this cell once you confirm.
```

You do not write the cell. You wait.

---

## 3. Mismatches — what to look for

Read the strategy spec with these categories in mind. **Every** disagreement gets surfaced, even small ones. Do not silently approximate.

| Category | What to check |
|---|---|
| **Rebalance frequency** | Spec asks for X — is X in the set the engine actually supports? (See `skills/02-strategy.md`.) Cron? Callable? Event-driven? |
| **Data fields** | Spec references `volume`, `fundamentals`, `signals` — does the user have them as a parquet/DataFrame? Will `DataView` expose them? |
| **PIT / lag** | Spec uses fields that need explicit lag (earnings, ratings). Is the user planning to lag them, or relying on the engine to? |
| **Universe** | Time-varying eligibility? Delisted assets? Survivorship-bias risk? |
| **Risk constructs** | Spec wants sector caps, beta-neutrality, factor-neutralization, stop-losses → does `RiskConfig` actually have those fields? (See `skills/03-risk.md`.) |
| **Cost model** | Spec assumes specific bps, impact functional form, borrow tiers. Does the engine's `costs/` cover that, or is it an approximation? |
| **Splitter** | Walk-forward? CPCV? Time-series K-fold? Purge/embargo values? |
| **Metrics** | Spec wants a metric not in `MetricsReport` (information ratio, hit rate, turnover). Is it derivable from `result.returns` or do we need to add it? |
| **MC scope** | Spec wants stress tests / synthetic paths. Which generators apply to this asset class? |
| **Selection** | How many trials? Are they correlated? DSR vs PSR? Effective-K? |
| **Bias risks** | Anything in the spec that smells like lookahead (e.g. "using the full sample to fit"), survivorship (e.g. "current S&P 500 members backtested 20 years"), or selection (e.g. "best of 50 lookbacks") — flag it explicitly. |

For each mismatch you find, your question to the user must follow this shape:

> "The spec says `<X>`. The engine supports `<Y>`. Options:
> (a) Use `<Y>` as-is — closest to spec but differs in `<concrete way>`.
> (b) Extend the engine — separate change, defer this cell.
> (c) Drop this requirement from the backtest — note in the registry.
>
> I recommend `(a)`. Which do you want?"

Always offer the **engine-faithful** default first.

---

## 4. One cell per turn — hard rule

This applies even when:

- The user says "just do all of it" — refuse, restate the rule, ask which stage to start with.
- Two cells "obviously" belong together — they don't. Stages are gates. One cell. One inspection. Go.
- The previous cell ran cleanly and the next is "trivial" — same rule. Announce, ask, wait, write.

Each cell **must** end with at least one line that produces output (a `print`, a bare expression, a `.head()`, a small plot). No silent cells. The user needs something to validate.

---

## 5. After writing a cell — validation

Once the cell is in the notebook:

1. **Run it** if your environment lets you. If not, ask: *"Please run the cell I just added and paste the output (or any error)."*
2. **Inspect the output**. Check:
   - Shape (rows × columns) matches what you predicted in the announcement.
   - No unexpected NaNs.
   - Magnitudes are sane (equity not negative, Sharpe within ±5, vols within 0–100%).
   - No silent fallbacks (e.g. strategy returning all zeros when it should be holding).
3. **State explicitly** what you saw and what passed/failed your expectation.
4. **Ask** for go-ahead before moving on. Do not start the next cell's announcement in the same turn unless the user has already given a standing "continue".

If the output is wrong: do **not** patch the cell silently. Announce the issue, propose a fix as a *new* cell or an edit, ask which.

### 5.1 The cell plot — every cell ends with one (adopted 2026-07-17)

Every code cell ends with a **representative plot**: a figure that captures what happened in that cell
well enough that a reader can grasp it at a glance **without reading the code**. Rules:

- Built with **plotly** (time series / equity / interactive forms) or **seaborn** (statistical forms),
  styled exclusively through `cellplot.py` (`from cellplot import cellplot, PAL, GRAY, DIV, setup`) —
  its palette is the validated dataviz reference instance; never introduce ad-hoc colors.
- Rendered **inline AND saved** via `cellplot(fig, cell_no, slug)` → `plots/cell_NN_slug.png` beside
  the notebook. The plots folder must read as a visual table of contents of the notebook.
- **Notebooks live in `notebooks/<name>/`**, one folder per notebook, `plots/` inside it. Never place
  a notebook loose in the repo root (plots scatter — the reason this rule exists).
- Plot content follows the dataviz method: one axis (never dual), ≤4 series with a legend, title
  states the takeaway (not the chart type), baselines/thresholds drawn as reference lines, only
  computed numbers (no illustrative/fabricated values).

---

## 6. Bias rules — always on, never overrideable

The engine enforces some of these; you enforce all of them. If you catch yourself about to violate one, stop and surface it.

- **No lookahead in `generate_weights`.** Only `data.prices.iloc[:t]` semantics. Never `pct_change()` on the full series, never `.shift(-1)`, never indexing beyond what `DataView` exposes.
- **No survivorship cleanup.** Do not drop assets that have NaNs late in the sample. Universe filtering is the engine's job.
- **No in-sample optimization in the strategy.** Parameter sweeps belong outside `Strategy`, run as separate trials, registered, and deflated with DSR.
- **No silent zero-weight fallback** without telling the user. If the strategy can't produce weights at `t`, the announcement must say so.
- **Warmup must be respected.** The first valid trading date is `panel.dates[warmup_bars]`, not `panel.dates[0]`.
- **Costs are mandatory.** Even though the engine accepts `costs=None`, you do not. Configure `CompositeCostModel` + `LiquidityCap` **before** the first engine run and keep them on for every subsequent run. The only exception is if the user explicitly asks for a gross-vs-net comparison cell, which is opt-in not default.
- **No signal returns on one-sided quotes.** (Data-hygiene rules; born from the #016 bid-only postmortem, alpha log 2026-07-15; full doc `finding-alphas/data-hygiene.md`, code helpers `data_hygiene.py`.) The data cell's announcement must state what each price column IS (trade print / bid / ask / mid / indicative) and when it was knowable. For quote-driven instruments (FX/CFD), returns are computed on the MID (`data_hygiene.load_mid_panel` refuses single-sided files) with spread charged as a separate, TIME-VARYING cost — never a daily median for a strategy that trades at specific clock times.
- **The gross mid-to-mid tripwire is part of every results cell.** One row: per-trade gross at true mids, zero costs (`data_hygiene.gross_tripwire`). Net profitable while that row reads ~0 = the edge is quote mechanics — stop and surface it. For any series of uncertain construction, run `data_hygiene.roll_effective_spread` in the data-validation cell and report the embedded spread next to the instrument's true half-spread.

---

## 7. When the engine doesn't support what the spec wants

You will hit this often. The rule is the same every time:

1. **Do not invent.** No helper functions, no shims, no monkey-patches, no "I'll just compute it manually here". The whole point of using the engine is to avoid re-implementing primitives.
2. **Announce the gap.** Be specific: what the spec asks, what the engine has, what's missing.
3. **Offer the three options** (use closest existing, extend separately, drop).
4. **Wait.**

If the user picks "extend": that is a separate piece of work outside this notebook. Note it as a TODO and proceed with the closest existing primitive for the current cell.

---

## 8. The stages, in order

These are the stages of a backtest. They go in this order. Each one has a matching skill file you must read before writing its cell(s).

| # | Stage | Skill | Required? | Typical cells |
|---|---|---|---|---|
| 1 | Setup + data load | `skills/01-data.md` | Yes | imports; load prices/volume; build `DataPanel` |
| 2 | Strategy class | `skills/02-strategy.md` | Yes | define `Strategy` subclass; instantiate |
| 3 | Costs + liquidity | `skills/04-costs.md` | Yes | configure `CompositeCostModel` + `LiquidityCap` **before** any engine run |
| 4 | Single-path Engine | `skills/05-engine-single-path.md` | Yes | run with costs + liquidity on; inspect equity & metrics |
| 5 | Risk overlay | `skills/03-risk.md` | Optional | `RiskConfig` tweaks; rerun |
| 6 | CPCV / multi-path | `skills/06-splitters.md` + `skills/07-multipath.md` | Recommended | `MultiPathEngine` with CPCV |
| 7 | Metrics report | `skills/08-metrics.md` | Yes | `compute_metrics`, inspect 21 fields |
| 8 | MC simulation | `skills/10-simulation.md` | Optional | generators, validator, fingerprint, adaptive, batch |
| 9 | Selection / DSR | `skills/09-selection.md` | Yes | DSR deflation — required even for a single trial, because the user will iterate |
| 10 | Trial registry | `skills/11-registry.md` | Required if user wants to commit results | `TrialRegistry` with causal graph |

You never skip a "required" stage. You always ask before skipping a "recommended" one.

The first cell of any session does both: **read this PROTOCOL, then read `skills/00-overview.md`**, then announce Cell 1.

---

## 9. What NOT to do

- Do not paste `STRATEGY_GUIDE.md` or `prd.md` into the notebook.
- Do not write helper functions in the notebook that duplicate something `backtest/` already provides.
- Do not refactor the user's strategy spec. Implement what they asked for.
- Do not add "demonstration" cells that aren't part of the spec.
- Do not claim a cell succeeded without inspecting its output.
- Do not claim a formula matches the engine without citing the file + line you verified against.
- Do not auto-correct what looks like a "bug" in the spec — surface it as a mismatch instead.
- Do not bundle multiple stages into one cell to "save time".

---

## 10. The session opener

Your first message in any backtest session **must** include:

1. Confirmation that you've read `PROTOCOL.md` and `skills/00-overview.md`.
2. A one-paragraph summary of the strategy as you understood it from the spec.
3. A proposed stage order (which of the 10 stages above will be in this notebook, in order).
4. A request for confirmation before announcing Cell 1.

You do **not** front-load every mismatch in the opener. Mismatches surface **per cell**, in the announcement (§2), at the moment they would actually affect the code you're about to write. This keeps each round focused on one decision instead of forcing the user to triage a wall of issues before any code exists.

Only after the user confirms do you start the cell loop.

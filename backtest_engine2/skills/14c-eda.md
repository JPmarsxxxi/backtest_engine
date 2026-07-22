# Skill 14c — Hypothesis-Driven EDA (alpha pipeline Stage 2.7)

> **Read `PROTOCOL.md`, `00-overview.md`, `12-alpha-overview.md` first.** Cell-loop and announcement rules apply to every EDA cell, exactly as they do to engine stages.

> Stage 2.7. Between `data_clamp` (2.5) and `signal_construction` (3). **Methodology only — no module.** The job is to *understand the data well enough to either justify or replace the hypothesis* before any signal is written. The job is **not** to run every diagnostic that exists.

---

## When to load

- Whenever a hunt starts with an **idea** (paper, hunch, inherited spec) and you need to verify the idea's claims hold in *this specific data*, OR
- Whenever you're starting from raw data with no hypothesis and need to **derive** one from observation.

Skip only if the hypothesis is already verified on identical data in a prior winners/ alpha card and you're re-running an evaluation, not exploring.

---

## The principle

EDA in this pipeline is **hypothesis-driven**, not exhaustive-by-default. Ch. 6 says *"data itself can inspire alpha ideas"* — but the corollary is **the data also kills bad ideas**. Both directions are legitimate uses of EDA. What is **not** legitimate is running 20 generic diagnostics and hoping a pattern jumps out — that's how spurious findings get believed.

Two rules:

1. **Every EDA cell tests a specific claim.** Either a sub-claim of the stated hypothesis, or a candidate sub-claim you would need to be true for an alternative hypothesis to work. State the claim in the cell announcement before writing any code.
2. **Exhaustive within the hypothesis, lean outside it.** If the hypothesis says "the spread mean-reverts when Hurst < 0.5," then exhaust the tools to verify mean-reversion (ADF, Hurst, OU half-life, variance ratio, ACF) on the spread. Do **not** also test calendar effects, regime switching, and PCA on returns unless you have a reason — that's the overkill the user explicitly rejected.

---

## Step 1 — decompose the hypothesis into testable sub-claims

Before any cell, write out the hypothesis and break it into the **falsifiable** atomic claims it depends on. Every claim becomes one or more EDA cells.

**Example.** The hypothesis from a real spec:

> "When a cryptocurrency pair's spread enters an anti-persistent regime (local Hurst H < 0.5), the spread is modelled as a fast mean-reverting discrete O-U process; calibrating profit-taking and stop-loss thresholds from the estimated O-U parameters maximises the Sharpe of the exit rule."

Decomposes into:

| # | Sub-claim | Falsifiable by |
|---|---|---|
| C1 | Pair spreads in the universe *can be* anti-persistent (H < 0.5 occurs non-trivially) | Distribution of rolling H across pairs / time |
| C2 | When H < 0.5, the spread *does* mean-revert (not random walk, not trending) | ADF on H<0.5 windows vs H≥0.5 windows; OU half-life finite and short |
| C3 | The selected pairs *are* cointegrated under the hedge ratio `b` | Engle-Granger ADF on OLS residuals; share of pairs with p < 0.05 |
| C4 | The 1-2σ entry band is *populated* often enough to generate trades | Frequency of `1σ < |s−m| < 2σ` while H<0.5 |
| C5 | O-U parameter estimates are *stable* enough to be useful (φ ∈ (0,1) typically) | Distribution of φ̂ across entry events; share with stationary OU |
| C6 | The 72h vertical barrier is *not the dominant exit* (would mean the rule is a stoploss-by-time, not by-OU) | Among simulated exits, distribution over the three exit types |

Each row → one or more announced cells. If C1-C3 fail in your data, the hypothesis is dead — there is no signal to construct, and you stop the hunt before writing `signal_construction.py`.

---

## Step 2 — pick the right tools per claim type

Don't memorise; recognise what kind of claim you have, then pick from this menu (or a tool not listed if it fits better).

| Claim type | Standard tools |
|---|---|
| "X is stationary / mean-reverting" | ADF, KPSS, Phillips-Perron; Hurst exponent; variance-ratio test; OU half-life from AR(1) |
| "X has long memory / momentum" | Hurst (>0.5); ACF/PACF; rescaled-range |
| "X and Y are cointegrated" | Engle-Granger (OLS residual ADF); Johansen for >2 series |
| "X has fat tails" | sample kurtosis; tail-index Hill estimator; Q-Q vs normal |
| "X has heteroskedasticity / vol clustering" | ARCH-LM; rolling-std plots; ACF of squared returns |
| "X has regime changes" | rolling mean/std plots; Markov-switching; CUSUM; Chow test |
| "X predicts forward Y" | IC (Pearson/Spearman) at horizons τ ∈ {1, 5, 20}; lead-lag cross-correlation |
| "Effect is concentrated in time-of-day / day-of-week" | groupby calendar column; ANOVA across groups |
| "X is monotone in Y" | quantile sorts of Y, plot mean(X) by quantile bin |
| "Distribution differs between groups" | KS test, Mann-Whitney, Welch's t |
| "Most variance lives in K factors" | PCA on returns; scree plot |
| "Outliers exist beyond `data_clamp`" | run-length of clipped bars; gap detection; flag-count summaries |

For tools not in this table (e.g. wavelet decompositions, fractal dimension, copulas), use them only if you can articulate **which claim** they're testing. If you can't, you're running diagnostics for the sake of running diagnostics.

---

## Step 3 — execute one cell per sub-claim, following the cell loop

Each EDA cell follows PROTOCOL §1 exactly. The announcement template adapts to:

```
### Cell <N> — EDA: <claim>

**Sub-claim being tested:** C<n>: "<exact text from the decomposition table>"

**Why it matters to the hypothesis:** <one sentence — what dies if this fails>

**Test(s) used:** <name>(s), e.g. "ADF on rolling 168h windows of spread BTC-ETH"

**Decision rule before running:** "I will consider C<n> **supported** if <criterion>; **refuted** if <criterion>; **inconclusive** if <criterion>." (Pre-commit so you can't post-hoc rationalise.)

**Engine/library APIs used:**
- `statsmodels.tsa.stattools.adfuller(...)` (or whichever)

**Decisions I need from you:** <if any — e.g. "Which window length?">

Ready to write this cell once you confirm.
```

After running:

1. State the result against the pre-committed decision rule. Don't soften the verdict.
2. If **refuted** for an atomic claim that the hypothesis cannot survive without, **stop the hunt and report**. Do not proceed to `signal_construction`. Either propose a revised hypothesis or end the hunt here.
3. If **supported**, move to the next sub-claim.
4. If **inconclusive**, propose either (a) a tighter test, or (b) accept and flag in the alpha card.

---

## Step 4 — the hypothesis-update cell

After all sub-claims have been tested, write one **non-code** cell summarising:

```
### Cell <N> — Hypothesis update after EDA

**Original hypothesis:** <verbatim or near-verbatim>

**Sub-claims tested:**
- C1: supported / refuted / inconclusive — <one-line evidence>
- C2: ...
- ...

**Net status:** <one of>
- (a) Original hypothesis stands as-is → proceed to `signal_construction` with the spec's logic.
- (b) Hypothesis partially holds → narrow to <subset>, e.g. "only on the top-10 by ADF p-value", "only between H_window 5-7 days, not 3 or 10". Proceed with the narrowed version.
- (c) Hypothesis refuted in this data → either propose a revised hypothesis grounded in observations, or end the hunt and report the negative result (negative results are useful — log them).

**Observations not in the original hypothesis** (the "data inspires ideas" path): list any patterns you noticed in EDA that the spec doesn't mention. These are candidate alphas for *future* hunts; do not chase them in this one (that's selection bias).

**Decisions I need from you:** confirm the net status before writing `signal_construction`.
```

This cell is the bridge between "we explored the data" and "we now write code that bakes the hypothesis in." Without it, EDA findings rot in the conversation and `signal_construction` happens by inertia.

---

## What does NOT belong here

- **Validating signal *output* quality** — that's after signal_construction. Use IC / dispersion / autocorr of the *signal*, not the data, and put it in evaluate or a separate diagnostic cell after Stage 3.
- **Backtesting candidate signals** — running a quick PnL check on a 1-line signal is `evaluate.run`, not EDA. EDA is *pre-signal*. If you're computing PnL, you've left EDA.
- **Generic plots with no claim attached** — a price chart "for visual inspection" with no follow-up question is not EDA, it's looking at pretty pictures. Either it tests a claim or it doesn't get a cell.
- **Tools you can't interpret** — if you run a Hurst exponent but cannot say what H = 0.42 means for your hypothesis, you shouldn't have run it. Tool first → interpretation rule second → run third.

---

## How exhaustive is "exhaustive"?

Exhaustive **within** the hypothesis. If the hypothesis claims mean-reversion, you should be done only when *every reasonable test of mean-reversion* in this data has been run and produced a consistent picture. Inconsistencies between tests (ADF says stationary, Hurst says momentum) are themselves observations — investigate, don't average them away.

Exhaustive does **not** mean "every diagnostic in the textbook." A clean three-test confirmation of mean-reversion (ADF + Hurst + OU half-life) on the pair spreads is more useful than running 30 tests including PCA, GARCH, copula fits, and seasonality decompositions that the hypothesis doesn't depend on. The latter creates noise and risks finding spurious patterns you'll be tempted to chase.

---

## Tool libraries you can use freely in EDA cells

These are standard Python. No new modules needed in `backtest/`.

- `numpy`, `pandas` — descriptive stats, rolling windows, groupby.
- `scipy.stats` — KS, Mann-Whitney, t-tests, kurtosis, skew.
- `statsmodels.tsa.stattools` — `adfuller`, `kpss`, `coint`.
- `statsmodels.tsa.vector_ar.vecm` — Johansen cointegration (`coint_johansen`).
- `statsmodels.tsa.api` — VAR, ARCH-LM (`het_arch`).
- `matplotlib` — distributions, rolling stats, ACF plots.
- Hand-rolled Hurst / variance-ratio / OU-half-life (~10 lines each, often clearer than pulling a library).

If you find yourself needing a heavy dependency (arch, ruptures, deepgraph, etc.) for one EDA cell, **stop and ask** — the cost of the dep usually isn't worth it for a single observation.

---

## Connection to `signal_construction` (Stage 3)

The output of EDA is a **refined, evidence-backed hypothesis** plus a list of decisions:
- Which lookbacks to use (informed by ACF / half-life — not pulled from the spec).
- Which subset of the universe to act on (informed by cointegration screen — not "top 50 by mcap" blindly).
- Which entry band / threshold (informed by the actual deviation distribution — not "1-2 σ" from the spec without checking).
- Whether to use the original signal logic at all (if EDA refuted core claims, you're writing a *different* signal).

Carry these into the `signal_construction` cell announcement (PROTOCOL §2 "Mismatches with the spec"). Every parameter that EDA touched should be cited with a `source:` pointing to the EDA cell that picked it.

---

## Minimal valid EDA section (sketch)

For a mean-reversion-on-pair-spread hunt, the minimum that would justify proceeding:

1. **C1 cell** — distribution of rolling Hurst across pairs / time. Verdict: "X% of bars satisfy H<0.5 across the median pair; threshold is hit often enough to matter."
2. **C2 cell** — ADF p-value distribution on H<0.5 windows vs H≥0.5 windows. Verdict: "ADF rejects unit root in Y% of H<0.5 windows vs Z% in H≥0.5 — mean-reversion is regime-specific."
3. **C3 cell** — Engle-Granger cointegration on all pairs at current hedge ratio. Verdict: "K pairs cointegrated at p < 0.05; this is our trade-eligible universe."
4. **C4 cell** — frequency of `1σ < |s−m| < 2σ` AND H<0.5 simultaneously. Verdict: "Entry conditions fire on N% of bars in eligible pairs — enough/not enough for trade flow."
5. **C5 cell** — OU half-life distribution from AR(1) fits on H<0.5 spread windows. Verdict: "Median half-life = H hours; consistent with the 72h vertical barrier or not."
6. **Cell N — hypothesis update** — net verdict, decisions for `signal_construction`.

Anything else (PCA on returns, calendar effects, vol clustering) only if a *specific claim* in the hypothesis depends on it. Otherwise out.

---

## What NOT to do

- **Do not** run EDA as a "show what you can do" exercise. Each cell answers a question that, if answered the wrong way, kills or modifies the hypothesis.
- **Do not** present a finding without a pre-committed decision rule — that's p-hacking.
- **Do not** silently ignore refuted sub-claims and proceed to `signal_construction` anyway. PROTOCOL §3 / §7 apply: surface, ask, decide.
- **Do not** chase observations that *aren't* part of the stated hypothesis in the same hunt. Log them as "candidate hypotheses for future hunts" in the hypothesis-update cell.
- **Do not** build a generic `eda.py` module. Each hunt's EDA is different; the methodology is the same. The skill is the asset, not the code.

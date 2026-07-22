# Skill 09 — Selection / DSR

> **Read `PROTOCOL.md` and `skills/00-overview.md` first.** Cell-loop and announcement rules in PROTOCOL apply.

> Stage 7 (metrics) must be done. This skill applies the multiple-testing correction every trial needs — even when the user only ran one. The trial registry (`skills/11-registry.md`) is where K comes from in serious work.

---

## When to load

- **Stage 9, mandatory**: deflating the trial's Sharpe against the count of trials the user has actually run (`PROTOCOL.md §6`). Required even for K=1 because users iterate.
- Any cell that imports `dsr`, `effective_k`, `sidak_alpha`, `bonferroni_alpha`, or `expected_max_sr`.
- Any cell where the user is interpreting PSR (`skills/08-metrics.md §4`) as evidence of edge — PSR alone is not enough; DSR is.

---

## What you'll find here

1. The problem this stage solves — why a single backtest's PSR can lie.
2. `dsr(returns, K, sr_variance=None)` — the main entry point.
3. `expected_max_sr(K, var_sr)` — the False Strategy Theorem.
4. `effective_k(trial_returns, threshold, method)` — correlation-aware K.
5. Sidak: `sidak_alpha`, `sidak_pvalue`.
6. Bonferroni: `bonferroni_alpha`, `bonferroni_pvalue`.
7. The K=1 case — what DSR collapses to and why you still need it.
8. Where K comes from in practice.
9. Common mismatches.
10. Minimal valid cell.
11. Validation checklist.
12. Anti-patterns.

---

## 1. The multiple-testing problem in one paragraph

You ran one strategy and got SR = 1.5. PSR (`skills/08-metrics.md §4.1`) says "the true SR is probably above 0 with confidence 0.96". But you didn't run *one* strategy — you tried 50 parameter combinations and reported the best. Under the null hypothesis (all 50 are pure luck), the expected maximum-of-50 sample Sharpes is *not* zero — it's `E[max SR]`, a positive number that grows with K. PSR vs 0 is the wrong threshold; the right threshold is `E[max SR] given K trials and the per-trial SR variance`. That's what DSR computes. The López de Prado 2014 *Deflated Sharpe Ratio* paper formalises the recipe; the engine implements it.

---

## 2. `dsr(returns, K, sr_variance=None)`

```python
dsr(returns, K: int, sr_variance: float | None = None) -> float
```
`source: backtest/selection/dsr.py:32–65`

Returns `P(true SR > E[max SR | K, var_sr] | sample)` — the probability that this strategy's true Sharpe beats the *expected luck-maximum* across K trials. In `[0, 1]` like PSR.

### 2.1 The formula

```python
sr_pb = r.mean() / r.std(ddof=1)                    # per-bar Sharpe
var_term = sharpe_var_term(r, sr_pb)                # paper Eq. 2 (skill 08 §3.2)
if sr_variance is None:
    sr_variance = var_term / T                       # asymptotic single-trial SR variance
sr_0 = expected_max_sr(K, sr_variance)               # §3 below
z = (sr_pb - sr_0) * sqrt(T - 1) / sqrt(var_term)
return Φ(z)
```
`source: backtest/selection/dsr.py:48–65`

The difference vs PSR (`skills/08-metrics.md §4.1`): the threshold `sr_0` is no longer `sr_star_pb = 0`; it's the expected luck-max across K trials. Everything else is the same z-score structure.

### 2.2 Edge cases

| Condition | Return | Source |
|---|---|---|
| `K < 1` | `ValueError("K must be >= 1")` | `dsr.py:46–47` |
| `len(r) < 2` after dropna | `NaN` | `dsr.py:49–50` |
| `r.std() < 1e-12` (constant returns) | `NaN` | `dsr.py:51–53` |
| `var_term <= 0` (degenerate skew/kurt combo) | `NaN` | `dsr.py:56–58` |

`returns` accepts a `pd.Series` (NaN-dropped) or array (finite-filtered). `source: backtest/selection/dsr.py:68–72`

### 2.3 `sr_variance` — when to override

The default uses **asymptotic single-trial variance** `var_term / T` — derived from this one sample's skew, kurtosis, and length. This is what López de Prado's paper recommends when you don't have a population of observed trials.

Override `sr_variance` with the **sample variance of per-trial Sharpes** across actual trials when you have them — e.g., `np.var([rep.sharpe for rep in per_path_reports], ddof=1) / ann_factor`. Use the per-bar variance, not annualized. `source: backtest/selection/dsr.py:60–61, selection_test.py:123–127`

### 2.4 No risk-free rate — paper-faithful

`dsr` does not subtract `rf/ann_factor` from returns. This matches LdP 2014: the DSR formula uses `SR̂ = μ/σ` from raw returns, not excess returns. `source: backtest/selection/dsr.py:54`

So if you've computed your *headline* Sharpe via `compute_metrics(..., rf=0.04)`, that number does not match the `sr_pb · √ann_factor` implied by DSR. Both are valid; they answer different questions ("how much excess return per unit risk" vs "is this Sharpe distinguishable from K-trial luck"). Report both if the user cares about both.

---

## 3. `expected_max_sr(K, var_sr)` — the False Strategy Theorem

```python
expected_max_sr(K: int, var_sr: float) -> float
```
`source: backtest/selection/dsr.py:14–29`

Closed-form expected value of the maximum-of-K independent Sharpe ratios, given per-trial SR variance `var_sr`:

```
E[max SR] = sqrt(var_sr) · [ (1 - γ) · Φ⁻¹(1 - 1/K) + γ · Φ⁻¹(1 - 1/(K·e)) ]
```

where `γ = 0.5772156649015329` is the **Euler-Mascheroni constant** (`EULER_MASCHERONI` is exported, in case you need it explicitly). `source: backtest/selection/dsr.py:11`

### 3.1 Special cases

- `K == 1`: returns `0.0`. One trial has no "maximum-of-others" to dodge. `dsr.py:24–25`
- `var_sr == 0`: returns `0.0`. No variance means no luck inflation. `dsr.py:24`
- `K == 0`: `ValueError`. `dsr.py:20–21`
- `var_sr < 0`: `ValueError`. `dsr.py:22–23`

### 3.2 Properties locked in by tests

`source: backtest/selection/selection_test.py:68–95`

- Monotone increasing in K: `E[max SR | K=2] < E[max SR | K=10] < E[max SR | K=100]`.
- Monotone increasing in `var_sr`: more variance → easier to get lucky → higher luck-max.

### 3.3 When to call it directly

You usually don't — `dsr` calls it internally. Reach for `expected_max_sr` only when:
- You want to *display* the threshold the strategy must beat (e.g. "DSR deflates against expected luck-max of 0.42").
- You're doing a custom calculation (e.g. modifying `sr_variance` by hand and want both numbers).

---

## 4. `effective_k(trial_returns, threshold=0.5, method="single")` — correlation-aware K

```python
effective_k(trial_returns: pd.DataFrame, threshold: float = 0.5, method: str = "single") -> int
```
`source: backtest/selection/clustering.py:9–41`

The "K" in DSR is supposed to be the number of *independent* trials. If you ran 50 strategies but 40 of them are minor variants of the same idea (returns highly correlated), the effective K is closer to ~10. Treating K=50 over-deflates; using `effective_k` is the López de Prado & Lewis 2019 recipe.

### 4.1 The algorithm

1. `trial_returns.corr()` — pairwise correlation matrix across trials (columns = trials).
2. Convert correlation to distance: `d = sqrt(0.5 · (1 - corr))`. Two perfectly correlated trials → `d = 0`; uncorrelated → `d ≈ 0.707`; anti-correlated → `d = 1`. `clustering.py:33`
3. Hierarchical linkage (`scipy.cluster.hierarchy.linkage`) using the supplied `method`.
4. Cut at the distance corresponding to the supplied correlation `threshold`: `dist_threshold = sqrt(0.5 · (1 - threshold))`. `clustering.py:39`
5. Number of resulting clusters = effective K.

### 4.2 Args

| Arg | Default | Meaning |
|---|---|---|
| `trial_returns` | required | `pd.DataFrame[date × trial]`. Each column = one trial's returns series. Typically `MultiPathResult.path_returns` or trials pulled from the registry. |
| `threshold` | `0.5` | Correlation **above** which trials are considered the same cluster. `0.5` means: anything correlating > 0.5 collapses together. `0.0` (everything in one cluster) and `1.0` (each trial separate) both accepted; in `[0, 1]`. |
| `method` | `"single"` | scipy linkage method. `"single"`, `"average"`, `"complete"`, `"ward"` — usually leave default. |

### 4.3 Behavior at degenerate inputs

- Empty DataFrame → returns `0`. `clustering.py:26–28, selection_test.py:168–170`
- Single column → returns `1`. `selection_test.py:163–165`
- All columns identical → returns `1`. `selection_test.py:136–140`
- All columns independent → returns ≈ n_cols (at threshold 0.5). `selection_test.py:143–150`
- Two clear clusters of identical-noise series → returns `2`. `selection_test.py:153–160`

### 4.4 Threshold validation

`ValueError("threshold must be in [0, 1]")` for any value outside the closed interval. `source: backtest/selection/clustering.py:24–25`

### 4.5 Feeding it into DSR

```python
K_eff = effective_k(mp.path_returns, threshold=0.5)
deflated = dsr(strategy_returns, K=K_eff)
```

For trials registered in the `TrialRegistry` (skill 11), the registry exposes a `dsr_for(..., method="effective", threshold=...)` helper that computes `K_eff` from logged trial returns and applies it automatically.

---

## 5. Sidak corrections

```python
sidak_alpha(alpha: float, K: int) -> float          # per-trial threshold for FWER alpha
sidak_pvalue(p: float, K: int) -> float             # corrected p-value
```
`source: backtest/selection/corrections.py:19–30`

Family-wise error rate (FWER) correction assuming **independent** trials:

- `sidak_alpha(α, K) = 1 - (1 - α)^(1/K)`
- `sidak_pvalue(p, K) = 1 - (1 - p)^K`

Round-trip: `sidak_pvalue(sidak_alpha(α, K), K) == α` exactly. `selection_test.py:33–35`

Validation: `α ∈ (0, 1)`, `K ≥ 1`, `p ∈ [0, 1]`. `corrections.py:4–16, selection_test.py:51–65`

### 5.1 When to prefer Sidak over DSR

DSR is the López de Prado-paper recipe; it's what PROTOCOL §6 mandates. Sidak / Bonferroni are general-purpose FWER corrections you'd see in any statistics textbook. Use Sidak / Bonferroni when:

- You're reporting a Sharpe **p-value** (from a hypothesis test) and need a corrected version, not a deflated Sharpe.
- You want a sanity-check number alongside DSR.
- The trials are **independent**, which is rarely true for parameter sweeps — `effective_k` is usually more honest.

---

## 6. Bonferroni corrections

```python
bonferroni_alpha(alpha: float, K: int) -> float     # = α / K
bonferroni_pvalue(p: float, K: int) -> float        # = min(p · K, 1.0)
```
`source: backtest/selection/corrections.py:33–44`

Strictly more conservative than Sidak (smaller per-trial threshold; larger corrected p-value). The `min(..., 1.0)` cap matters — `bonferroni_pvalue(0.5, K=10) = 1.0`, not `5.0`. `selection_test.py:46–48`

For most practical K (say, K ≤ 100), Sidak and Bonferroni agree to 3 decimal places. Bonferroni is the conservative bound; Sidak is exact under independence.

---

## 7. The K=1 case — why DSR is still required

PROTOCOL §6 says DSR is mandatory **even at K=1**. The reason is in the math.

At `K=1`:
- `expected_max_sr(K=1, var_sr) = 0.0` exactly. `dsr.py:24–25`
- The DSR z-score reduces to `(sr_pb - 0) · sqrt(T-1) / sqrt(var_term)`.
- That's exactly the PSR z-score against `sr_star = 0`.

So `dsr(r, K=1) == psr(r, sr_star=0)`. Locked in by `test_dsr_k1_matches_psr_zero`. `source: backtest/selection/selection_test.py:104–106`

Why is the K=1 result still required then? Two reasons:

1. **You will iterate.** As soon as the user changes a parameter and re-runs, K increases. If DSR was not the headline number from the start, the comparison across attempts is misleading. Forcing DSR at K=1 keeps the unit consistent across the session.
2. **It surfaces the K decision early.** Even at K=1, the user has to think about: "Is this really the only trial I'll consider?" The act of computing DSR is itself the prompt. Skipping it because "K=1 is trivial" is the failure mode the protocol prevents.

---

## 8. Where K comes from in practice

Three sources, in increasing rigor:

1. **The user reports it.** ("I tried 5 lookbacks.") Use that K. Fragile; users under-count by orders of magnitude.
2. **The trial registry**. `TrialRegistry.k(family=...)` returns the count of logged trials in a strategy family (skill 11). This is the canonical source — every executed trial registers itself; under-counting requires *deleting* trials, which is a different protocol failure.
3. **`effective_k(trial_returns_df)`**. The correlation-corrected K. Always smaller-or-equal-to the count, and more honest. The registry can return this via `dsr_for(..., method="effective")`.

**Recommendation for the cell:** if the registry is in use, pull `K = reg.k(family=...)` (or its effective variant) and pass that. If not, pull K from the user with an explicit count, and warn that effective-K is unavailable without a returns table for all trials.

---

## 9. Common mismatches — surface these by default

Use the question shape from `PROTOCOL.md §3`. Engine-faithful default first.

| Spec says | Engine has | Question to ask |
|---|---|---|
| "Only one strategy was tested, so DSR isn't needed" | `PROTOCOL §6` mandates DSR even at K=1. At K=1, `dsr` equals `psr(r, sr_star=0)` exactly. | Run DSR at K=1 and report it — the user has to think about whether K will grow. Surface the "you'll iterate" framing from §7. |
| "I ran 50 parameter combinations" (no registry) | DSR needs K. With 50 reported but no trials logged, you can't compute effective-K. | (a) Use `dsr(r, K=50)` — over-deflates if trials are correlated. (b) Persist all 50 trials in the registry and use `effective_k`. (c) Accept the user's number with a warning. Recommend (b). |
| "Use Sharpe p-value with Sidak correction" | `sidak_pvalue(p, K)` available. But for a Sharpe with skew/kurt-aware variance, the "p-value" is `1 - psr(...)` (skill 08 §4.1). Sidak applies to *that* p-value. | Confirm: do they want corrected p-value or deflated Sharpe? PROTOCOL mandates DSR; Sidak/Bonferroni are sanity checks. |
| "Use bootstrap to compute DSR" | The engine's DSR is closed-form via the asymptotic SR distribution + False Strategy Theorem — same recipe as the paper. | Out of scope. Bootstrap DSR is not provided. |
| "Use K from the number of CPCV paths" | CPCV produces N paths from one strategy. That's not "K independent trials" — it's one trial measured N ways. K=1, not K=N. | Surface as a mismatch. The paths inform the **per-trial SR variance** (override `sr_variance` with the empirical variance across paths); K still comes from the trial count. |
| "Different correlation threshold for effective-K" | `effective_k(..., threshold=...)`. Default `0.5`. | Confirm the chosen threshold. Higher threshold → more clusters → higher K (more conservative). Recommend keeping the same threshold across trials within a family. |
| "Effective-K with `ward` linkage instead of single" | `effective_k(..., method="ward")`. Any scipy linkage method accepted. | Acceptable. Default is `"single"` (closest-pair) which is sensitive to chaining; `"average"` is often more robust. Keep the choice consistent within a family. |
| "DSR against a non-zero Sharpe threshold" | Not supported. DSR always uses `sr_0 = expected_max_sr(K, var_sr)`. There's no `sr_star` arg. | Out of scope. For "PSR vs sr_star with K-adjusted threshold," compute by hand: `psr(r, sr_star=sr_target + expected_max_sr(K, var_sr) * sqrt(ann_factor))`. Or extend the engine. |
| "Risk-free rate of 4% inside DSR" | The DSR paper (LdP 2014) computes `SR̂ = μ/σ` from raw returns — no `rf` adjustment. The engine's `dsr` is faithful to that. | Confirm with the user: DSR and rf-adjusted Sharpe (`compute_metrics(..., rf=0.04)`) are different concepts. Report both if needed; do not try to merge them. |
| "FDR (Benjamini-Hochberg) instead of FWER" | Not provided. Only Sidak / Bonferroni / DSR. | Out of scope. Recommend DSR (which deflates expected luck-max, not p-values). |

---

## 10. Minimal valid cell — Stage 9, DSR

```python
from backtest.selection import dsr, effective_k, expected_max_sr

# K source — choose ONE:
K = 1                                              # only trial run this session
# K = reg.k(family="momentum_xs")                  # from trial registry (skill 11)
# K = effective_k(trial_returns_df, threshold=0.5) # correlation-corrected K

deflated = dsr(result.returns, K=K)

# Visible artifact for validation.
from backtest.metrics.core import sharpe_var_term
import numpy as np
r = result.returns.dropna().to_numpy()
T = len(r)
sr_pb = r.mean() / r.std(ddof=1)
var_term = sharpe_var_term(r, sr_pb)
luck_max_pb = expected_max_sr(K, var_term / T)
luck_max_ann = luck_max_pb * np.sqrt(252)

print(f"Trials (K):            {K}")
print(f"Realized Sharpe (ann.):{sr_pb * np.sqrt(252):.3f}")
print(f"Luck-max threshold:    {luck_max_ann:.3f}  (E[max SR] given K and per-trial variance)")
print(f"DSR:                   {deflated:.3f}    {'significant' if deflated > 0.95 else 'NOT significant'} at 95%")
print(f"  (compare to undeflated PSR: {result.returns.dropna().pipe(lambda r: __import__('backtest.metrics', fromlist=['psr']).psr(r)):.3f})")
```

For multi-path runs (skill 07), iterate per path and aggregate:

```python
deflated = [dsr(res.path_returns[c].dropna(), K=K) for c in res.path_returns.columns]
print(f"DSR across {res.n_paths} paths: median={np.median(deflated):.3f}, "
      f"min={np.min(deflated):.3f}, max={np.max(deflated):.3f}")
```

Do **not** persist the trial in this cell. Trial registry is Stage 10 (`skills/11-registry.md`); it requires a causal-graph image path and is a separate decision.

---

## 11. Validation checklist (after the cell runs)

- [ ] **`K`** matches what the user actually intends as their trial count (manual count, registry count, or effective-K). If you guessed, ask.
- [ ] **`deflated`** is in `[0, 1]`. NaN means the SR couldn't be computed (constant returns or `< 2` obs) — fix Stage 4 / 6 first.
- [ ] **`deflated < psr(returns)`** for K > 1 (DSR is more conservative than PSR by construction).
- [ ] **`deflated == psr(returns, sr_star=0)`** at K=1 (within numerical noise).
- [ ] **`luck_max_ann`** is positive and grows with K. If you printed it (per §10's cell), the magnitude should feel reasonable: K=10 with daily data → typically luck-max around 0.3–0.6 in annualized terms.
- [ ] **Realized Sharpe > luck_max** is the basic "did we beat luck?" question. If realized < luck_max, DSR will be < 0.5. Surface to the user.
- [ ] If `K` came from `effective_k`: confirm the threshold the user picked and that the trials table actually has > 1 column (or `K` will be 1 trivially).

If any fails: state which. Do not proceed to Stage 10 (registry) with a NaN DSR.

---

## 12. What NOT to do

- **Do not skip DSR at K=1.** PROTOCOL §6 mandates it. The math collapses to PSR-vs-zero, but the *act* of computing it surfaces the K decision and keeps units consistent as the user iterates.
- **Do not report PSR as "the significance" without DSR.** PSR vs 0 is a single-trial probability. Anyone who looks at multiple parameter values has K > 1 and must deflate.
- **Do not use the count of CPCV paths as `K`.** Paths from one strategy are one trial measured many ways. K is the number of *distinct strategies/parameter combinations tried*, not the number of equity curves. See §9 mismatches.
- **Do not assume Sidak == Bonferroni** for small K. They diverge for `K > ~10`. Sidak is exact under independence; Bonferroni is the conservative bound. For real trial counts (parameter sweeps), neither is right — use DSR with effective-K instead.
- **Do not override `sr_variance` with the annualized variance.** The function expects **per-bar** variance. Divide your annualized number by `ann_factor` before passing.
- **Do not pass `K = -1` or `K = 0`.** `dsr`, `expected_max_sr`, `sidak_*`, `bonferroni_*` all raise `ValueError`. K is a count, not a sentinel.
- **Do not feed `effective_k` a DataFrame where columns are uncorrelated random walks and expect "true" effective K.** With pure noise at threshold 0.5, you get ≈ n_cols (each is its own cluster). That's correct behavior but doesn't mean the trials were "really independent" in the strategy sense — they were independent **noise series**. Effective-K trusts the returns; it doesn't audit the strategy design.
- **Do not use `expected_max_sr(K, var_sr)` with an annualized `var_sr`.** Same per-bar convention as `dsr`. Annualized variance would inflate the luck-max threshold by `ann_factor`.
- **Do not report DSR rounded to 2 decimals as "0.95 → significant".** PSR/DSR are probabilities, and the decision boundary at 0.95 is conventional, not derived. Tell the user the raw value and let them apply their own threshold.
- **Do not change K mid-comparison.** If you reported DSR at K=10 for one variant and K=20 for another, you can't directly compare them — the deflation threshold differs. Compute both at the same K (the higher one, or the registry's running count).
- **Do not feed `result.returns` to `dsr` for a multi-path result.** Use per-path returns from `MultiPathResult.path_returns` and either pick a path or aggregate. The single-path `dsr` does not understand the multi-path structure.
- **Do not interpret DSR as a hit/miss flag.** It's a continuous probability. "DSR = 0.92" means "92% confident this beats expected luck-max" — not "fails at 95% threshold". Surface the number, let the user decide.

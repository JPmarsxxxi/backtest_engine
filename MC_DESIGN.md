# Monte Carlo Simulation Layer — Design

Status: implemented in `backtest/simulation/` (10 chunks, 189 tests passing as of 2026-05-12).
Reference: López de Prado, *Advances in Financial Machine Learning*, Ch. 16.6.

## Goal

Add a Monte Carlo layer on top of the existing generic backtest engine so any
strategy can be evaluated across thousands of synthetic price paths. Engine
stays generic and untouched; MC plugs in as a wrapper.

## Core principles

1. **Engine stays generic.** It consumes price/return arrays — it doesn't know
   or care whether they're real or synthetic.
2. **Path-generator validity is strategy-dependent.** The generator must
   preserve the statistical property the strategy is trying to exploit, or the
   MC tells you nothing. (E.g. running a momentum strategy on IID Gaussian
   paths returns Sharpe ≈ 0 by construction — meaningless.)
3. **Don't auto-pick a single generator. Fingerprint the strategy, run several,
   report sensitivity.** A strategy that survives only one generator is fragile;
   one that survives multiple is robust. The matrix is more honest than any
   single number.
4. **Permutation null is non-optional.** Any strategy that "works" on shuffled
   returns has a look-ahead leak or a bug. Always run it.

## Architecture

```
┌──────────────────────────────────────────────────┐
│ mc_run(strategy, generators="auto"|[list], N)    │
│   ├─ fingerprint(strategy)  →  score vector      │
│   ├─ select generators                           │
│   ├─ for each generator:                         │
│   │     paths = generator.sample(N, seed)        │
│   │     results = engine.run(strategy, paths)    │
│   │     metrics = aggregate(results)             │
│   └─ sensitivity_table(metrics_per_generator)    │
└──────────────────────────────────────────────────┘
```

### Path generators (pluggable)

Common interface:

```python
class PathGenerator:
    def sample(self, n_paths: int, n_steps: int, n_assets: int,
               seed: int) -> np.ndarray:  # (n_paths, n_steps, n_assets)
        ...
```

| Generator | Preserves | Use as |
|---|---|---|
| `GaussianGenerator` | μ, σ only | baseline / null |
| `BlockBootstrapGenerator` | autocorr, vol clustering, tails | **cheap default** |
| `GARCHGenerator` | explicit vol dynamics | vol-sensitive strategies |
| `JumpDiffusionGenerator` | fat tails / jumps | tail strategies |
| `MultivariateGenerator` | empirical Σ + random shocks | HRP / portfolio strategies |
| `HistoricalReplay` | everything in the sample | reality check |
| `PermutationGenerator` | marginal dist only (autocorr destroyed) | **null — always include** |

### Adapters (compose with any generator)

- `LambertWTailAdapter(base, delta)` — wraps a Gaussian-based generator to
  add heavy tails via Goerg's Lambert W × N transform. Bijective.
  QMC-compatible (smooth post-map on Gaussian samples). See "Realism &
  validation" for the wrapper sketch and validator hooks.

### Strategy fingerprinting

```python
fingerprint(strategy) -> dict[str, float]
```

Run the strategy on a fixed battery of small probe datasets. Each probe
isolates **one** market property. Score = `Sharpe(probe) - Sharpe(IID baseline)`.
Cache by hash(strategy_code + params).

| Probe | Property isolated | If strategy scores here → it depends on… |
|---|---|---|
| IID Gaussian | none | drift only (or it's a bug) |
| AR(1) +ρ | return autocorrelation | momentum |
| AR(1) −ρ | mean reversion | reversal |
| GARCH(1,1), const μ | volatility clustering | vol regime |
| Jump diffusion | fat tails / jumps | tail behavior |
| Multi-asset Σ block | cross-section structure | pairs / diversification / HRP |
| Permuted real returns | marginal dist only | distributional edge (rare) |
| Real historical | everything | sanity baseline |

### Generator selection from fingerprint

```
fingerprint axis     → generators to include
{ autocorr }         → block bootstrap, AR/GARCH
{ vol clustering }   → GARCH, regime-switching
{ jumps / tails }    → jump-diffusion, block bootstrap of real
{ cross-section }    → multivariate w/ empirical Σ + shocks
{ many axes }        → block bootstrap of real (preserves everything cheaply)
always               → Gaussian baseline + Permutation null
```

### Output: sensitivity table

```
                  Median Sharpe   5th pct   95th pct   Survives?
Gaussian               0.05        -0.40      0.50         ✗
Block bootstrap        1.20         0.30      2.10         ✓
GARCH                  1.15         0.25      2.05         ✓
Jump diffusion         0.90        -0.10      1.80         ~
Permuted               0.02        -0.45      0.49      ✓ (no leak)
```

Plus per-generator distributions of: Sharpe, max DD, CAGR, turnover, vol.
Always report distributions, not point estimates.

## Implementation plan

1. `probes/` — fixed deterministic probe datasets, cached on disk.
2. `generators/` — path generator implementations behind common interface.
3. `fingerprint.py` — runs probe battery, caches by strategy hash.
4. `mc_run.py` — orchestrator (auto mode or explicit generator list).
5. `report.py` — sensitivity table + per-generator distribution plots.
6. Integration point: `mc_run` wraps the existing engine; engine code untouched.

## Required engineering practices (non-negotiable)

- **Common Random Numbers** when comparing strategies — same paths for A and B.
- **Per-path seeds**, not just global — so path #4,217 is reproducible.
- **Frictions on by default** — costs, slippage. §16.6 explicitly: CLA looks
  much worse once rebalance costs are included.
- **Convergence check** — compute the metric every N paths, plot, stop when
  CI stabilizes. 10,000 is overkill for most metrics; 1–2k often enough.
- **Fingerprint cache** keyed by strategy code+params hash — avoid re-running
  the probe battery on every MC run.

## Realism & validation

"Market-like" is a measurable property, enforced by a fixed validator that
every generator must pass before `mc_run` will use it. Reference: Cont
(2001), *"Empirical properties of asset returns."*

### Stylized facts to enforce

| # | Fact | Diagnostic | Why a backtest cares |
|---|---|---|---|
| 1 | Heavy tails | Kurt 4–30 daily; Hill index 3–5 | Underestimating tails inflates Sharpe |
| 2 | Volatility clustering | ACF(\|r\|) positive, slow decay; LB on r² rejects | Vol-targeting / breakout strategies need it |
| 3 | No autocorrelation in raw returns | ACF(r) ≈ 0 past lag 1 | Else momentum looks artificially good |
| 4 | Leverage effect | corr(r_t, σ_{t+1}) < 0 | Asymmetric tails; risk parity / vol targeting |
| 5 | Aggregational Gaussianity | Kurt(monthly) < Kurt(daily) | Sanity |
| 6 | Slight negative skew | ≈ −0.2 to −0.5 (equity) | Affects MaxDD distribution |
| 7 | Cross-sectional structure | Eigenvalues = MP bulk + spikes | HRP / risk parity depend on it |
| 8 | Long memory in vol | ACF(\|r\|) ~ k^{−α}, α ∈ [0.2, 0.4] | GARCH (short memory) vs FIGARCH/HAR |
| 9 | Jumps | Lee-Mykland / BNS detect them | Strategies holding through events |
| 10 | Regimes | HMM finds 2–4 (calm/volatile/crisis) | Strategy may only work in one |

Pick the subset relevant to the strategies under evaluation; enforce in the
validator. Don't gate on facts that don't matter for the question.

### Generator coverage of stylized facts

| Generator | Heavy tails | Vol clustering | Cross-section |
|---|---|---|---|
| Gaussian | ✗ | ✗ | from Σ |
| Stationary block bootstrap | ✓ inherited | ✓ inherited | ✓ inherited |
| GJR-GARCH(1,1) + Student-t | ✓ via Student-t | ✓ explicit | needs DCC-GARCH |
| Jump-diffusion (Merton) | ✓ jumps | ✗ alone | needs correlated jumps |
| Multivariate w/ shrunk Σ | partial | ✗ alone | ✓ structured |
| Historical replay | ✓ literal | ✓ literal | ✓ literal |
| Permutation | ✓ marginal | ✗ destroyed (point) | ✓ marginal |
| Gaussian + Lambert W adapter | ✓ via δ | ✗ | from Σ |

**Practical default**: stationary block bootstrap of real returns + optional
jump overlay. Free stylized facts plus a knob for tail stress beyond history.

### Validator interface

```python
class GeneratorValidator:
    def validate(generator, real_returns) -> dict[str, float]:
        synth = generator.sample(...)
        return {
            # Marginal distribution
            "mean_z":          abs(mean(synth) - mean(real)) / std(real),  # < 0.1
            "std_ratio":       std(synth) / std(real),                     # in [0.9, 1.1]
            "skew_diff":       skew(synth) - skew(real),                   # |.| < 0.3
            "kurt_ratio":      kurt(synth) / kurt(real),                   # in [0.7, 1.3]
            "ks_pvalue":       KS test synth vs real,                      # > 0.05
            "hill_tail_diff":  hill_index(synth) - hill_index(real),       # |.| < 0.5
            # Temporal
            "acf_r_lag1":      abs(ACF(synth)[1] - ACF(real)[1]),          # < 0.05
            "acf_abs_r":       area diff of ACF(|synth|) vs ACF(|real|),   # < 0.1
            "ljung_box_r2":    LB on synth² rejects independence,          # p < 0.01
            "vol_clust_decay": |α_synth - α_real|, ACF(|r|) ~ k^{-α},      # < 0.1
            "leverage_corr":   corr(r_t, σ_{t+1}) sign and magnitude,      # match real
            # Cross-sectional (multi-asset)
            "eig_spectrum_ks": KS dist between eigenvalue distributions,   # < 0.1
            "corr_frob":       ||Σ_synth - Σ_real||_F / ||Σ_real||_F,      # < 0.15
            # Lambert W diagnostic (when heavy-tail adapter is in chain)
            "lambert_delta":   abs(fit_delta(synth) - fit_delta(real)),    # < 0.05
            "gaussianized_kurt": kurt(gaussianize(synth, δ̂)) ≈ 3,
        }
```

Persist scores in `registry/` alongside generator config. **Any score outside
threshold → generator fails CI, can't be selected by `mc_run`.** Tearsheets
(QQ plot, ACF overlay, eigenvalue plot) are for human review; the numerical
battery is the gate.

### Calibration discipline (non-negotiable)

1. **Never calibrate on the test period.** Strict prior window only.
2. **Refit on rolling windows.** Each year, or each walk-forward step.
   Version params, store in registry.
3. **Audit the fit itself.** Cache likelihood, residual ACF, parameter
   stability across windows. A GARCH whose params jump 50% YoY is a bad
   fit, not a discovery.

### LambertWTailAdapter (Goerg's framework)

Wraps any Gaussian-based generator to add heavy tails. Bijective; the
inverse is what the validator uses to separate body-shape failures from
tail-heaviness failures.

```python
class LambertWTailAdapter(PathGenerator):
    """Y = U * exp(δ/2 * U²), U ~ N(0,1).
    δ = 0 → passthrough. δ ≈ 0.25 ≈ Student-t(4). Tail index = 1/δ."""
    def __init__(self, base: PathGenerator, delta: float = 0.0):
        self.base = base
        self.delta = delta

    def sample(self, n_paths, n_steps, n_assets, seed) -> np.ndarray:
        u = self.base.sample(n_paths, n_steps, n_assets, seed)
        return u * np.exp(0.5 * self.delta * u * u)

    @staticmethod
    def gaussianize(y: np.ndarray, delta: float) -> np.ndarray:
        from scipy.special import lambertw
        if delta == 0:
            return y
        return np.sign(y) * np.sqrt(lambertw(delta * y * y).real / delta)

    @classmethod
    def fit(cls, returns: np.ndarray, base: PathGenerator) -> "LambertWTailAdapter":
        # method of moments: solve δ to match excess kurtosis
        ...
```

**Use cases:**
- Wrap `GaussianGenerator` → heavy-tail Gaussian without rebuilding sampler.
- Wrap `MultivariateGenerator` per-asset → preserves marginals, **distorts
  cross-sectional correlation for δ > 0** (multivariate Lambert W is a
  v2 follow-up).
- Skip for GARCH — Student-t innovations there are battle-tested and `arch`
  handles them natively.
- QMC-compatible: δ-transform is a smooth post-map on Gaussian samples,
  preserves Sobol convergence properties.

**Why it earns its place over just using Student-t everywhere:** wrapper
pattern (no sampler swap), QMC-friendly, and the bijection lets the
validator Gaussianize real data to compare body-shape and tail-shape
*separately*. Student-t still wins for GARCH innovations and for the
multivariate case.

## Speed optimization

Target: every ms matters, **no accuracy compromise**. Variance reduction
(fewer paths needed for same answer) compounds with per-path compute, so
attack both axes.

### Tier 1 — Variance reduction (mathematically exact, fewer paths needed)

| Technique | Win | Notes |
|---|---|---|
| **Quasi-Monte Carlo (Sobol, scrambled)** | 10–100× | `O(1/N)` vs `O(1/√N)`. `scipy.stats.qmc.Sobol` + `norm.ppf`. Owen scrambling for unbiased + error bars. Brownian-bridge construction past ~1k dim. |
| **Antithetic variates** | 2× free | For every ε also use −ε. Works on any near-linear-in-noise quantity. |
| **Common Random Numbers (CRN)** | 10–100× *for differences* | Same paths across strategy variants. Mandatory for HRP-vs-CLA-style comparisons. Bake into `mc_run` API. |
| **Control variates** | 10–100× for tails | Subtract correlated known-mean variable (e.g. raw drift). Big for VaR / cVaR / MaxDD. |
| **Stratified / Latin Hypercube** | 2–10× | `scipy.stats.qmc.LatinHypercube`. Stacks with above. |
| **Importance sampling** | Big for rare events | Only when tail metrics are the deliverable. |
| **Adaptive convergence stopping** | 2–5× avg | Stop when CI < threshold instead of fixed N. |

### Tier 2 — Per-path compute (stack-aware: numba/joblib/bottleneck already in)

- **Vectorize path gen**: one `(N, T, K)` tensor, Cholesky once, broadcast
  `Z @ L.T + μ`. 100–1000× over per-path loops.
- **Path-parallel strategy execution**: mirror `MultiPathEngine`'s joblib
  pattern. Use `prefer="threads"` IF numba kernels have `nogil=True` —
  avoids per-task pickle overhead.
- **`generate_weights_batch` opt-in**: optional Strategy method returning
  weights for all paths at once. `BatchDataView` exposes `prices` as
  `(N, T, K)`. Pure-numpy strategies adapt with no logic change.
  Another 10–100× on top of joblib for adopters.
- **RNG**: `np.random.default_rng(seed)` (PCG64) — 2–4× over legacy. Bulk
  draw all randoms once, not per step.
- **Numba flags**: `parallel=True` + `prange`, `nogil=True`, `cache=True`.
  **Do NOT use `fastmath=True`** — reorders FP, drops NaN handling.
- **Memory layout**: `np.ascontiguousarray`, C-order, pre-allocate buffers,
  `np.empty` over `np.zeros` when overwriting. `float32` only after
  validating drift on cumulative products is < 1bp.
- **Single-pass metrics**: fuse Sharpe / DD / TUW / VaR into one numba
  kernel — 4× memory bandwidth → 1×.

### Tier 3 — Architectural

- **Path-tensor cache** via `registry/hashing.py` keyed on
  `(generator_config, seed, N, T, K)` → parquet. Second runs are I/O only.
- **Fingerprint cache** keyed on `(strategy_repr, probe_battery_version)`.
- **Lazy / streaming generation** only if `(N, T, K) × float64` exceeds RAM.
- **GPU (CuPy / JAX)**: defer until profiling shows path gen is the
  bottleneck. JAX becomes natural fit when v2 brings GAN/GARCH DGP — gets
  autodiff for fitting the DGP for free.

### Anti-patterns under "no accuracy compromise"

- `numba fastmath=True`
- `float16`
- shorter T or skipped rebalance dates (changes the problem)
- skipping the permutation null
- eyeballing convergence without a CI threshold

### ROI-ordered implementation priority

1. Vectorized path gen `(N, T, K)` + Cholesky once
2. CRN baked into `mc_run` API
3. Sobol QMC option in generators
4. Antithetic variates flag
5. `generate_weights_batch` opt-in fast path
6. Joblib path-parallelism with `nogil=True`
7. Path-tensor caching
8. Convergence-based early stopping
9. Control variates for tail metrics
10. GPU (only after profiling)

Combined realistic speedup vs naive: **~1000× on Sharpe/DD studies**.

## Open questions

- What's the engine's current data interface? (informs generator output shape)
- Where does the engine live in the repo? Where to plug `mc_run` in?
- Existing parallelism / vectorization story?

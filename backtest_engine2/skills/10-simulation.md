# Skill 10 — Monte Carlo Simulation

> **Read `PROTOCOL.md` and `skills/00-overview.md` first.** Cell-loop and announcement rules in PROTOCOL apply.

> Stage 8 (optional per PROTOCOL §8). Run after Stage 7 metrics and before Stage 9 selection if the user wants synthetic-path stress testing. Engine source: `backtest/simulation/`.

---

## When to load

- **Stage 8 (optional)**: stress-testing the strategy across synthetic paths to see if its edge persists outside the single observed sample.
- Any cell that imports from `backtest.simulation`.
- Any cell that mentions: generators, path tensors, fingerprinting, permutation null, validator, sensitivity table, Monte Carlo.

This is the deepest skill. Read sections you need; don't memorize.

---

## What you'll find here

1. The 4-step workflow (fingerprint → select → validate → run).
2. `PathGenerator` ABC — the shape every generator returns.
3. The 8 generators with stylized-fact coverage.
4. The 3 adapters (jump overlay, antithetic, Lambert-W tail).
5. `GeneratorValidator` + `DEFAULT_THRESHOLDS` — the CI gate.
6. `Probe` / `ProbeBattery` — deterministic synthetic datasets.
7. `fingerprint` + `FingerprintCache` — strategy → score vector.
8. `select_generators` — fingerprint → generator dict.
9. `MonteCarloEngine` + `MonteCarloResult`.
10. `run_until_converged` + `AdaptiveResult` — convergence-based stopping.
11. `run_batch` + `BatchResult` + `BatchDataView` — the vectorized fast path.
12. `PathTensorCache` — disk cache for path tensors.
13. `panel_from_returns` — the materialization helper.
14. Common mismatches.
15. Minimal valid cell.
16. Validation checklist.
17. Anti-patterns.

---

## 1. The four-step workflow

The simulation module is structured around four steps, each with its own primitive:

```
1. fingerprint(strategy)             → dict[probe_name -> Δ Sharpe vs IID baseline]
2. select_generators(fp, real_rets)  → dict[gen_name -> PathGenerator]
3. validator.validate(generator)     → pass/fail per Cont-2001 stylized fact
4. MonteCarloEngine(strategy, gens). → MonteCarloResult with N paths per generator
       .run(n_paths, seed)
```

You can skip steps. For a one-off stress test with a known generator, go straight to step 4 with a hand-picked `generators=` dict. For a published-grade result, run all four and report the sensitivity table per generator.

**Out of scope for v1 MC layer (would be v2+ work):** GARCH calibration via MLE (only method-of-moments is provided), regime-switching, GAN-based DGP, leverage effect coupled across assets (DCC-GARCH). Surface as a mismatch if the spec calls for them.

---

## 2. `PathGenerator` ABC

```python
class PathGenerator(ABC):
    @abstractmethod
    def sample(
        self, n_paths: int, n_steps: int, n_assets: int, seed: int
    ) -> np.ndarray:  # shape (n_paths, n_steps, n_assets) of simple returns
        ...

    @abstractmethod
    def config(self) -> dict:  # serializable; used by PathTensorCache key
        ...
```
`source: backtest/simulation/base.py:12–29`

Every generator must return a 3-D ndarray of simple returns and a JSON-serializable config dict. The config keys the disk cache (`§12`) — change a parameter, the cache misses, exactly the behavior you want.

Same `seed` → same tensor. **Common Random Numbers (CRN):** comparing strategies A and B with the same `MonteCarloEngine(generators=..., ...)` and the same `seed` produces bit-identical path tensors for both, so any difference is attributable to the strategy. This is automatic; no extra wiring. `source: backtest/simulation/mc.py:186–193`

---

## 3. The 8 generators

`source: backtest/simulation/generators.py:12–498`

| Generator | What it preserves | Use as | `.fit(real_returns)` classmethod? |
|---|---|---|---|
| `GaussianGenerator(mu, sigma)` | μ, σ only | baseline / null | Yes — pooled mean/std per asset |
| `SobolGaussianGenerator(mu, sigma, scramble=True)` | μ, σ + faster convergence (Sobol QMC) | Gaussian when n_paths is a power of 2 | Yes |
| `PermutationGenerator(real_returns)` | marginal dist + contemporaneous correlation; **destroys** autocorrelation / vol clustering | **lookahead-leak null — always include** | n/a (constructed from real) |
| `HistoricalReplayGenerator(real_returns)` | everything in random contiguous slices | sanity baseline | n/a |
| `BlockBootstrapGenerator(real_returns, block_length)` | heavy tails, vol clustering, autocorrelation, cross-section (Politis & Romano 1994 stationary bootstrap) | **practical MC default** | Yes — heuristic block length = 2 × avg first-insignificant ACF lag |
| `GarchGenerator(mu, omega, alpha, beta, gamma, nu=None)` | explicit vol dynamics (GJR-GARCH(1,1)), leverage effect, optional Student-t fat tails | vol-sensitive strategies | Yes — method-of-moments for `omega, mu`; other params input |
| `MultivariateGenerator(mu, cov, shrinkage=0.0)` | μ vector + full covariance Σ | HRP / risk-parity / portfolio strategies | Yes |
| `SobolMultivariateGenerator(mu, cov, shrinkage, scramble)` | same as `MultivariateGenerator` + Sobol QMC convergence | high-dimensional MC where convergence rate matters | Yes |

### 3.1 Sobol convergence note

`SobolGaussianGenerator` / `SobolMultivariateGenerator` use scrambled Sobol uniforms (`scipy.stats.qmc.Sobol`) inverted through `ndtri` to standard normals. Convergence is `O(1/N)` on linear functionals vs `O(1/√N)` for IID Gaussian — typically 10-100× speedup at the same precision. Best when `n_paths` is a power of 2 (scipy emits a UserWarning otherwise — suppressed locally). `source: generators.py:213–251, 439–497`

### 3.2 Block bootstrap geometry

`BlockBootstrapGenerator` uses the stationary bootstrap of Politis & Romano (1994). At each step, with probability `1/block_length`, start a new block at a random index; otherwise advance the prior index by 1 (wrap-around at `T`). Geometric block lengths mean **expected** length = `block_length`, not fixed. Wrap means `n_steps > T` is allowed. `source: generators.py:150–172`

The `.fit(real_returns)` heuristic picks `block_length = 2 × avg(first-insignificant-ACF-lag)` across assets, with a max of `max_lag=min(50, n//4)`. Sensible default; tune if your strategy reads autocorrelation past that lag. `source: generators.py:182–210`

### 3.3 GARCH constraints

`alpha + beta + gamma/2 < 1` for stationarity — hard-enforced at construction. `nu > 2` for finite variance (Student-t). Standard equity defaults: `alpha=0.05, beta=0.90, gamma=0.05`. `.fit(real_returns)` sets `omega` and `mu` from pooled mean/var; alpha/beta/gamma/nu are user-supplied. `source: generators.py:268–371`

### 3.4 Multivariate shrinkage

`MultivariateGenerator(..., shrinkage=α)` linearly shrinks `cov` toward `diag(diag(cov))`: `(1-α) · cov + α · diag(diag(cov))`. `α=0` = raw sample cov; `α=1` = diagonal only (no correlation). Use when sample size is small relative to K and the sample cov is unstable. Cholesky is computed once at construction, then vectorized `z @ L.T + mu` per sample. `source: generators.py:374–436`

---

## 4. The 3 adapters

Adapters wrap a base `PathGenerator` and modify its output. All adapters are themselves `PathGenerator` subclasses, so they compose.

`source: backtest/simulation/adapters.py:11–195`

### 4.1 `JumpOverlayAdapter(base, jump_intensity, jump_mu=0, jump_sigma=0.05, correlate_assets=False)`

Adds Poisson-thinned Gaussian jumps to a base generator's returns. At each `(path, step)` cell, with probability `jump_intensity` a jump is added; `correlate_assets=True` shares a single jump value across all assets on that bar (market-wide event); `=False` each asset draws independently.

Use to stress-test tail risk beyond what the base sample contains. Common composition: `JumpOverlayAdapter(BlockBootstrapGenerator(...), jump_intensity=0.01)` — historical paths with a crash overlay. `source: adapters.py:11–66`

### 4.2 `AntitheticAdapter(base)`

Pairs each path with its antithetic (`2·mu - r`) for ~2× variance reduction. `n_paths` must be even; even-indexed paths are base, odd-indexed are antithetic. Requires the base to expose `.mu` (Gaussian-based generators only — raises `TypeError` otherwise). `source: adapters.py:69–105`

### 4.3 `LambertWTailAdapter(base, delta=0.0)`

Goerg's `Y = U · exp(δ/2 · U²)` heavy-tail wrapper. Per asset: standardize base output, apply the transform, renormalize. `δ = 0` is bit-identical passthrough; `δ ∈ [0, 0.25)` keeps kurtosis finite (hard-enforced at construction). Composes with any base; tails are added via marginal transform.

`.fit(real_returns, base)` solves `excess_kurtosis(delta) = excess_kurtosis(real_returns)` for `δ` via `scipy.optimize.brentq`. Returns `δ=0` if real has non-positive excess kurtosis. `source: adapters.py:108–195`

`gaussianize(y, delta)` is the inverse — useful for the validator's body-shape-vs-tail-shape separation. `source: adapters.py:154–163`

---

## 5. `GeneratorValidator` + `DEFAULT_THRESHOLDS` — the CI gate

`source: backtest/simulation/validator.py:263–425`

```python
GeneratorValidator(real_returns, n_paths=20, n_steps=None, seed=0, thresholds=None)
    .validate(generator) -> ValidationResult
```

Scores a generator's output against a battery of Cont (2001) stylized facts. If any score falls outside its `Threshold`, `overall_passed=False` and you should not use that generator in production.

### 5.1 The ten scored metrics

`source: backtest/simulation/validator.py:323–338`

| Metric | What it checks | Default threshold |
|---|---|---|
| `mean_z` | `\|mean(synth) - mean(real)\| / std(real)`, asset-averaged | `≤ 0.1` |
| `std_ratio` | `std(synth) / std(real)`, asset-averaged | `∈ [0.9, 1.1]` |
| `skew_diff` | `skew(synth) - skew(real)`, asset-averaged | `∈ [-0.3, 0.3]` |
| `kurt_ratio` | `kurt(synth, raw) / kurt(real, raw)` | `∈ [0.5, 1.5]` |
| `ks_pvalue` | mean KS-test p across assets | `≥ 0.05` |
| `hill_tail_diff` | Hill tail index diff (synth − real) | `∈ [-0.5, 0.5]` |
| `acf_r_lag1_diff` | mean `\|ACF(r)_lag1 synth − real\|` | `≤ 0.05` |
| `acf_abs_r_lag1_diff` | same on `\|r\|` (vol clustering) | `≤ 0.10` |
| `ljung_box_r2_pvalue` | LB p-value on `r²` synth (lower = synth has clustering) | `≤ 0.01` |
| `corr_frobenius_ratio` | `‖Σ_synth − Σ_real‖_F / ‖Σ_real‖_F` | `≤ 0.15` |

Override per-metric thresholds via `GeneratorValidator(..., thresholds={...})`. Pass `thresholds={}` to disable all gating (everything passes). The result object always carries both the raw scores and the per-metric `passes`. `source: validator.py:184–195, 269–321`

### 5.2 `ValidationResult`

```python
@dataclass
class ValidationResult:
    scores:             dict[str, float]
    thresholds:         dict[str, Threshold]
    passed:             dict[str, bool]
    overall_passed:     bool
    generator_config:   dict
    n_paths:            int
    n_steps:            int
    real_returns:       np.ndarray | None  # repr=False
    synth_returns:      np.ndarray | None  # repr=False
```
`source: validator.py:31–55`

`result.report()` → tidy `pd.DataFrame[metric × (score, lower, upper, passed)]`. `result.plot()` → 6-panel tearsheet (QQ, marginal histogram, ACF(r), ACF(|r|), eigenvalue spectrum, pass/fail table). `source: validator.py:43–179`

### 5.3 Why the gate matters

A generator that doesn't reproduce the stylized facts of the real returns will mislead the MC. A momentum strategy run on a generator with no autocorrelation gets a meaningless null distribution. The validator turns "is this generator realistic enough?" from eyeball judgment into a numerical gate. **Run the validator before trusting a generator's results** — surface the pass/fail in the cell.

---

## 6. `Probe` and `ProbeBattery`

`source: backtest/simulation/probes.py:16–172`

```python
@dataclass(frozen=True)
class Probe:
    name: str
    description: str
    panel: DataPanel
    isolates: str  # "momentum" | "mean_reversion" | "vol_clustering" | "tails" | "cross_section" | "none"
```

A `Probe` is a fixed, deterministic synthetic dataset isolating one market property. Six probes ship in `ProbeBattery`:

| Probe | Isolates | What it looks like |
|---|---|---|
| `iid_gaussian` | `none` | IID N(0.0003, 0.015). Baseline — any strategy here has only drift edge. |
| `momentum` | `momentum` | Persistent per-asset alphas `linspace(-0.0015, 0.0015)`. XS momentum profits. |
| `mean_reversion` | `mean_reversion` | OU-style log-price mean-reversion to a common attractor. XS momentum loses. |
| `vol_clustering` | `vol_clustering` | GARCH(1,1)-like `alpha=0.10, beta=0.85`. Vol-targeting profits. |
| `jump_diffusion` | `tails` | Gaussian + Poisson jumps (intensity 0.02). Tail-aware strategies show up. |
| `cross_section` | `cross_section` | Two-block correlation (0.7 within, 0.2 across). HRP / portfolio strategies profit. |

`ProbeBattery(n_assets=5, n_steps=520, seed=42, init_price=100.0)` — defaults are reproducible. `battery_id()` returns a short hex digest of the battery config; used by `FingerprintCache` to invalidate cache when battery changes. `source: probes.py:31–64`

`battery.baseline()` returns the iid_gaussian probe; `battery.probes()` returns the full list of 6.

---

## 7. `fingerprint` + `FingerprintCache`

`source: backtest/simulation/fingerprint.py:20–84`

```python
fingerprint(
    strategy: Strategy | Callable[[], Strategy],
    battery: ProbeBattery | None = None,
    warmup_bars: int = 60,
    cache: FingerprintCache | None = None,
    costs: CostModel | None = None,
    initial_capital: float = 1_000_000.0,
) -> dict[str, float]
```

Runs the strategy on each probe, computes `Sharpe(probe) - Sharpe(iid_gaussian)`, returns a dict keyed by probe name. The output is the **delta-Sharpe vector** — positive values mean the strategy exploits that property; near-zero means it doesn't depend on it; negative means it's actively hurt by it.

Example fingerprint for a cross-sectional momentum strategy:
```
{'iid_gaussian': 0.0, 'momentum': +1.8, 'mean_reversion': -1.2,
 'vol_clustering': 0.1, 'jump_diffusion': 0.0, 'cross_section': 0.3}
```

### 7.1 Reading the fingerprint

- **All near-zero**: strategy is drift-only (likely a bug or trivial buy-and-hold).
- **One spike**: strategy depends on exactly that property. Choose generators that preserve it.
- **Multiple spikes**: combined edge; preserve all of them. Block-bootstrap of real is usually the easiest way.
- **Negative on permutation/iid**: leak. The strategy "works" on data with no exploitable structure — usually a lookahead bug. **Stop and fix**, don't proceed to MC.

### 7.2 `FingerprintCache`

`source: backtest/simulation/fingerprint.py:66–83`

JSON-file cache keyed by `(strategy_hash, battery_id)`. `strategy_hash` is the registry's `repr(strategy)`-based hash (skill 11); the battery_id is the short hex from `battery.battery_id()`. Re-running `fingerprint(...)` with the same strategy and battery hits the cache; changing any `__init__` arg of the strategy or any field of `ProbeBattery` misses.

Construct with `FingerprintCache("./fingerprint_cache")` and pass to `fingerprint(..., cache=...)`. Stable `__repr__` on the strategy is required for cache correctness (skill 02 covers `__repr__`).

---

## 8. `select_generators`

`source: backtest/simulation/auto_select.py:14–43`

```python
select_generators(
    fingerprint_scores: dict[str, float],
    real_returns: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, PathGenerator]
```

Translates fingerprint scores into a generator dict ready for `MonteCarloEngine`. Routing per fingerprint axis:

| Triggered by | Generator added |
|---|---|
| **Always** | `gaussian` (baseline) + `permutation` (null) |
| Any non-baseline probe > `threshold` | `block_bootstrap` (preserves all stylized facts of real returns cheaply) |
| `jump_diffusion` > `threshold` | `block_bootstrap_jumps` (block bootstrap + `JumpOverlayAdapter`) |
| `vol_clustering` > `threshold` | `garch` (explicit GJR-GARCH(1,1) vol dynamics; method-of-moments fit) |
| `cross_section` > `threshold` **and** K ≥ 2 | `multivariate` (empirical Σ + Cholesky shocks) |

`source: backtest/simulation/auto_select.py:14–61`

Override by constructing the generators dict yourself when you want explicit control over which generators run or with non-default parameters. `select_generators` always returns at least 2 generators (`gaussian` + `permutation`).

---

## 9. `MonteCarloEngine` + `MonteCarloResult`

`source: backtest/simulation/mc.py:179–308`

```python
MonteCarloEngine(
    strategy: Strategy | Callable[[], Strategy],
    generators: dict[str, PathGenerator],     # at least one; PROTOCOL wants gaussian + permutation always
    assets: Sequence[str],
    dates: pd.DatetimeIndex,
    costs: CostModel | None = None,
    liquidity: LiquidityCap | None = None,
    initial_capital: float = 1_000_000.0,
    warmup_bars: int = 0,
    path_cache: PathTensorCache | None = None,
)
    .run(n_paths=1000, seed=0, n_jobs=-1) -> MonteCarloResult
```

### 9.1 Construction validation

`source: mc.py:207–214`

- `initial_capital > 0`
- `generators` dict non-empty
- `0 <= warmup_bars < len(dates)`

### 9.2 Per-generator seed isolation

Within one `MonteCarloEngine.run(seed=S)`, generator `i` receives `effective_seed = S + i · 10_000_007`. Different generators with the same base seed produce **non-overlapping** noise streams. `source: mc.py:21, 240–242`

### 9.3 Parallelism

Each `(generator, path_idx)` pair becomes one joblib job. Each worker calls `panel_from_returns` → deep-copies the strategy → constructs an `Engine` → calls `engine.run(start=panel.dates[warmup_bars])`. `n_jobs=-1` (all cores) by default. `n_jobs=1` for sequential debugging. `source: mc.py:267–289, 316–343`

### 9.4 `MonteCarloResult`

```python
@dataclass
class MonteCarloResult:
    path_returns:       dict[str, pd.DataFrame]   # gen_name → DataFrame[date × path_i]
    path_equity:        dict[str, pd.DataFrame]
    initial_capital:    float
    n_paths:            int
    seed:               int
    generator_configs:  dict[str, dict]
```
`source: mc.py:24–31`

Methods:

- `metrics_per_path() -> dict[gen_name, list[MetricsReport]]`. `source: mc.py:33–45`
- `aggregate_metrics() -> dict[gen_name, dict[metric_stat, float]]`. 19 metrics × 4 stats = **76 keys per generator**, matching `MultiPathResult.aggregate_metrics` (skill 07). Keys: `sharpe, sortino, calmar, modified_sharpe, psr, max_drawdown, time_under_water, var_5, cvar_5, ann_return, ann_vol, total_return, total_pnl, total_costs, longest_underwater, hit_rate, turnover, information_ratio, beta` — each suffixed with `_mean / _std / _min / _max`. `source: mc.py:47–73`
- `sensitivity_table(metrics=..., quantiles=(0.05, 0.50, 0.95)) -> pd.DataFrame`. The headline output: per-generator distribution summary as a table. `source: mc.py:72–95`
- `summary(metrics=..., quantiles=...)` — prints the sensitivity table and returns it. `source: mc.py:97–110`
- `plot(figsize=(13, 8))` — multi-panel tearsheet: per-generator equity overlays, per-generator Sharpe histogram, per-generator max-DD histogram. Requires matplotlib. `source: mc.py:112–176`

The sensitivity table is what to show the user. If `sharpe_q05` (5th-percentile annualized Sharpe) is positive across all generators, the strategy is robust. If it's positive on `block_bootstrap` but negative on `gaussian`, the edge depends on stylized facts of real returns — fine, but the user should know.

---

## 10. `run_until_converged` + `AdaptiveResult`

`source: backtest/simulation/adaptive.py:21–105`

```python
run_until_converged(
    engine: MonteCarloEngine,
    metric: str = "sharpe",
    target_se: float = 0.05,
    batch_size: int = 100,
    max_paths: int = 10_000,
    min_paths: int = 200,
    seed: int = 0,
    n_jobs: int = -1,
    verbose: bool = False,
) -> AdaptiveResult
```

Runs paths in batches until `SE = std(metric across paths) / sqrt(n) < target_se` across **all** generators (max SE rule), or `n_paths` reaches `max_paths`. Per-batch seeds are offset by `batch_size × 1_000_003` so paths don't repeat. `source: adaptive.py:55–95`

`AdaptiveResult` carries `result` (the final `MonteCarloResult`), `converged` (bool — did SE fall below `target_se` before `max_paths`), `n_paths_used`, `se_trace` (per-generator list of SE at each batch), `metric`, `target_se`.

Use when you don't know upfront how many paths you need. Cheaper than picking a large fixed `n_paths`. **Don't** use when CRN is needed for two-strategy comparison — adaptive stopping uses different `n_paths` per run.

---

## 11. `run_batch` + `BatchResult` + `BatchDataView` — the vectorized fast path

`source: backtest/simulation/batch.py:13–248`

```python
run_batch(
    strategy: Strategy,                          # must define generate_weights_batch
    paths: np.ndarray,                           # (N, T, K) of returns
    assets: Sequence[str],
    dates: pd.DatetimeIndex,
    warmup_bars: int = 0,
    initial_capital: float = 1_000_000.0,
    init_price: float = 100.0,
) -> BatchResult
```

Vectorized version of `MonteCarloEngine` for strategies that can compute weights across all paths in one numpy operation. The strategy must define:

```python
def generate_weights_batch(self, view: BatchDataView, t: pd.Timestamp) -> np.ndarray:
    # view.prices is (N, T_so_far, K). Return shape (N, K).
    ...
```

`BatchDataView` is the path-aware analogue of `DataView`. `view.prices[:, :t+1, :]` exposes all paths' prices up to bar `t`. No DataFrame conversion per path — numpy throughout. **10–100× speedup** over `MonteCarloEngine` when adoptable.

### 11.1 Trade-offs

- **No costs.** `run_batch` does not consume a `CostModel`. Compare-to-`MonteCarloEngine` only when you've already established the strategy's net edge with costs.
- **No liquidity caps.** Same reason.
- **Identity risk only.** `apply_risk` is not called; weights returned by `generate_weights_batch` are used as-is.
- **First post-warmup bar's return is `NaN`.** Matches per-path `Engine` convention so `to_monte_carlo_result()` plugs cleanly into the rest of the pipeline. `source: batch.py:235–238`

### 11.2 `BatchResult`

```python
@dataclass
class BatchResult:
    path_returns: np.ndarray      # (N, T - warmup_bars)
    path_equity:  np.ndarray      # (N, T - warmup_bars)
    weights:      np.ndarray      # (N, T - warmup_bars, K)
    dates:        pd.DatetimeIndex
    initial_capital: float
    n_paths:      int
```
`source: batch.py:33–40`

- `to_monte_carlo_result(gen_name="batch", seed=0)` — wraps as a `MonteCarloResult` with one generator entry, so `summary()` / `sensitivity_table()` work uniformly. `source: batch.py:42–60`
- `summary()` — prints per-path aggregate stats; returns a dict.

Use `run_batch` when the strategy is pure-numpy and you can afford to skip costs/liquidity/risk (typically: exploration phase, large MC sweeps for distribution shape). For the published result, run with costs through `MonteCarloEngine`.

---

## 12. `PathTensorCache` — disk cache for path tensors

`source: backtest/simulation/cache.py:13–76`

```python
PathTensorCache(cache_dir: str | Path)
    .get(generator, seed, n_paths, n_steps, n_assets) -> np.ndarray | None
    .put(generator, seed, n_paths, n_steps, n_assets, tensor) -> None
    .clear() -> None
    .stats() -> {"count": int, "total_bytes": int}
```

Disk cache keyed by `sha256(json(generator.config()) + "|seed|n_paths|n_steps|n_assets")[:16]`. Stores `.npy` files. Pass via `MonteCarloEngine(..., path_cache=cache)` — the engine reads on hit, writes on miss. Identical generator config + dims + seed → same cached tensor.

Use for:
- Comparing strategies A and B on the same paths (CRN, second-time cheap).
- Iterating on metrics analysis without regenerating paths.
- Skipping path generation altogether across machines if you share the cache dir.

The cache key includes every field of `generator.config()`, so changing any generator parameter invalidates. The historical-data generators (`Permutation`, `HistoricalReplay`, `BlockBootstrap`) include `real_shape` and `real_checksum` in their config — same real data → same key. Different real data → different key. `source: generators.py:87–92, 121–126, 174–180`

---

## 13. `panel_from_returns`

`source: backtest/simulation/base.py:32–57`

```python
panel_from_returns(
    returns: np.ndarray,                # (n_steps, n_assets) of simple returns
    assets: Sequence[str],
    dates: pd.DatetimeIndex,
    init_price: float = 100.0,
) -> DataPanel
```

Materializes one path's returns into a `DataPanel` by seeding prices at `init_price` and `np.cumprod(1 + returns, axis=0)`. Used internally by `MonteCarloEngine`'s workers and by `ProbeBattery` to build probe panels. `check_outliers=False` is set so synthetic data doesn't trigger real-data outlier warnings. `source: base.py:54–57`

Reach for it directly when you want to:
- Run a single Engine against a generator-produced path manually.
- Construct custom probe datasets.

---

## 14. Common mismatches — surface these by default

| Spec says | Engine has | Question to ask |
|---|---|---|
| "Run Monte Carlo with 10000 paths" | `MonteCarloEngine.run(n_paths=10000)`. | Confirm: do they want fixed N=10000 or adaptive convergence (`run_until_converged`)? Recommend adaptive unless they specifically asked for a fixed count. |
| "Use a GAN to generate paths" | Not in v1. Generators are: Gaussian, Sobol-Gaussian, Permutation, Historical-Replay, Block-Bootstrap, GARCH, Multivariate, Sobol-Multivariate, plus 3 adapters. | Out of scope. Recommend block-bootstrap as the practical default; surface the limitation. |
| "Regime-switching DGP" | Not in v1. `GarchGenerator` is single-regime. | Out of scope. Surface; suggest block-bootstrap of real returns covering multiple regimes as a workaround. |
| "GARCH with leverage effect coupled across assets" | `GarchGenerator` is asset-independent (per-asset GARCH process driven by shared parameters but independent innovations). True DCC-GARCH not provided. | Surface. Use `MultivariateGenerator` for cross-section + `GarchGenerator` for univariate vol if both matter. `select_generators` adds both automatically when both fingerprint axes light up. |
| "Auto-pick generators" | `fingerprint(strategy) → select_generators(scores, real_returns)`. | Confirm threshold (default 0.5). Recommend running `fingerprint` first to surface the strategy's actual edge type. |
| "Compare strategy A vs B on the same paths" | CRN is automatic: same generators dict + same seed → bit-identical tensors. With `path_cache`, second strategy hits cache and skips generation. | Confirm: do they want identical paths (CRN) or independent paths (recommend independent for confidence intervals of differences). |
| "Validate the generator first" | `GeneratorValidator(real_returns).validate(generator)`. Pass/fail per Cont (2001) stylized fact. | Surface: which thresholds matter for their strategy. Default thresholds are conservative; loosen explicitly if a fact is irrelevant. |
| "Why is my strategy's Sharpe negative on the permutation null?" | A working strategy should be near zero on permuted returns. Negative = lookahead leak in the strategy. | **Stop. Surface the leak as a Stage 2 bug.** Do not proceed to MC. |
| "Use jump diffusion for crash risk" | `JumpOverlayAdapter(base, jump_intensity, jump_sigma)`. Wrap any base. | Surface: typical equity-market intensity is ~0.01–0.02 per day with sigma 0.03–0.05. Tune relative to spec's "crash" definition. |
| "How many paths do I need" | `run_until_converged(engine, metric, target_se)`. Default `target_se=0.05`. | Surface: convergence-based stopping is the right answer. Show the SE trace in the cell. |
| "I want to reuse paths across runs" | `PathTensorCache(dir)` → pass to `MonteCarloEngine(..., path_cache=cache)`. | Confirm cache dir and whether they intend to share it (same generator config + dims + seed = cache key). |
| "Use Sobol for faster convergence" | `SobolGaussianGenerator` / `SobolMultivariateGenerator`. Best when `n_paths` is a power of 2. | Surface. Speedup is 10–100× on linear functionals (mean returns, mean Sharpe); less helpful for tail metrics. |
| "Skip costs in the MC for speed" | `run_batch(strategy, ...)` — no costs, no liquidity, vectorized. Strategy must implement `generate_weights_batch`. | Surface the constraint: only for exploration; final result must include costs via `MonteCarloEngine`. |
| "Validate but skip the eigenvalue spectrum check" | `GeneratorValidator(..., thresholds={...})` — pass a subset of `DEFAULT_THRESHOLDS` excluding `corr_frobenius_ratio`. | Confirm which facts they want to gate on. |

---

## 15. Minimal valid cell — Stage 8

Two paths through the cell, depending on how invested the user is in MC.

### 15.1 Quick stress test (skip fingerprinting)

```python
import numpy as np
import pandas as pd
from backtest.simulation import (
    MonteCarloEngine, BlockBootstrapGenerator, GaussianGenerator,
    PermutationGenerator, GeneratorValidator,
)

# 1. Fit generators on the real returns the engine saw.
real_returns = result.returns.dropna().to_numpy().reshape(-1, 1)  # 1 asset; reshape if multi-asset
gens = {
    "gaussian":        GaussianGenerator.fit(real_returns),
    "block_bootstrap": BlockBootstrapGenerator.fit(real_returns),
    "permutation":     PermutationGenerator(real_returns),
}

# 2. Validate (CI gate).
validator = GeneratorValidator(real_returns, n_paths=20, seed=0)
for name, gen in gens.items():
    vr = validator.validate(gen)
    print(f"{name:18s}  overall_passed={vr.overall_passed}  "
          f"failed={[m for m, p in vr.passed.items() if not p]}")

# 3. Run.
mc = MonteCarloEngine(
    strategy=strat,                                 # from Stage 2
    generators=gens,
    assets=list(panel.assets_all),
    dates=panel.dates,
    costs=costs,                                    # from Stage 3 — REQUIRED
    initial_capital=1_000_000,
    warmup_bars=60,
).run(n_paths=500, seed=0, n_jobs=-1)

# 4. Visible artifact for validation.
mc.summary()                                        # prints sensitivity table
```

### 15.2 Full pipeline (fingerprint → select → validate → run)

```python
from backtest.simulation import (
    MonteCarloEngine, GeneratorValidator,
    fingerprint, FingerprintCache, select_generators,
)

# 1. Fingerprint the strategy.
fp_cache = FingerprintCache("./fingerprint_cache")
fp = fingerprint(strat, cache=fp_cache, costs=costs)
print(f"Fingerprint: {fp}")

# 2. Select generators from the fingerprint.
real_returns = ...   # the real returns the engine saw, shape (T, K)
gens = select_generators(fp, real_returns, threshold=0.5)
print(f"Selected generators: {list(gens.keys())}")

# 3. Validate.
validator = GeneratorValidator(real_returns, n_paths=20, seed=0)
for name, gen in gens.items():
    vr = validator.validate(gen)
    if not vr.overall_passed:
        print(f"WARNING: {name} failed validation: "
              f"{[m for m, p in vr.passed.items() if not p]}")

# 4. Run + summarize.
mc = MonteCarloEngine(
    strategy=strat, generators=gens,
    assets=list(panel.assets_all), dates=panel.dates,
    costs=costs, initial_capital=1_000_000, warmup_bars=60,
).run(n_paths=1000, seed=0, n_jobs=-1)
mc.summary()
```

Do **not** combine 15.1 and 15.2 in one cell. Pick one or the other based on whether the user wants the auto-routing.

---

## 16. Validation checklist (after the cell runs)

- [ ] **Generators include `gaussian` AND `permutation`** at minimum. The MC_DESIGN doc treats permutation as the lookahead-leak null and Gaussian as the drift baseline. Both are mandatory; neither is automatic unless you used `select_generators`.
- [ ] **Validator passed on every selected generator.** Or, for any failure, the failed metric is irrelevant to the strategy's edge (e.g. `corr_frobenius_ratio` failing on a single-asset run is moot).
- [ ] **Strategy's Sharpe distribution on `permutation` is centered near zero** — if median Sharpe on permuted returns is `> 0.5`, the strategy has a lookahead leak. Stop and fix.
- [ ] **`gen_configs`** snapshot in the result matches what you passed. Useful for trial-registry logging (skill 11).
- [ ] **`n_paths` finished == `n_paths` requested.** Joblib doesn't truncate; if numbers don't match, something raised inside a worker.
- [ ] **Sensitivity table has all selected generators as rows.** Empty rows mean every path on that generator returned NaN — usually means the warmup ate the whole panel or the strategy returned all-zero weights.
- [ ] **Quantile spread (q05 → q95) is positive and reasonable.** Very narrow spread on `block_bootstrap` (e.g. q05 ≈ q95) with 1000 paths suggests the strategy has near-deterministic behavior on the path tensor — investigate before believing.

If any fails: state which. Do not proceed to Stage 9 (selection) with a broken MC result.

---

## 17. What NOT to do

- **Do not omit the `permutation` generator.** It's the only generator that diagnoses lookahead leaks. Always include it — `select_generators` does by default.
- **Do not skip the validator.** `MC_DESIGN.md` calls it a CI gate. Surface pass/fail in the cell. A generator that fails validation can still run, but its output is not trustworthy.
- **Do not interpret a high Sharpe on the `gaussian` generator as edge.** Gaussian returns have no exploitable structure beyond drift. If a strategy shows positive Sharpe on Gaussian, the user is reading drift, not skill.
- **Do not pass `costs=None`** to `MonteCarloEngine` unless the user has explicitly asked for a gross-vs-net stress test. `PROTOCOL.md §6`.
- **Do not assume `n_paths` of `MonteCarloEngine.run` is the same as `n_paths` from a prior `run_until_converged` run.** Adaptive stopping picks its own number based on SE; record it from `AdaptiveResult.n_paths_used` if reproducibility matters.
- **Do not run with `n_jobs=-1` while debugging a failing generator or strategy.** Switch to `n_jobs=1` so the underlying traceback comes through.
- **Do not pass `n_paths < 2`** to a Sharpe-stat aggregator — std and CIs degenerate. `MonteCarloEngine.run` accepts any positive `n_paths` but downstream metrics need ≥ 2.
- **Do not use `AntitheticAdapter` on a `BlockBootstrapGenerator` / `PermutationGenerator` / `HistoricalReplayGenerator`.** It raises `TypeError` because those don't expose `.mu`. Antithetic is for Gaussian-based generators only.
- **Do not set `LambertWTailAdapter(..., delta >= 0.25)`** — raises at construction. The math is undefined past that point (kurtosis goes infinite).
- **Do not pass `block_length=1`** to `BlockBootstrapGenerator` — it raises (block length must be positive). `1` would still be allowed but it degenerates to IID resampling, which destroys autocorrelation. Use `PermutationGenerator` if that's what you want.
- **Do not call `run_batch` and `MonteCarloEngine` in the same cell** and treat their outputs as equivalent. Batch skips costs / liquidity / risk; engine does not. They answer different questions.
- **Do not assume `MonteCarloResult.path_returns` is indexed like `result.returns`** from the engine. It's `{gen_name → DataFrame[date × path_i]}`. Two levels of indirection; use `mc.path_returns["block_bootstrap"]["path_0"]`.
- **Do not assume `aggregate_metrics()` covers every `MetricsReport` field.** It covers 19 metrics (the same set as `MultiPathResult.aggregate_metrics`). Excluded: `gross_pnl, sharpe_std, sharpe_ci_low/high, min_trl, n_obs, ann_factor` (CI fields don't aggregate cleanly across paths; the others are metadata or constants). Compute from `metrics_per_path()` if needed.
- **Do not mutate the prototype strategy between MC runs.** Each MC run deep-copies per path; the prototype itself is not mutated. But if you change `prototype.lookback = 90` between runs, subsequent runs see the new value and your previous result's `generator_configs` won't reflect it.
- **Do not skip `fingerprint` for a parameter sweep.** Each parameter combination is a different strategy and may have a different fingerprint. If you sweep without re-fingerprinting, the generator selection is wrong for most of the sweep.
- **Do not use `PathTensorCache` across machines without confirming the generator config serializes deterministically.** JSON keys are sorted (`json.dumps(..., sort_keys=True)`), and the config values are JSON-serializable scalars — should round-trip cleanly. Still: test before relying.
- **Do not assume the validator's `overall_passed` means "this generator is realistic enough".** It means "every score is in its threshold band". The bands are conservative defaults; relax explicitly if a fact doesn't matter, or tighten if it does.

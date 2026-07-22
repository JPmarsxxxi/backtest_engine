"""Build examples/end_to_end.ipynb from a list of cells (stdlib json, no nbformat)."""
import json
from pathlib import Path

CELLS = []


def _next_id() -> str:
    return f"cell_{len(CELLS):03d}"


def md(text: str) -> None:
    CELLS.append({
        "cell_type": "markdown",
        "id": _next_id(),
        "metadata": {},
        "source": text,
    })


def code(text: str) -> None:
    CELLS.append({
        "cell_type": "code",
        "execution_count": None,
        "id": _next_id(),
        "metadata": {},
        "outputs": [],
        "source": text,
    })


# === Section 1 — Setup =====================================================

md("""# Backtest Engine — End-to-End Walkthrough

This notebook demonstrates every category of the engine with detailed plots:
data + strategy + Engine + costs + risk + multi-path CPCV + all 21
`MetricsReport` fields + the entire Monte Carlo simulation layer
(8 generators, 3 adapters, validator, probes/fingerprint, cache, adaptive,
batch) + DSR.

Start Jupyter from the **repo root** or from `examples/` — the data-loading
cell tries both.

## 1. Setup""")

code("""import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from backtest.data import DataPanel
from backtest.engine import Engine, MultiPathEngine
from backtest.costs import (
    Commission, Spread, MarketImpact, ShortBorrow,
    CompositeCostModel, LiquidityCap,
)
from backtest.splitters import WalkForward, CombinatorialPurgedCV
from backtest.metrics import compute_metrics, MetricsReport, sharpe_ratio, psr
from backtest.metrics.plot import plot_sharpe_distribution
from backtest.strategy import Strategy
from backtest.risk import RiskConfig
from backtest.registry import TrialRegistry
from backtest.selection import dsr

from backtest.simulation import (
    GaussianGenerator, SobolGaussianGenerator, PermutationGenerator,
    HistoricalReplayGenerator, BlockBootstrapGenerator, GarchGenerator,
    MultivariateGenerator, SobolMultivariateGenerator,
    JumpOverlayAdapter, AntitheticAdapter, LambertWTailAdapter,
    MonteCarloEngine, PathTensorCache,
    GeneratorValidator, ProbeBattery, fingerprint,
    select_generators, run_until_converged, run_batch,
)

plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["figure.dpi"] = 100""")

code("""# Locate scratch data — works from repo root or from examples/.
for candidate in [Path("scratch/prices.parquet"), Path("../scratch/prices.parquet")]:
    if candidate.exists():
        DATA_DIR = candidate.parent
        break
else:
    raise FileNotFoundError(
        "Cannot find scratch/prices.parquet; "
        "start jupyter from repo root or examples/"
    )

prices = pd.read_parquet(DATA_DIR / "prices.parquet")
volume = pd.read_parquet(DATA_DIR / "volume.parquet")

print(f"Prices: {prices.shape[0]} bars × {prices.shape[1]} assets")
print(f"Date range: {prices.index[0].date()} → {prices.index[-1].date()}")
print(f"Assets: {list(prices.columns)}")
prices.head()""")

code("""panel = DataPanel(prices, volume=volume, check_outliers=False)
print(f"Panel: {len(panel.dates)} bars, {len(panel.assets_all)} assets")""")


# === Section 2 — Strategy ==================================================

md("""## 2. Strategy

Cross-sectional momentum: rank assets by past return, go long the top half
and short the bottom half. Implements both `generate_weights` (per-path,
used by `Engine`) and `generate_weights_batch` (vectorized, used by
`run_batch`).""")

code("""class XSMomentum(Strategy):
    \"\"\"Cross-sectional momentum, monthly rebalance, both interfaces.\"\"\"
    rebalance_frequency = "monthly"
    risk = RiskConfig(max_position=0.30, max_gross=1.0, max_net=1.0)

    def __init__(self, lookback: int = 60):
        self.lookback = lookback

    def __repr__(self):
        return f"XSMomentum(lookback={self.lookback})"

    def generate_weights(self, data, t):
        if len(data.prices) < self.lookback + 1:
            return pd.Series(0.0, index=data.assets)
        rets = data.prices.iloc[-1] / data.prices.iloc[-self.lookback - 1] - 1
        rets = rets.reindex(data.assets).dropna()
        if len(rets) < 2:
            return pd.Series(0.0, index=data.assets)
        median = rets.median()
        n = len(rets)
        w = pd.Series(0.0, index=rets.index)
        w[rets > median] = 1.0 / n
        w[rets < median] = -1.0 / n
        return w

    def generate_weights_batch(self, view, t):
        prices = view.prices  # (N, t+1, K)
        N, T_so_far, K = prices.shape
        if T_so_far < self.lookback + 1:
            return np.zeros((N, K))
        rets = prices[:, -1, :] / prices[:, -self.lookback - 1, :] - 1
        median = np.median(rets, axis=1, keepdims=True)
        w = np.zeros((N, K))
        w[rets > median] = 1.0 / K
        w[rets < median] = -1.0 / K
        return w

strat = XSMomentum(lookback=60)
print(strat)""")

code("""rebal = strat.rebalance_dates(panel.dates)
print(f"Rebalances: {len(rebal)} ({len(rebal) / (len(panel.dates) / 252):.1f}/year)")
print(f"First 5: {[d.date() for d in rebal[:5]]}")""")


# === Section 3 — Single-path Engine ========================================

md("""## 3. Single-Path Engine

`Engine.run()` returns a `BacktestResult`. Its `.summary()` plots a 2×2
dashboard (equity, drawdown, rolling Sharpe, asymptotic SR distribution)
and returns a `MetricsReport` with all 21 metrics.""")

code("""engine = Engine(panel, strat, initial_capital=1_000_000)
result = engine.run(start=panel.dates[252])  # 1y warmup
print(f"Bars: {len(result.equity_curve)}, Rebalances: {result.metadata['n_rebalances']}")
print(f"Final equity: ${result.equity_curve.iloc[-1]:,.0f}")""")

code("""report = result.summary()  # plots dashboard, returns MetricsReport""")

code("""# All 21 MetricsReport fields rendered as HTML table
report""")


# === Section 4 — Costs + Liquidity =========================================

md("""## 4. Costs + Liquidity

Composable cost model: commission + half-spread + market impact + short
borrow. Liquidity cap throttles trades to a fraction of recent ADV.""")

code("""costs = CompositeCostModel([
    Commission(bps=2),                                      # 2 bps commission
    Spread(half_bps=2),                                     # 2 bps half-spread
    MarketImpact(k=10, kind="sqrt", adv_lookback=20),       # sqrt impact
    ShortBorrow(annual_bps=300),                            # 3% annual borrow
])
liq = LiquidityCap(cap_pct=0.10, adv_lookback=20)           # 10% ADV cap

result_costs = Engine(panel, strat, costs=costs, liquidity=liq).run(start=panel.dates[252])

print(f"Total cost ($):       {result_costs.costs.sum():>14,.0f}")
print(f"Net final equity ($): {result_costs.equity_curve.iloc[-1]:>14,.0f}")
print(f"Gross final equity ($): {result.equity_curve.iloc[-1]:>12,.0f}")""")

code("""fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(result.equity_curve.index, result.equity_curve, label="gross (no costs)", lw=1.5)
ax.plot(result_costs.equity_curve.index, result_costs.equity_curve, label="net (with costs)", lw=1.5)
ax.axhline(1_000_000, color="gray", ls="--", lw=0.8)
ax.set_ylabel("equity ($)")
ax.set_title("Cost impact on equity curve")
ax.legend()
plt.show()""")


# === Section 5 — Risk Overlay ==============================================

md("""## 5. Risk Overlay

`RiskConfig` clips per-asset positions, gross/net exposure, applies vol
targeting, and can flatten positions on drawdown.""")

code("""class XSMomTV(XSMomentum):
    risk = RiskConfig(
        max_position=0.20,
        max_gross=1.0,
        max_net=1.0,
        target_vol=0.10,        # annualized vol target
        vol_lookback=60,
    )
    def __repr__(self):
        return f"XSMomTV(lookback={self.lookback})"

result_tv = Engine(panel, XSMomTV(60), costs=costs, liquidity=liq).run(start=panel.dates[252])

ann_vol_default = float(result_costs.returns.std() * np.sqrt(252))
ann_vol_tv = float(result_tv.returns.std() * np.sqrt(252))
print(f"Realized ann. vol — default:        {ann_vol_default:.2%}")
print(f"Realized ann. vol — vol-targeted:   {ann_vol_tv:.2%}")""")


# === Section 6 — CPCV Multi-Path ===========================================

md("""## 6. CPCV Multi-Path

Combinatorial purged cross-validation produces multiple equity paths from
one dataset. Purging removes train labels overlapping test; embargo drops
train bars immediately after test windows. `MultiPathResult.summary()`
plots a per-path equity overlay + per-path Sharpe distribution.""")

code("""cv = CombinatorialPurgedCV(n_splits=6, n_test_groups=2, purge_bars=21, embargo_pct=0.01)
mp = MultiPathEngine(
    panel, XSMomentum(60), cv,
    costs=costs, liquidity=liq, initial_capital=1_000_000,
)
mp_result = mp.run(dates=panel.dates[252:], n_jobs=1)
print(f"Splits: {mp_result.n_splits}, Paths: {mp_result.n_paths}")""")

code("""mp_agg = mp_result.summary()  # plots tearsheet, prints aggregate metrics""")


# === Section 7 — Metrics Deep-Dive =========================================

md("""## 7. Metrics Deep-Dive

`compute_metrics` produces a `MetricsReport` with: returns
(total/annualized/vol/PnL/costs/gross), risk-adjusted (Sharpe + SD + CI,
Sortino, Calmar, Modified Sharpe, PSR, MinTRL), tail (VaR, cVaR), drawdown
(max DD, time-under-water, longest underwater), and bookkeeping
(n_obs, ann_factor).""")

code("""rep = compute_metrics(
    result_costs.returns, result_costs.equity_curve,
    costs=result_costs.costs, ann_factor=252, ci_level=0.95,
)
rep""")

code("""fig, ax = plt.subplots(figsize=(10, 4.5))
plot_sharpe_distribution(rep.returns, ann_factor=252, ax=ax)
plt.show()""")


# === Section 8 — MC Layer ==================================================

md("""## 8. Monte Carlo Simulation Layer

### 8.1 Generator Tour

Each generator preserves a different set of statistical properties. Sample
paths from the same seed:""")

code("""real_returns = panel._prices.pct_change().dropna().values  # (T, K)
K = real_returns.shape[1]

generators = {
    "Gaussian":          GaussianGenerator.fit(real_returns),
    "SobolGaussian":     SobolGaussianGenerator.fit(real_returns),
    "Permutation":       PermutationGenerator(real_returns),
    "HistoricalReplay":  HistoricalReplayGenerator(real_returns),
    "BlockBootstrap":    BlockBootstrapGenerator.fit(real_returns),
    "Garch":             GarchGenerator.fit(real_returns),
    "Multivariate":      MultivariateGenerator.fit(real_returns),
    "SobolMultivariate": SobolMultivariateGenerator.fit(real_returns),
}

fig, axes = plt.subplots(4, 2, figsize=(13, 10), sharex=True)
for ax, (name, gen) in zip(axes.flat, generators.items()):
    paths = gen.sample(n_paths=1, n_steps=500, n_assets=K, seed=0)
    prices_synth = 100.0 * np.cumprod(1 + paths[0, :, 0])
    ax.plot(prices_synth, lw=1.0)
    ax.set_title(name)
    ax.set_ylabel("price")
fig.suptitle("Sample path (asset 0) from each generator", y=1.01)
fig.tight_layout()
plt.show()""")

code("""# Adapters: same Gaussian base, different tail behavior
base = GaussianGenerator.fit(real_returns)
adapters = {
    "Plain Gaussian":          base.sample(2, 500, K, seed=0)[0, :, 0],
    "+ JumpOverlay":           JumpOverlayAdapter(base, jump_intensity=0.02, jump_sigma=0.05).sample(2, 500, K, seed=0)[0, :, 0],
    "+ LambertWTail (delta=0.18)": LambertWTailAdapter(base, delta=0.18).sample(2, 500, K, seed=0)[0, :, 0],
    "+ Antithetic":            AntitheticAdapter(base).sample(2, 500, K, seed=0)[0, :, 0],
}

fig, axes = plt.subplots(2, 2, figsize=(13, 6))
for ax, (name, returns) in zip(axes.flat, adapters.items()):
    ax.hist(returns, bins=60, alpha=0.7, color="C0")
    ax.set_title(name)
    ax.set_xlabel("daily return")
fig.suptitle("Return distributions: same base, different adapters", y=1.01)
fig.tight_layout()
plt.show()""")


md("""### 8.2 Generator Validation

`GeneratorValidator` scores each generator on 10 stylized-facts metrics:
mean_z, std_ratio, skew_diff, kurt_ratio, ks_pvalue, hill_tail_diff,
acf_r_lag1_diff, acf_abs_r_lag1_diff, ljung_box_r2_pvalue,
corr_frobenius_ratio. Pass/fail vs default thresholds.""")

code("""val = GeneratorValidator(real_returns, n_paths=10, seed=0)

val_results = {}
for name, gen in generators.items():
    val_results[name] = val.validate(gen)

score_summary = pd.DataFrame({name: r.passed for name, r in val_results.items()}).T
score_summary["overall_passed"] = score_summary.all(axis=1)
score_summary""")

code("""fig = val_results["BlockBootstrap"].plot()
fig.suptitle("BlockBootstrap — validator tearsheet", y=1.02)
plt.show()""")

code("""fig = val_results["Gaussian"].plot()
fig.suptitle("Gaussian — validator tearsheet (note: ljung_box_r2_pvalue FAIL)", y=1.02)
plt.show()""")


md("""### 8.3 Probe Battery + Strategy Fingerprint

The probe battery is 6 deterministic synthetic datasets, each isolating one
market property: `iid_gaussian` (no signal), `momentum` (persistent alphas),
`mean_reversion` (price reversion), `vol_clustering` (GARCH-like),
`jump_diffusion` (Poisson jumps), `cross_section` (block correlation).
Running the strategy on each tells us what kind of edge it has.""")

code("""battery = ProbeBattery(n_assets=5, n_steps=520, seed=42)
probes = battery.probes()

fig, axes = plt.subplots(2, 3, figsize=(13, 6))
for ax, probe in zip(axes.flat, probes):
    p = probe.panel._prices
    for col in p.columns:
        ax.plot(p.index, p[col] / p[col].iloc[0], lw=0.8, alpha=0.7)
    ax.set_title(f"{probe.name}\\n({probe.isolates})")
    ax.set_ylabel("normalized price")
fig.suptitle("Probe battery — 6 deterministic synthetic datasets", y=1.01)
fig.tight_layout()
plt.show()""")

code("""scores = fingerprint(XSMomentum(60), battery=battery)

fig, ax = plt.subplots(figsize=(10, 4))
names = list(scores.keys())
vals = list(scores.values())
colors = ["gray" if v == 0 else ("C2" if v > 0 else "C3") for v in vals]
ax.bar(names, vals, color=colors)
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("Sharpe relative to IID baseline")
ax.set_title("XSMomentum(60) fingerprint — delta-Sharpe per probe")
plt.xticks(rotation=15)
plt.show()
print({k: round(v, 3) for k, v in scores.items()})""")

code("""auto_gens = select_generators(scores, real_returns, threshold=0.5)
print(f"Auto-selected: {list(auto_gens.keys())}")""")


md("""### 8.4 MonteCarloEngine + Path Cache

Run the strategy on N synthetic paths from multiple generators.
`PathTensorCache` memoizes path tensors keyed by (generator config, seed,
dims) so reruns hit the cache.""")

code("""cache = PathTensorCache(Path("./mc_cache"))
cache.clear()

# Use a date slice shorter than the real-returns history so HistoricalReplay
# has multiple distinct windows to sample from (otherwise it returns the same
# window every time).
mc_dates = panel.dates[:1000]

mc_gens = {
    "gaussian":        GaussianGenerator.fit(real_returns),
    "historical":      HistoricalReplayGenerator(real_returns),
    "block_bootstrap": BlockBootstrapGenerator.fit(real_returns),
}
mc_engine = MonteCarloEngine(
    strategy=XSMomentum(60),
    generators=mc_gens,
    assets=list(panel.assets_all),
    dates=mc_dates,
    initial_capital=1_000_000,
    warmup_bars=252,
    path_cache=cache,
)
mc_result = mc_engine.run(n_paths=50, seed=0, n_jobs=1)
print(f"Cache after first run: {cache.stats()}")""")

code("""mc_result.summary()  # sensitivity table per generator""")

code("""fig = mc_result.plot()  # equity overlays + Sharpe dist + MaxDD dist
plt.show()""")


md("""### 8.5 Adaptive Convergence

Stop sampling when the metric's standard error drops below a threshold
instead of running a fixed N. Typically 2–5× cheaper on average.""")

code("""adaptive = run_until_converged(
    mc_engine,
    metric="sharpe",
    target_se=0.10,
    batch_size=20,
    min_paths=40,
    max_paths=200,
    seed=100,
    n_jobs=1,
)
print(f"Converged: {adaptive.converged}, paths used: {adaptive.n_paths_used}")

fig, ax = plt.subplots(figsize=(10, 4))
for name, trace in adaptive.se_trace.items():
    xs = np.arange(1, len(trace) + 1) * 20 + 20
    ax.plot(xs, trace, marker="o", label=name)
ax.axhline(adaptive.target_se, color="gray", ls="--", lw=0.8, label="target SE")
ax.set_xlabel("paths accumulated")
ax.set_ylabel(f"SE of {adaptive.metric}")
ax.set_title("Adaptive convergence trace")
ax.legend()
plt.show()""")


md("""### 8.6 Batch Fast Path

For strategies that implement `generate_weights_batch`, `run_batch` runs all
paths in vectorized matrix operations. Bit-equivalent to per-path Engine
when there are no costs / liquidity / risk overlay.""")

code("""gen = BlockBootstrapGenerator.fit(real_returns)
paths_tensor = gen.sample(
    n_paths=20, n_steps=len(mc_dates), n_assets=K, seed=0,
)

t0 = time.perf_counter()
batch_result = run_batch(
    XSMomentum(60), paths_tensor,
    assets=list(panel.assets_all), dates=mc_dates,
    warmup_bars=252, initial_capital=1_000_000,
)
batch_time = time.perf_counter() - t0
print(f"Batch (N=20): {batch_time:.2f}s")""")

code("""batch_agg = batch_result.summary()""")

code("""fig = batch_result.plot()
plt.show()""")


# === Section 9 — Selection / DSR ===========================================

md("""## 9. Selection Bias + DSR

Trying many strategies inflates the chance of finding a spurious edge. The
Deflated Sharpe Ratio (DSR) accounts for the number of trials. Compare PSR
vs DSR across a parameter sweep.""")

code("""sweep = [30, 45, 60, 90, 120, 180]
trial_results = []
for lb in sweep:
    r = Engine(panel, XSMomentum(lookback=lb), costs=costs, liquidity=liq).run(
        start=panel.dates[252]
    )
    rets = r.returns.dropna().values
    trial_results.append({
        "lookback": lb,
        "sharpe":   sharpe_ratio(rets, ann_factor=252),
        "psr_vs_0": psr(rets, sr_star=0.0, ann_factor=252),
        "dsr_K":    dsr(rets, K=len(sweep)),
    })

df = pd.DataFrame(trial_results)
print(df.round(3).to_string(index=False))""")

code("""fig, ax = plt.subplots(figsize=(10, 4))
x = np.arange(len(df))
w = 0.35
ax.bar(x - w/2, df["psr_vs_0"], w, label="PSR (vs 0)", color="C0")
ax.bar(x + w/2, df["dsr_K"], w, label=f"DSR (K={len(sweep)})", color="C3")
ax.set_xticks(x)
ax.set_xticklabels([f"lb={lb}" for lb in df["lookback"]])
ax.set_ylabel("probability")
ax.set_title("PSR vs DSR — multiple-testing deflation")
ax.legend()
ax.set_ylim(0, 1)
plt.show()""")

code("""# Optional: register trials via TrialRegistry — supports effective-K via clustering
import tempfile

with tempfile.TemporaryDirectory() as tmpdir:
    reg = TrialRegistry(tmpdir)
    fake_graph = Path(tmpdir) / "graph.png"
    fake_graph.write_bytes(b"fake")  # registry requires a causal-graph artifact

    for lb in sweep:
        reg.run(
            Engine(panel, XSMomentum(lookback=lb), costs=costs, liquidity=liq),
            causal_graph_path=str(fake_graph),
            family="xs_momentum",
            start=panel.dates[252],
        )

    K_count = reg.k(family="xs_momentum")
    K_eff = reg.k(family="xs_momentum", method="effective", threshold=0.7)
    print(f"K registered: {K_count}, effective K (correlation-clustered): {K_eff}")""")


# === Section 10 — Takeaways ================================================

md("""## 10. Takeaways

You've now seen every category of the engine + the full Monte Carlo
simulation layer end-to-end:

- **Data**: `DataPanel` with PIT enforcement, missing-data policies,
  universe filtering.
- **Strategy**: `Strategy` subclass with `generate_weights` (per-path) and
  optionally `generate_weights_batch` (vectorized).
- **Engine**: single-path `Engine.run()` returning `BacktestResult` with
  full `MetricsReport`.
- **Costs**: `Commission`, `Spread`, `MarketImpact`, `ShortBorrow`
  composable via `CompositeCostModel`; `LiquidityCap` for ADV constraints.
- **Risk**: `RiskConfig` for per-asset/gross/net caps, vol targeting,
  drawdown breaker.
- **Splitters**: `WalkForward`, `CombinatorialPurgedCV` with purging and
  embargo.
- **MultiPathEngine**: parallel CPCV paths via joblib.
- **Metrics**: 21-field `MetricsReport`; PSR, DSR, MinTRL with confidence
  intervals.
- **MC layer**: 8 generators × 3 adapters, `GeneratorValidator`
  (10 stylized facts), `ProbeBattery` + `fingerprint` + `select_generators`,
  `MonteCarloEngine` + `PathTensorCache`, `run_until_converged`,
  `run_batch` (bit-equivalent vectorized fast path).
- **Selection**: DSR-deflated Sharpe accounting for multiple trials,
  effective-K via correlation clustering in the registry.

For deeper documentation see `MC_DESIGN.md` (Monte Carlo design) and
`STRATEGY_GUIDE.md` (strategy-authoring patterns).""")


# === Build and write =======================================================

NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.13"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT = Path(__file__).parent / "end_to_end.ipynb"
OUT.write_text(json.dumps(NOTEBOOK, indent=1))
print(f"Wrote {OUT} with {len(CELLS)} cells")

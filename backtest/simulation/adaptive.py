from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest.simulation.mc import MonteCarloEngine, MonteCarloResult


@dataclass
class AdaptiveResult:
    result: MonteCarloResult
    converged: bool
    n_paths_used: int
    se_trace: dict[str, list[float]]
    metric: str
    target_se: float


def run_until_converged(
    engine: MonteCarloEngine,
    metric: str = "sharpe",
    target_se: float = 0.05,
    batch_size: int = 100,
    max_paths: int = 10_000,
    min_paths: int = 200,
    seed: int = 0,
    n_jobs: int = -1,
    verbose: bool = False,
) -> AdaptiveResult:
    """Run paths in batches until metric SE < target_se across all generators.

    SE = std(metric across paths, ddof=1) / sqrt(n). Stops when max SE across
    generators falls below target_se, or n_paths reaches max_paths.
    """
    from backtest.metrics import compute_metrics

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_paths < batch_size:
        raise ValueError("max_paths must be >= batch_size")
    if min_paths < 2:
        raise ValueError("min_paths must be >= 2")
    if target_se <= 0:
        raise ValueError("target_se must be positive")

    rets_acc: dict[str, list[pd.Series]] = {n: [] for n in engine.generators}
    eq_acc: dict[str, list[pd.Series]] = {n: [] for n in engine.generators}
    metric_acc: dict[str, list[float]] = {n: [] for n in engine.generators}
    se_trace: dict[str, list[float]] = {n: [] for n in engine.generators}

    n_done = 0
    converged = False
    next_seed = seed

    while n_done < max_paths:
        this_batch = min(batch_size, max_paths - n_done)
        batch = engine.run(n_paths=this_batch, seed=next_seed, n_jobs=n_jobs)
        # offset next batch's seed by a large prime times the batch size so paths don't repeat.
        next_seed += this_batch * 1_000_003

        for name in engine.generators:
            for col in batch.path_returns[name].columns:
                r = batch.path_returns[name][col].dropna()
                e = batch.path_equity[name][col].dropna()
                rets_acc[name].append(batch.path_returns[name][col])
                eq_acc[name].append(batch.path_equity[name][col])
                v = getattr(compute_metrics(r, e), metric)
                metric_acc[name].append(float(v))

        n_done += this_batch

        if n_done < min_paths:
            continue

        max_se = 0.0
        for name in engine.generators:
            vals = np.array(
                [v for v in metric_acc[name] if np.isfinite(v)], dtype=np.float64
            )
            if len(vals) > 1:
                se = float(vals.std(ddof=1) / np.sqrt(len(vals)))
            else:
                se = float("inf")
            se_trace[name].append(se)
            if se > max_se:
                max_se = se

        if verbose:
            print(f"[adaptive] n={n_done} max_SE({metric})={max_se:.5f}")

        if max_se < target_se:
            converged = True
            break

    final = _build_result(engine, rets_acc, eq_acc, n_done, seed)
    return AdaptiveResult(
        result=final,
        converged=converged,
        n_paths_used=n_done,
        se_trace=se_trace,
        metric=metric,
        target_se=target_se,
    )


def _build_result(
    engine: MonteCarloEngine,
    rets_acc: dict[str, list[pd.Series]],
    eq_acc: dict[str, list[pd.Series]],
    n_paths: int,
    seed: int,
) -> MonteCarloResult:
    rets_df = {}
    eq_df = {}
    for name in engine.generators:
        rets_df[name] = pd.DataFrame(
            {f"path_{i}": s for i, s in enumerate(rets_acc[name])}
        )
        eq_df[name] = pd.DataFrame(
            {f"path_{i}": s for i, s in enumerate(eq_acc[name])}
        )
    return MonteCarloResult(
        path_returns=rets_df,
        path_equity=eq_df,
        initial_capital=engine.initial_capital,
        n_paths=n_paths,
        seed=seed,
        generator_configs={n: g.config() for n, g in engine.generators.items()},
    )

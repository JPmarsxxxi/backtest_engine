from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from backtest.costs.base import CostModel
from backtest.costs.liquidity import LiquidityCap
from backtest.engine.engine import Engine
from backtest.simulation.base import PathGenerator, panel_from_returns
from backtest.simulation.cache import PathTensorCache
from backtest.strategy.base import Strategy

StrategyOrFactory = Union[Strategy, Callable[[], Strategy]]

# Per-generator seed offset; large prime to avoid noise overlap across small base seeds.
_GEN_SEED_STRIDE = 10_000_007


@dataclass
class MonteCarloResult:
    path_returns: dict[str, pd.DataFrame]      # gen_name -> DataFrame[date x path_i]
    path_equity: dict[str, pd.DataFrame]
    initial_capital: float
    n_paths: int
    seed: int
    generator_configs: dict[str, dict]

    def metrics_per_path(self) -> dict[str, list]:
        from backtest.metrics.report import compute_metrics

        out: dict[str, list] = {}
        for name, rets_df in self.path_returns.items():
            eq_df = self.path_equity[name]
            reports = []
            for col in rets_df.columns:
                ret = rets_df[col].dropna()
                eq = eq_df[col].dropna()
                reports.append(compute_metrics(ret, eq))
            out[name] = reports
        return out

    def aggregate_metrics(self) -> dict[str, dict[str, float]]:
        all_reports = self.metrics_per_path()
        keys = [
            "sharpe", "sortino", "calmar", "modified_sharpe", "psr",
            "max_drawdown", "time_under_water", "var_5", "cvar_5",
            "ann_return", "ann_vol", "total_return", "total_pnl",
            "total_costs", "longest_underwater",
            "hit_rate", "turnover", "information_ratio", "beta",
        ]
        out: dict[str, dict[str, float]] = {}
        for name, reports in all_reports.items():
            d: dict[str, float] = {}
            for k in keys:
                arr = np.asarray([getattr(r, k) for r in reports], dtype=float)
                arr = arr[np.isfinite(arr)]
                if len(arr) == 0:
                    d[f"{k}_mean"] = float("nan")
                    d[f"{k}_std"] = float("nan")
                    d[f"{k}_min"] = float("nan")
                    d[f"{k}_max"] = float("nan")
                else:
                    d[f"{k}_mean"] = float(arr.mean())
                    d[f"{k}_std"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
                    d[f"{k}_min"] = float(arr.min())
                    d[f"{k}_max"] = float(arr.max())
            out[name] = d
        return out

    def sensitivity_table(
        self,
        metrics: Sequence[str] = ("sharpe", "sortino", "max_drawdown", "psr"),
        quantiles: Sequence[float] = (0.05, 0.50, 0.95),
    ) -> pd.DataFrame:
        all_reports = self.metrics_per_path()
        rows = []
        idx = []
        for name, reports in all_reports.items():
            row: dict[str, float] = {}
            for m in metrics:
                arr = np.asarray([getattr(r, m) for r in reports], dtype=float)
                arr = arr[np.isfinite(arr)]
                if len(arr) == 0:
                    row[f"{m}_mean"] = float("nan")
                    for q in quantiles:
                        row[f"{m}_q{int(round(q * 100)):02d}"] = float("nan")
                else:
                    row[f"{m}_mean"] = float(arr.mean())
                    for q in quantiles:
                        row[f"{m}_q{int(round(q * 100)):02d}"] = float(np.quantile(arr, q))
            rows.append(row)
            idx.append(name)
        return pd.DataFrame(rows, index=pd.Index(idx, name="generator"))

    def summary(
        self,
        metrics: Sequence[str] = ("sharpe", "sortino", "max_drawdown", "psr"),
        quantiles: Sequence[float] = (0.05, 0.50, 0.95),
    ) -> pd.DataFrame:
        """Print per-generator sensitivity table; return the DataFrame."""
        table = self.sensitivity_table(metrics=metrics, quantiles=quantiles)
        gens = list(self.path_returns.keys())
        print(
            f"Monte Carlo summary: {self.n_paths} paths × {len(gens)} generators "
            f"({', '.join(gens)})"
        )
        print(table.round(4).to_string())
        return table

    def plot(self, figsize: tuple = (13, 8)):
        """Multi-panel matplotlib tearsheet: per-generator equity overlays
        (top), Sharpe distribution (middle), max-drawdown distribution (bottom).
        Returns the matplotlib Figure.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "matplotlib is required for plot(). "
                "Install with: pip install backtest[viz]"
            ) from e

        gens = list(self.path_returns.keys())
        n_gens = max(len(gens), 1)
        reports = self.metrics_per_path()

        fig = plt.figure(figsize=figsize)
        for i, name in enumerate(gens):
            ax = plt.subplot2grid((3, n_gens), (0, i), fig=fig)
            eq_df = self.path_equity[name]
            for col in eq_df.columns:
                ax.plot(eq_df.index, eq_df[col], color="C0", alpha=0.25, lw=0.7)
            median = eq_df.median(axis=1)
            ax.plot(eq_df.index, median, color="C3", lw=1.5, label="median")
            ax.axhline(self.initial_capital, color="gray", ls="--", lw=0.8)
            ax.set_title(name)
            if i == 0:
                ax.set_ylabel("equity")
            ax.tick_params(axis="x", rotation=30)

        ax_sr = plt.subplot2grid((3, n_gens), (1, 0), colspan=n_gens, fig=fig)
        for name in gens:
            sharpes = np.array(
                [r.sharpe for r in reports[name] if np.isfinite(r.sharpe)]
            )
            if len(sharpes) > 0:
                ax_sr.hist(
                    sharpes, bins=max(5, len(sharpes) // 2),
                    alpha=0.5, label=name,
                )
        ax_sr.axvline(0, color="gray", ls="--", lw=0.8)
        ax_sr.set_xlabel("annualized Sharpe")
        ax_sr.set_ylabel("count")
        ax_sr.set_title("Per-generator Sharpe distribution")
        ax_sr.legend(loc="best", frameon=False)

        ax_dd = plt.subplot2grid((3, n_gens), (2, 0), colspan=n_gens, fig=fig)
        for name in gens:
            dds = np.array(
                [r.max_drawdown for r in reports[name]
                 if np.isfinite(r.max_drawdown)]
            )
            if len(dds) > 0:
                ax_dd.hist(
                    dds, bins=max(5, len(dds) // 2),
                    alpha=0.5, label=name,
                )
        ax_dd.set_xlabel("max drawdown")
        ax_dd.set_ylabel("count")
        ax_dd.set_title("Per-generator max drawdown distribution")
        ax_dd.legend(loc="best", frameon=False)

        fig.tight_layout()
        return fig


class MonteCarloEngine:
    """Runs a strategy across many synthetic paths from one or more generators.

    Path tensors are drawn vectorized per generator from a deterministic
    per-generator seed. Each (generator, path) is dispatched as one joblib job;
    each worker materializes the panel, deepcopies the strategy, and runs Engine.

    CRN: same `seed` produces identical path tensors per generator. Two strategies
    instantiated with the same generators dict and run with the same seed see
    bit-identical paths -- no extra wiring needed.

    path_cache (optional): when provided, path tensors are read from disk on hit
    and written on miss. Lets you reuse identical paths across runs (e.g.,
    comparing strategies, iterating on analysis).
    """

    def __init__(
        self,
        strategy: StrategyOrFactory,
        generators: dict[str, PathGenerator],
        assets: Sequence[str],
        dates: pd.DatetimeIndex,
        costs: Optional[CostModel] = None,
        liquidity: Optional[LiquidityCap] = None,
        initial_capital: float = 1_000_000.0,
        warmup_bars: int = 0,
        path_cache: Optional[PathTensorCache] = None,
    ):
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if not generators:
            raise ValueError("generators dict must be non-empty")
        if warmup_bars < 0:
            raise ValueError("warmup_bars must be >= 0")
        if warmup_bars >= len(dates):
            raise ValueError("warmup_bars must be < len(dates)")

        self.strategy = strategy
        self.generators = dict(generators)
        self.assets = list(assets)
        self.dates = pd.DatetimeIndex(dates)
        self.costs = costs
        self.liquidity = liquidity
        self.initial_capital = float(initial_capital)
        self.warmup_bars = warmup_bars
        self.path_cache = path_cache
        self._is_factory = callable(strategy) and not isinstance(strategy, Strategy)

    def run(
        self,
        n_paths: int = 1000,
        seed: int = 0,
        n_jobs: int = -1,
    ) -> MonteCarloResult:
        if n_paths <= 0:
            raise ValueError("n_paths must be positive")

        n_assets = len(self.assets)
        n_steps = len(self.dates)

        gen_tensors: dict[str, np.ndarray] = {}
        gen_configs: dict[str, dict] = {}
        for i, (name, gen) in enumerate(self.generators.items()):
            effective_seed = seed + i * _GEN_SEED_STRIDE
            tensor = None
            if self.path_cache is not None:
                tensor = self.path_cache.get(
                    gen, effective_seed, n_paths, n_steps, n_assets
                )
            if tensor is None:
                tensor = gen.sample(
                    n_paths=n_paths,
                    n_steps=n_steps,
                    n_assets=n_assets,
                    seed=effective_seed,
                )
                if self.path_cache is not None:
                    self.path_cache.put(
                        gen, effective_seed, n_paths, n_steps, n_assets, tensor
                    )
            if tensor.shape != (n_paths, n_steps, n_assets):
                raise ValueError(
                    f"generator {name!r} returned shape {tensor.shape}, "
                    f"expected ({n_paths}, {n_steps}, {n_assets})"
                )
            gen_tensors[name] = tensor
            gen_configs[name] = gen.config()

        args = []
        for name, tensor in gen_tensors.items():
            for p in range(n_paths):
                args.append((
                    name,
                    p,
                    tensor[p],
                    self.assets,
                    self.dates,
                    self.strategy,
                    self._is_factory,
                    self.costs,
                    self.liquidity,
                    self.initial_capital,
                    self.warmup_bars,
                ))

        if n_jobs == 1:
            results = [_run_one_path(*a) for a in args]
        else:
            results = Parallel(n_jobs=n_jobs)(
                delayed(_run_one_path)(*a) for a in args
            )

        rets_by_gen: dict[str, dict[str, pd.Series]] = {n: {} for n in self.generators}
        eq_by_gen: dict[str, dict[str, pd.Series]] = {n: {} for n in self.generators}
        for (name, p, ret, eq) in results:
            col = f"path_{p}"
            rets_by_gen[name][col] = ret
            eq_by_gen[name][col] = eq

        path_returns_df = {n: _build_sorted_df(d) for n, d in rets_by_gen.items()}
        path_equity_df = {n: _build_sorted_df(d) for n, d in eq_by_gen.items()}

        return MonteCarloResult(
            path_returns=path_returns_df,
            path_equity=path_equity_df,
            initial_capital=self.initial_capital,
            n_paths=n_paths,
            seed=seed,
            generator_configs=gen_configs,
        )


def _build_sorted_df(cols: dict[str, pd.Series]) -> pd.DataFrame:
    ordered = sorted(cols.keys(), key=lambda s: int(s.split("_")[1]))
    return pd.DataFrame({c: cols[c] for c in ordered})


def _run_one_path(
    gen_name: str,
    path_idx: int,
    returns: np.ndarray,
    assets: Sequence[str],
    dates: pd.DatetimeIndex,
    strategy_or_factory: StrategyOrFactory,
    is_factory: bool,
    costs: Optional[CostModel],
    liquidity: Optional[LiquidityCap],
    initial_capital: float,
    warmup_bars: int,
) -> tuple[str, int, pd.Series, pd.Series]:
    panel = panel_from_returns(returns, assets, dates)
    if is_factory:
        strategy = strategy_or_factory()
    else:
        strategy = copy.deepcopy(strategy_or_factory)
    engine = Engine(
        data=panel,
        strategy=strategy,
        costs=costs,
        liquidity=liquidity,
        initial_capital=initial_capital,
    )
    start = panel.dates[warmup_bars] if warmup_bars > 0 else None
    result = engine.run(start=start)
    return gen_name, path_idx, result.returns, result.equity_curve

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from backtest.simulation.mc import MonteCarloResult
from backtest.strategy.base import Strategy


class BatchDataView:
    """Read-only batch snapshot at bar t.

    Mirrors DataView.prices but as ndarray (N, T_so_far, K) instead of
    DataFrame. Strategies that opt into batch mode read from this in their
    generate_weights_batch(view, t) method.
    """

    __slots__ = ("_all_prices", "t", "assets")

    def __init__(self, all_prices: np.ndarray, t: int, assets: Sequence[str]):
        self._all_prices = all_prices
        self.t = int(t)
        self.assets = list(assets)

    @property
    def prices(self) -> np.ndarray:
        return self._all_prices[:, : self.t + 1, :]


@dataclass
class BatchResult:
    path_returns: np.ndarray            # (N, T - warmup_bars)
    path_equity: np.ndarray             # (N, T - warmup_bars)
    weights: np.ndarray                 # (N, T - warmup_bars, K)
    dates: pd.DatetimeIndex             # length T - warmup_bars
    initial_capital: float
    n_paths: int

    def to_monte_carlo_result(
        self,
        gen_name: str = "batch",
        seed: int = 0,
        generator_configs: Optional[dict] = None,
    ) -> MonteCarloResult:
        """Repackage as MonteCarloResult for downstream sensitivity_table etc."""
        cols = [f"path_{i}" for i in range(self.n_paths)]
        rets_df = pd.DataFrame(self.path_returns.T, index=self.dates, columns=cols)
        eq_df = pd.DataFrame(self.path_equity.T, index=self.dates, columns=cols)
        cfgs = generator_configs if generator_configs is not None else {gen_name: {}}
        return MonteCarloResult(
            path_returns={gen_name: rets_df},
            path_equity={gen_name: eq_df},
            initial_capital=self.initial_capital,
            n_paths=self.n_paths,
            seed=seed,
            generator_configs=cfgs,
        )

    def summary(self) -> dict:
        """Print per-path aggregate stats; return aggregate metrics dict."""
        from backtest.metrics import compute_metrics

        reports = []
        for p in range(self.n_paths):
            r = pd.Series(self.path_returns[p], index=self.dates).dropna()
            e = pd.Series(self.path_equity[p], index=self.dates).dropna()
            reports.append(compute_metrics(r, e))

        keys = ["sharpe", "sortino", "calmar", "max_drawdown", "psr"]
        agg: dict = {}
        for k in keys:
            arr = np.array(
                [getattr(r, k) for r in reports if np.isfinite(getattr(r, k))]
            )
            if len(arr) > 0:
                agg[f"{k}_mean"] = float(arr.mean())
                agg[f"{k}_std"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
                agg[f"{k}_min"] = float(arr.min())
                agg[f"{k}_max"] = float(arr.max())
            else:
                for s in ("mean", "std", "min", "max"):
                    agg[f"{k}_{s}"] = float("nan")

        print(f"BatchResult: {self.n_paths} paths × {len(self.dates)} bars")
        for k in keys:
            print(
                f"  {k:<14s} mean={agg[f'{k}_mean']:+.4f}  "
                f"std={agg[f'{k}_std']:.4f}  "
                f"min={agg[f'{k}_min']:+.4f}  max={agg[f'{k}_max']:+.4f}"
            )
        return agg

    def plot(self, figsize: tuple = (13, 4.5)):
        """1×2 dashboard: per-path equity overlay + per-path Sharpe distribution.
        Returns the matplotlib Figure.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "matplotlib is required for plot(). "
                "Install with: pip install backtest[viz]"
            ) from e
        from backtest.metrics import compute_metrics

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        for p in range(self.n_paths):
            axes[0].plot(
                self.dates, self.path_equity[p],
                color="C0", alpha=0.35, lw=0.9,
            )
        axes[0].axhline(self.initial_capital, color="gray", ls="--", lw=0.8)
        axes[0].set_title(f"Per-path equity ({self.n_paths} paths)")
        axes[0].set_ylabel("equity")

        sharpes = []
        for p in range(self.n_paths):
            r = pd.Series(self.path_returns[p], index=self.dates).dropna()
            e = pd.Series(self.path_equity[p], index=self.dates).dropna()
            rep = compute_metrics(r, e)
            if np.isfinite(rep.sharpe):
                sharpes.append(rep.sharpe)
        sharpes = np.array(sharpes)
        if len(sharpes) > 0:
            axes[1].hist(
                sharpes, bins=max(5, len(sharpes) // 2),
                color="C0", alpha=0.7, edgecolor="white",
            )
            axes[1].axvline(
                sharpes.mean(), color="C3", lw=2.0,
                label=f"mean = {sharpes.mean():.2f}",
            )
        axes[1].axvline(0, color="gray", ls="--", lw=0.8)
        axes[1].set_xlabel("annualized Sharpe")
        axes[1].set_ylabel("count")
        axes[1].set_title("Per-path Sharpe distribution")
        axes[1].legend(loc="best", frameon=False, fontsize=8)

        fig.tight_layout()
        return fig


def run_batch(
    strategy: Strategy,
    paths: np.ndarray,
    assets: Sequence[str],
    dates: pd.DatetimeIndex,
    warmup_bars: int = 0,
    initial_capital: float = 1_000_000.0,
    init_price: float = 100.0,
) -> BatchResult:
    """Vectorized backtest of a batch-capable strategy across N paths.

    Strategy must define generate_weights_batch(view, t) -> ndarray (N, K).
    Constraints in exchange for speed: no costs, no liquidity caps,
    no risk overlay, all assets always eligible. paths is (N, T, K) of
    simple returns; output is sliced to [warmup_bars:] for parity with
    MonteCarloEngine.
    """
    if not hasattr(strategy, "generate_weights_batch"):
        raise TypeError(
            f"{type(strategy).__name__} does not implement generate_weights_batch; "
            "batch mode requires the strategy to opt in."
        )
    if paths.ndim != 3:
        raise ValueError(f"paths must be 3-D (N, T, K), got {paths.shape}")
    N, T, K = paths.shape
    if T != len(dates):
        raise ValueError(f"paths has {T} steps, dates has {len(dates)}")
    if K != len(assets):
        raise ValueError(f"paths has {K} assets, assets has {len(assets)}")
    if warmup_bars < 0 or warmup_bars >= T:
        raise ValueError(f"warmup_bars must be in [0, {T})")
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive")

    dates = pd.DatetimeIndex(dates)
    prices = init_price * np.cumprod(1.0 + paths, axis=1)  # (N, T, K)

    # Use post-warmup dates for rebalance bar selection so the first post-warmup
    # bar IS a rebalance — matches MonteCarloEngine's behavior under start=.
    post_warmup_dates = dates[warmup_bars:]
    rebal = pd.DatetimeIndex(strategy.rebalance_dates(post_warmup_dates))
    rebalance_mask = dates.isin(rebal)

    positions = np.zeros((N, K), dtype=np.float64)
    cash = np.full(N, float(initial_capital), dtype=np.float64)
    eq = np.empty((N, T), dtype=np.float64)
    rets = np.zeros((N, T), dtype=np.float64)
    weights_hist = np.zeros((N, T, K), dtype=np.float64)

    for t in range(T):
        if t > 0:
            with np.errstate(divide="ignore", invalid="ignore"):
                ratios = prices[:, t, :] / prices[:, t - 1, :]
            ratios = np.where(np.isfinite(ratios), ratios, 1.0)
            positions = positions * ratios

        equity = positions.sum(axis=1) + cash
        eq[:, t] = equity

        if t > 0:
            with np.errstate(divide="ignore", invalid="ignore"):
                rets[:, t] = (eq[:, t] - eq[:, t - 1]) / np.where(
                    eq[:, t - 1] != 0, eq[:, t - 1], 1.0
                )

        if t < warmup_bars:
            continue

        if rebalance_mask[t]:
            view = BatchDataView(prices, t, assets)
            target_w = strategy.generate_weights_batch(view, dates[t])
            target_w = np.asarray(target_w, dtype=np.float64)
            if target_w.shape != (N, K):
                raise ValueError(
                    f"generate_weights_batch returned shape {target_w.shape}, "
                    f"expected ({N}, {K})"
                )
            target_dollars = target_w * equity[:, None]
            trades = target_dollars - positions
            positions = target_dollars
            cash = cash - trades.sum(axis=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            weights_hist[:, t, :] = positions / np.where(
                equity[:, None] != 0, equity[:, None], 1.0
            )

    # Match per-path Engine convention: returns[0] is NaN on the entry bar
    # (no prior bar in the post-warmup frame to diff against).
    path_rets = rets[:, warmup_bars:].copy()
    path_rets[:, 0] = np.nan

    return BatchResult(
        path_returns=path_rets,
        path_equity=eq[:, warmup_bars:],
        weights=weights_hist[:, warmup_bars:, :],
        dates=pd.DatetimeIndex(dates[warmup_bars:]),
        initial_capital=float(initial_capital),
        n_paths=N,
    )

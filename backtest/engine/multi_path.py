from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from backtest.costs.base import CostModel
from backtest.costs.liquidity import LiquidityCap
from backtest.data.panel import DataPanel
from backtest.engine.engine import BacktestResult, Engine
from backtest.splitters.splitters import Splitter
from backtest.strategy.base import Strategy

StrategyOrFactory = Union[Strategy, Callable[[], Strategy]]


@dataclass
class MultiPathResult:
    split_returns: List[pd.Series]
    path_returns: pd.DataFrame
    path_equity: pd.DataFrame
    splitter: Splitter
    initial_capital: float
    n_splits: int
    n_paths: int
    # One BacktestResult per (split, segment). CPCV with n_test_groups > 1
    # produces multiple segments per split; walk-forward has 1 segment per split.
    split_results: List[List[BacktestResult]] = field(default_factory=list)

    def metrics_per_path(self):
        from backtest.metrics import compute_metrics

        out = []
        for col in self.path_returns.columns:
            ret = self.path_returns[col].dropna()
            eq = self.path_equity[col].dropna()
            out.append(compute_metrics(ret, eq))
        return out

    def aggregate_metrics(self) -> dict:
        reports = self.metrics_per_path()
        keys = [
            "sharpe", "sortino", "calmar", "modified_sharpe", "psr",
            "max_drawdown", "time_under_water", "var_5", "cvar_5",
            "ann_return", "ann_vol", "total_return", "total_pnl",
            "total_costs", "longest_underwater",
            "hit_rate", "turnover", "information_ratio", "beta",
        ]
        out: dict = {}
        for k in keys:
            arr = np.asarray([getattr(r, k) for r in reports], dtype=float)
            arr = arr[np.isfinite(arr)]
            if len(arr) == 0:
                out[f"{k}_mean"] = float("nan")
                out[f"{k}_std"] = float("nan")
                out[f"{k}_min"] = float("nan")
                out[f"{k}_max"] = float("nan")
            else:
                out[f"{k}_mean"] = float(arr.mean())
                out[f"{k}_std"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
                out[f"{k}_min"] = float(arr.min())
                out[f"{k}_max"] = float(arr.max())
        return out

    def summary(
        self,
        ann_factor: int = 252,
        figsize: tuple = (13, 4.5),
    ) -> dict:
        """Print aggregate metrics, plot per-path equity overlay + per-path SR histogram.

        Returns the aggregate metrics dict.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "matplotlib is required for summary(). "
                "Install with: pip install backtest[viz]"
            ) from e

        agg = self.aggregate_metrics()
        reports = self.metrics_per_path()
        sharpes = np.asarray([r.sharpe for r in reports], dtype=float)
        sharpes = sharpes[np.isfinite(sharpes)]

        try:
            import pandas as pd
            from IPython.display import display
            stats = ["mean", "std", "min", "max"]
            metrics = ["sharpe", "sortino", "calmar", "max_drawdown", "psr"]
            tbl = pd.DataFrame(
                {s: [agg[f"{m}_{s}"] for m in metrics] for s in stats},
                index=metrics,
            ).round(4)
            display(tbl)
        except ImportError:
            for k, v in agg.items():
                print(f"  {k:<28s} {v:.4f}")

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        for col in self.path_equity.columns:
            axes[0].plot(self.path_equity.index, self.path_equity[col],
                         color="C0", alpha=0.35, lw=0.9)
        axes[0].axhline(self.initial_capital, color="gray", ls="--", lw=0.8)
        axes[0].set_title(f"Per-path equity ({self.n_paths} paths)")
        axes[0].set_ylabel("equity")

        if len(sharpes) > 0:
            axes[1].hist(sharpes, bins=max(5, len(sharpes) // 2),
                         color="C0", alpha=0.7, edgecolor="white")
            axes[1].axvline(sharpes.mean(), color="C3", lw=2.0,
                            label=f"mean = {sharpes.mean():.2f}")
            axes[1].axvline(0, color="gray", ls="--", lw=0.8)
        axes[1].set_xlabel("annualized Sharpe")
        axes[1].set_ylabel("count")
        axes[1].set_title("Per-path Sharpe distribution")
        axes[1].legend(loc="best", frameon=False, fontsize=8)

        fig.tight_layout()
        return agg


class MultiPathEngine:
    """Runs a strategy across all splits from a Splitter, in parallel.

    strategy may be a Strategy instance (deepcopied per split) or a zero-arg
    callable (called per split). Each split fits a fresh strategy on its train
    window, then evaluates on the test window. For CPCV with non-contiguous
    test sets, the engine runs once per contiguous test segment and concatenates
    returns to avoid cross-segment drift artifacts.
    """

    def __init__(
        self,
        data: DataPanel,
        strategy: StrategyOrFactory,
        splitter: Splitter,
        costs: Optional[CostModel] = None,
        liquidity: Optional[LiquidityCap] = None,
        initial_capital: float = 1_000_000.0,
        cash_rate: float = 0.0,
        borrow_rate: float = 0.0,
    ):
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        self.data = data
        self.strategy = strategy
        self.splitter = splitter
        self.costs = costs
        self.liquidity = liquidity
        self.initial_capital = float(initial_capital)
        self.cash_rate = float(cash_rate)
        self.borrow_rate = float(borrow_rate)
        self._is_factory = callable(strategy) and not isinstance(strategy, Strategy)

    def run(
        self,
        dates: Optional[pd.DatetimeIndex] = None,
        n_jobs: int = -1,
    ) -> MultiPathResult:
        dates = self.data.dates if dates is None else pd.DatetimeIndex(dates)
        all_splits = list(self.splitter.split(dates))

        args = [
            (
                self.data,
                self.strategy,
                self.costs,
                self.liquidity,
                self.initial_capital,
                self.cash_rate,
                self.borrow_rate,
                train,
                test,
                self._is_factory,
            )
            for train, test in all_splits
        ]

        if n_jobs == 1:
            outputs = [_run_one_split(*a) for a in args]
        else:
            try:
                outputs = Parallel(n_jobs=n_jobs)(
                    delayed(_run_one_split)(*a) for a in args
                )
            except Exception as e:
                # Joblib wraps worker tracebacks; re-run sequentially so the
                # user sees the underlying error directly.
                warnings.warn(
                    f"MultiPathEngine parallel run raised {type(e).__name__}; "
                    "retrying with n_jobs=1 to surface the underlying traceback.",
                    stacklevel=2,
                )
                outputs = [_run_one_split(*a) for a in args]

        split_returns = [out[0] for out in outputs]
        split_results = [out[1] for out in outputs]

        path_returns = self.splitter.assemble_paths(dates, split_returns)
        path_equity = (1.0 + path_returns.fillna(0.0)).cumprod() * self.initial_capital

        return MultiPathResult(
            split_returns=split_returns,
            split_results=split_results,
            path_returns=path_returns,
            path_equity=path_equity,
            splitter=self.splitter,
            initial_capital=self.initial_capital,
            n_splits=self.splitter.n_splits(),
            n_paths=self.splitter.n_paths(),
        )


def _run_one_split(
    data: DataPanel,
    strategy_or_factory: StrategyOrFactory,
    costs: Optional[CostModel],
    liquidity: Optional[LiquidityCap],
    initial_capital: float,
    cash_rate: float,
    borrow_rate: float,
    train: pd.DatetimeIndex,
    test: pd.DatetimeIndex,
    is_factory: bool,
) -> Tuple[pd.Series, List[BacktestResult]]:
    if is_factory:
        strategy = strategy_or_factory()
    else:
        strategy = copy.deepcopy(strategy_or_factory)

    if len(test) == 0:
        return pd.Series(dtype=float, name="returns"), []

    if len(train) > 0:
        strategy.fit(data.as_of(train[-1]))

    segments = _contiguous_segments(test, data.dates)
    engine = Engine(
        data=data,
        strategy=strategy,
        costs=costs,
        liquidity=liquidity,
        initial_capital=initial_capital,
        cash_rate=cash_rate,
        borrow_rate=borrow_rate,
    )

    pieces = []
    results = []
    for seg in segments:
        result = engine.run(test_dates=seg)
        pieces.append(result.returns)
        results.append(result)
    return pd.concat(pieces).sort_index(), results


def _contiguous_segments(
    test: pd.DatetimeIndex,
    all_dates: pd.DatetimeIndex,
) -> List[pd.DatetimeIndex]:
    if len(test) == 0:
        return []
    pos = all_dates.get_indexer(test)
    if (pos < 0).any():
        raise ValueError("test contains dates not in data panel")
    breaks = np.where(np.diff(pos) > 1)[0]
    segments = []
    start = 0
    for b in breaks:
        segments.append(test[start : b + 1])
        start = b + 1
    segments.append(test[start:])
    return segments

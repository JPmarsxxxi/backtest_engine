from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from backtest.costs.base import CostModel
from backtest.costs.liquidity import LiquidityCap
from backtest.data.panel import DataPanel
from backtest.strategy.base import Strategy

_TRADING_DAYS_PER_YEAR = 252


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    returns: pd.Series
    weights: pd.DataFrame
    positions: pd.DataFrame
    trades: pd.DataFrame
    costs: pd.Series
    unfilled: pd.DataFrame
    metadata: dict

    @property
    def pnl(self) -> pd.Series:
        """Net dollar PnL per bar (after costs). Sums to equity_curve[-1] - initial_capital."""
        return self.equity_curve.diff().fillna(0.0).rename("pnl")

    @property
    def gross_pnl(self) -> pd.Series:
        """Pre-cost dollar PnL per bar (net pnl + costs)."""
        return (self.pnl + self.costs).rename("gross_pnl")

    def summary(
        self,
        ann_factor: int = 252,
        ci_level: float = 0.95,
        rolling_window: int = 60,
        figsize: tuple = (13, 8),
    ):
        """Print metrics, plot a 2x2 dashboard, return the MetricsReport.

        Plots: equity curve, drawdown, rolling Sharpe, asymptotic SR distribution.
        Requires matplotlib (install with: pip install backtest[viz]).
        """
        from backtest.metrics import compute_metrics
        from backtest.metrics.plot import plot_sharpe_distribution
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "matplotlib is required for summary(). "
                "Install with: pip install backtest[viz]"
            ) from e

        rep = compute_metrics(
            self.returns, self.equity_curve,
            costs=self.costs, trades=self.trades,
            ann_factor=ann_factor, ci_level=ci_level,
        )
        try:
            from IPython.display import display
            display(rep)
        except ImportError:
            print(rep)

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        eq = self.equity_curve
        peak = np.maximum.accumulate(eq.values)
        dd = np.where(peak > 0, (eq.values - peak) / peak, 0.0)

        axes[0, 0].plot(eq.index, eq.values, color="C0", lw=1.2)
        axes[0, 0].axhline(
            self.metadata.get("initial_capital", eq.iloc[0]),
            color="gray", ls="--", lw=0.8,
        )
        axes[0, 0].set_title("Equity curve")
        axes[0, 0].set_ylabel("equity")

        axes[0, 1].fill_between(eq.index, dd, 0, color="C3", alpha=0.4)
        axes[0, 1].set_title(f"Drawdown (max {rep.max_drawdown:.1%})")
        axes[0, 1].set_ylabel("drawdown")

        rs = (
            self.returns.rolling(rolling_window)
            .apply(
                lambda x: x.mean() / x.std(ddof=1) * np.sqrt(ann_factor)
                if x.std(ddof=1) > 0 else np.nan,
                raw=False,
            )
        )
        axes[1, 0].plot(rs.index, rs.values, color="C2", lw=1.0)
        axes[1, 0].axhline(0, color="gray", ls="--", lw=0.8)
        axes[1, 0].axhline(rep.sharpe, color="C0", ls=":", lw=0.8,
                           label=f"full-sample = {rep.sharpe:.2f}")
        axes[1, 0].set_title(f"Rolling Sharpe ({rolling_window} bars, ann.)")
        axes[1, 0].set_ylabel("Sharpe")
        axes[1, 0].legend(loc="best", frameon=False, fontsize=8)

        plot_sharpe_distribution(
            self.returns, ann_factor=ann_factor,
            ci_level=ci_level, ax=axes[1, 1],
        )

        fig.tight_layout()
        return rep


class Engine:
    """Single-path walk-forward backtester.

    Inputs are immutable references to a DataPanel and a Strategy. Costs and
    liquidity are optional. run() iterates dates, marks-to-market on every bar,
    rebalances on strategy-declared dates, and returns a BacktestResult.
    """

    def __init__(
        self,
        data: DataPanel,
        strategy: Strategy,
        costs: Optional[CostModel] = None,
        liquidity: Optional[LiquidityCap] = None,
        initial_capital: float = 1_000_000.0,
        cash_rate: float = 0.0,
        borrow_rate: float = 0.0,
    ):
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        missing = [name for name in strategy.required_data() if not data.has_field(name)]
        if missing:
            raise ValueError(
                f"Strategy {type(strategy).__name__} declared required_data "
                f"fields {missing} that are not registered on the panel. "
                f"Available: {data.available_fields()}."
            )
        if liquidity is not None and not data.has_field("volume"):
            raise ValueError(
                "LiquidityCap was supplied but the panel has no volume. "
                "Without volume the cap silently does nothing. "
                "Either pass volume to DataPanel or drop the LiquidityCap."
            )
        self.data = data
        self.strategy = strategy
        self.costs = costs
        self.liquidity = liquidity
        self.initial_capital = float(initial_capital)
        self.cash_rate = float(cash_rate)
        self.borrow_rate = float(borrow_rate)

    def run(
        self,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
        train_dates: Optional[pd.DatetimeIndex] = None,
        test_dates: Optional[pd.DatetimeIndex] = None,
    ) -> BacktestResult:
        dates = self._resolve_dates(start, end, test_dates)
        if len(dates) == 0:
            raise ValueError("no dates in range")

        if train_dates is not None and len(train_dates) > 0:
            train_end = pd.Timestamp(train_dates[-1])
            self.strategy.fit(self.data.as_of(train_end))

        rebalance_dates = self.strategy.rebalance_dates(dates)
        rebalance_mask = dates.isin(rebalance_dates)

        assets = self.data.assets_all
        n = len(dates)
        m = len(assets)

        prices_arr = (
            self.data._prices.reindex(index=dates, columns=assets).to_numpy()
        )
        ratios = self._compute_ratios(prices_arr)

        eq_curve = np.zeros(n)
        ret_arr = np.full(n, np.nan)
        weights_arr = np.zeros((n, m))
        positions_arr = np.zeros((n, m))
        trades_arr = np.zeros((n, m))
        unfilled_arr = np.zeros((n, m))
        costs_arr = np.zeros(n)

        positions = np.zeros(m)
        cash = self.initial_capital
        peak = self.initial_capital

        for i in range(n):
            t = dates[i]

            if i > 0:
                positions = positions * ratios[i - 1]

                nan_price = ~np.isfinite(prices_arr[i])
                if nan_price.any() and (positions[nan_price] != 0).any():
                    cash += positions[nan_price].sum()
                    positions[nan_price] = 0.0

            equity_pre = positions.sum() + cash
            if equity_pre > peak:
                peak = equity_pre
            dd = (peak - equity_pre) / peak if peak > 0 else 0.0

            need_view = (self.costs is not None) or rebalance_mask[i]
            view = self.data.as_of(t) if need_view else None

            hc = 0.0
            if self.costs is not None:
                pos_series = pd.Series(positions, index=assets)
                hc = self.costs.holding_cost(pos_series, view)
                cash -= hc

            if cash > 0 and self.cash_rate != 0.0:
                cash *= 1.0 + self.cash_rate / _TRADING_DAYS_PER_YEAR
            elif cash < 0 and self.borrow_rate != 0.0:
                cash *= 1.0 + self.borrow_rate / _TRADING_DAYS_PER_YEAR

            tc = 0.0
            executed = np.zeros(m)
            unfilled = np.zeros(m)

            if rebalance_mask[i]:
                equity = positions.sum() + cash
                pos_series = pd.Series(positions, index=assets)
                state = {
                    "drawdown": dd,
                    "equity": equity,
                    "positions": pos_series,
                    "cash": cash,
                    "peak": peak,
                }

                proposed = self.strategy.generate_weights(view, t)
                proposed = proposed.reindex(assets).fillna(0.0).astype(float)

                eligible = view.assets
                ineligible = ~proposed.index.isin(eligible)
                if ineligible.any():
                    proposed.loc[ineligible] = 0.0

                final_w = self.strategy.apply_risk(proposed, state, view)
                final_w = final_w.reindex(assets).fillna(0.0).astype(float)
                target_dollars = final_w.to_numpy() * equity

                proposed_trades = target_dollars - positions
                price_invalid = ~np.isfinite(prices_arr[i])
                if price_invalid.any():
                    proposed_trades = np.where(price_invalid, 0.0, proposed_trades)

                if self.liquidity is not None:
                    pt_series = pd.Series(proposed_trades, index=assets)
                    exec_s, unfill_s = self.liquidity.apply(pt_series, view)
                    executed = exec_s.reindex(assets).fillna(0.0).to_numpy()
                    unfilled = unfill_s.reindex(assets).fillna(0.0).to_numpy()
                else:
                    executed = proposed_trades

                if self.costs is not None:
                    exec_series = pd.Series(executed, index=assets)
                    tc = self.costs.trade_cost(exec_series, view)

                positions = positions + executed
                cash = cash - executed.sum() - tc

            equity_now = positions.sum() + cash
            eq_curve[i] = equity_now
            positions_arr[i] = positions
            trades_arr[i] = executed
            unfilled_arr[i] = unfilled
            costs_arr[i] = hc + tc
            if equity_now != 0:
                weights_arr[i] = positions / equity_now
            if i > 0 and eq_curve[i - 1] != 0:
                ret_arr[i] = (eq_curve[i] - eq_curve[i - 1]) / eq_curve[i - 1]
            # ret_arr[0] remains NaN — no prior bar to diff against.

        meta = {
            "start": dates[0],
            "end": dates[-1],
            "initial_capital": self.initial_capital,
            "n_bars": n,
            "n_rebalances": int(rebalance_mask.sum()),
            "fit_called": train_dates is not None and len(train_dates) > 0,
        }

        return BacktestResult(
            equity_curve=pd.Series(eq_curve, index=dates, name="equity"),
            returns=pd.Series(ret_arr, index=dates, name="returns"),
            weights=pd.DataFrame(weights_arr, index=dates, columns=assets),
            positions=pd.DataFrame(positions_arr, index=dates, columns=assets),
            trades=pd.DataFrame(trades_arr, index=dates, columns=assets),
            costs=pd.Series(costs_arr, index=dates, name="costs"),
            unfilled=pd.DataFrame(unfilled_arr, index=dates, columns=assets),
            metadata=meta,
        )

    def _resolve_dates(self, start, end, test_dates) -> pd.DatetimeIndex:
        if test_dates is not None:
            return pd.DatetimeIndex(test_dates).sort_values()
        dates = self.data.dates
        if start is not None:
            dates = dates[dates >= pd.Timestamp(start)]
        if end is not None:
            dates = dates[dates <= pd.Timestamp(end)]
        return dates

    @staticmethod
    def _compute_ratios(prices_arr: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = prices_arr[1:] / prices_arr[:-1]
        return np.where(np.isfinite(ratios), ratios, 1.0)

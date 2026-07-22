from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from scipy.stats import norm

from backtest.metrics.core import (
    DEFAULT_ANN_FACTOR,
    beta as beta_fn,
    calmar_ratio,
    conditional_value_at_risk,
    hit_rate as hit_rate_fn,
    information_ratio as information_ratio_fn,
    min_trl,
    modified_sharpe,
    psr,
    sharpe_distribution,
    sharpe_ratio,
    sortino_ratio,
    margin as margin_fn,
    turnover as turnover_fn,
    value_at_risk,
)
from backtest.metrics.path import max_drawdown, time_under_water
from backtest.metrics.period import (
    median_arith_annual_return,
    median_calendar_year_return,
    yearly_metrics,
)


@dataclass
class MetricsReport:
    total_return: float
    ann_return: float
    ann_vol: float
    hit_rate: float
    total_pnl: float
    total_costs: float
    gross_pnl: float
    turnover: float
    margin: float
    sharpe: float
    sharpe_std: float
    sharpe_ci_low: float
    sharpe_ci_high: float
    ci_level: float
    sortino: float
    calmar: float
    information_ratio: float
    beta: float
    modified_sharpe: float
    psr: float
    min_trl: float
    var_5: float
    cvar_5: float
    max_drawdown: float
    time_under_water: float
    longest_underwater: int
    n_obs: int
    ann_factor: int
    median_cal_year_return: float = float("nan")
    median_arith_ann_return: float = float("nan")
    returns: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    yearly: Optional[pd.DataFrame] = field(default=None, repr=False, compare=False)

    def psr_at(self, sr_star: float) -> float:
        return psr(self.returns, sr_star=sr_star, ann_factor=self.ann_factor)

    def min_trl_at(self, sr_star: float, alpha: float = 0.05) -> float:
        return min_trl(
            self.returns, sr_star=sr_star, alpha=alpha, ann_factor=self.ann_factor
        )

    def plot_sharpe_distribution(self, ax=None, ci_level: Optional[float] = None):
        from backtest.metrics.plot import plot_sharpe_distribution
        return plot_sharpe_distribution(
            self.returns,
            ann_factor=self.ann_factor,
            ci_level=self.ci_level if ci_level is None else ci_level,
            ax=ax,
        )

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k not in ("returns", "yearly")}

    def _rows(self) -> list[tuple[str, str]]:
        ci_pct = int(round(self.ci_level * 100))
        rows = [
            ("Total return",         f"{self.total_return:+.2%}"),
            ("Annualized return",    f"{self.ann_return:+.2%}"),
            ("Annualized vol",       f"{self.ann_vol:.2%}"),
        ]
        if np.isfinite(self.hit_rate):
            rows.append(("Hit rate",          f"{self.hit_rate:.2%}"))
        rows.append(("Total PnL ($)",        f"{self.total_pnl:+,.2f}"))
        if np.isfinite(self.total_costs):
            rows.append(("Total costs ($)",   f"{self.total_costs:,.2f}"))
            rows.append(("Gross PnL ($)",     f"{self.gross_pnl:+,.2f}"))
        if np.isfinite(self.turnover):
            rows.append(("Turnover",          f"{self.turnover:.2f}x"))
        if np.isfinite(self.margin):
            rows.append(("Margin",            f"{self.margin:.4%}"))
        rows += [
            ("Sharpe (ann.)",        f"{self.sharpe:.3f}"),
            ("Sharpe SD (ann.)",     f"{self.sharpe_std:.3f}"),
            (f"Sharpe {ci_pct}% CI",     f"[{self.sharpe_ci_low:.3f}, {self.sharpe_ci_high:.3f}]"),
            ("Sortino",              f"{self.sortino:.3f}"),
            ("Calmar",               f"{self.calmar:.3f}"),
        ]
        if np.isfinite(self.information_ratio):
            rows.append(("Information ratio", f"{self.information_ratio:.3f}"))
        if np.isfinite(self.beta):
            rows.append(("Beta",              f"{self.beta:.3f}"))
        rows += [
            ("Modified Sharpe",      f"{self.modified_sharpe:.3f}"),
            ("PSR (vs 0)",           f"{self.psr:.3f}"),
            ("MinTRL (vs 0)",        f"{self.min_trl:.0f} bars"),
            ("VaR 5%",               f"{self.var_5:.4f}"),
            ("cVaR 5%",              f"{self.cvar_5:.4f}"),
            ("Max drawdown",         f"{self.max_drawdown:.2%}"),
            ("Time underwater",      f"{self.time_under_water:.2%}"),
            ("Longest underwater",   f"{self.longest_underwater} bars"),
            ("N obs",                f"{self.n_obs}"),
        ]
        if np.isfinite(self.median_cal_year_return):
            rows.append(
                ("Median cal-year ret", f"{self.median_cal_year_return:+.2%}")
            )
        if np.isfinite(self.median_arith_ann_return):
            rows.append(
                ("Median arith ann",    f"{self.median_arith_ann_return:+.2%}")
            )
        return rows

    def yearly_table(self) -> pd.DataFrame:
        """Per-calendar-year metrics (empty if index was not datetime/period)."""
        if self.yearly is None:
            return pd.DataFrame()
        return self.yearly.copy()

    def __repr__(self) -> str:
        rows = self._rows()
        width = max(len(k) for k, _ in rows)
        lines = ["MetricsReport:"]
        for k, v in rows:
            lines.append(f"  {k:<{width}}  {v}")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        rows = self._rows()
        body = "".join(
            f"<tr><td style='padding:2px 12px 2px 0;color:#555'>{k}</td>"
            f"<td style='padding:2px 0;font-family:monospace'>{v}</td></tr>"
            for k, v in rows
        )
        return (
            "<table style='border-collapse:collapse;font-size:90%'>"
            "<caption style='text-align:left;font-weight:bold;padding-bottom:4px'>"
            "MetricsReport</caption>"
            f"{body}</table>"
        )


def compute_metrics(
    returns,
    equity,
    costs=None,
    trades=None,
    benchmark=None,
    ann_factor: int = DEFAULT_ANN_FACTOR,
    rf: float = 0.0,
    var_alpha: float = 0.05,
    ci_level: float = 0.95,
) -> MetricsReport:
    if not 0 < ci_level < 1:
        raise ValueError("ci_level must be in (0, 1)")

    returns_series = returns if isinstance(returns, pd.Series) else None
    if returns_series is not None:
        r_arr = returns_series.dropna().to_numpy(dtype=float)
    else:
        r_arr = np.asarray(returns, dtype=float)
        r_arr = r_arr[np.isfinite(r_arr)]

    eq_arr = (
        equity.to_numpy() if isinstance(equity, pd.Series) else np.asarray(equity, float)
    )
    eq_arr = np.ascontiguousarray(eq_arr, dtype=np.float64)

    mdd = max_drawdown(eq_arr)
    tu, longest = time_under_water(eq_arr)

    sr_mean, sr_std = sharpe_distribution(r_arr, ann_factor=ann_factor)
    if np.isfinite(sr_std):
        z = float(norm.ppf(0.5 + ci_level / 2.0))
        ci_low, ci_high = sr_mean - z * sr_std, sr_mean + z * sr_std
    else:
        ci_low, ci_high = float("nan"), float("nan")

    if len(eq_arr) >= 2 and eq_arr[0] > 0 and np.isfinite(eq_arr[-1]):
        total_ret = float(eq_arr[-1] / eq_arr[0] - 1.0)
        total_pnl_val = float(eq_arr[-1] - eq_arr[0])
        # Prefer calendar time when the equity Series has a DatetimeIndex; fall
        # back to bar-count for array inputs or non-datetime indices.
        if (
            isinstance(equity, pd.Series)
            and isinstance(equity.index, pd.DatetimeIndex)
            and len(equity) >= 2
        ):
            years = (equity.index[-1] - equity.index[0]).days / 365.25
        else:
            years = len(r_arr) / ann_factor
        if years > 0 and (1.0 + total_ret) > 0:
            ann_ret = float((1.0 + total_ret) ** (1.0 / years) - 1.0)
        else:
            ann_ret = float("nan")
    else:
        total_ret = float("nan")
        total_pnl_val = float("nan")
        ann_ret = float("nan")
    ann_vol_val = (
        float(r_arr.std(ddof=1) * np.sqrt(ann_factor)) if len(r_arr) >= 2 else float("nan")
    )

    if costs is not None:
        c_arr = costs.to_numpy() if isinstance(costs, pd.Series) else np.asarray(costs, float)
        total_costs_val = float(np.nansum(c_arr))
        gross_pnl_val = total_pnl_val + total_costs_val if np.isfinite(total_pnl_val) else float("nan")
    else:
        total_costs_val = float("nan")
        gross_pnl_val = float("nan")

    hit_rate_val = hit_rate_fn(r_arr)
    turnover_val = (
        turnover_fn(trades, equity)
        if trades is not None
        else float("nan")
    )
    margin_val = (
        margin_fn(total_pnl_val, trades)
        if trades is not None and np.isfinite(total_pnl_val)
        else float("nan")
    )
    if benchmark is not None:
        ir_val = information_ratio_fn(returns, benchmark, ann_factor=ann_factor)
        beta_val = beta_fn(returns, benchmark)
    else:
        ir_val = float("nan")
        beta_val = float("nan")

    med_cal = float("nan")
    med_arith = float("nan")
    yearly_df = None
    if returns_series is not None:
        try:
            med_cal = median_calendar_year_return(returns_series)
            med_arith = median_arith_annual_return(returns_series, ann_factor=ann_factor)
            yearly_df = yearly_metrics(
                returns_series,
                equity if isinstance(equity, pd.Series) else None,
                ann_factor=ann_factor,
            )
        except TypeError:
            pass

    return MetricsReport(
        total_return=total_ret,
        ann_return=ann_ret,
        ann_vol=ann_vol_val,
        hit_rate=hit_rate_val,
        total_pnl=total_pnl_val,
        total_costs=total_costs_val,
        gross_pnl=gross_pnl_val,
        turnover=turnover_val,
        margin=margin_val,
        sharpe=sharpe_ratio(r_arr, ann_factor=ann_factor, rf=rf),
        sharpe_std=sr_std,
        sharpe_ci_low=ci_low,
        sharpe_ci_high=ci_high,
        ci_level=ci_level,
        sortino=sortino_ratio(r_arr, ann_factor=ann_factor, rf=rf),
        calmar=calmar_ratio(r_arr, eq_arr, ann_factor=ann_factor),
        information_ratio=ir_val,
        beta=beta_val,
        modified_sharpe=modified_sharpe(r_arr, alpha=var_alpha, ann_factor=ann_factor),
        psr=psr(r_arr, sr_star=0.0, ann_factor=ann_factor),
        min_trl=min_trl(r_arr, sr_star=0.0, ann_factor=ann_factor),
        var_5=value_at_risk(r_arr, alpha=var_alpha),
        cvar_5=conditional_value_at_risk(r_arr, alpha=var_alpha),
        max_drawdown=float(mdd),
        time_under_water=float(tu),
        longest_underwater=int(longest),
        n_obs=len(r_arr),
        ann_factor=ann_factor,
        median_cal_year_return=med_cal,
        median_arith_ann_return=med_arith,
        returns=r_arr,
        yearly=yearly_df,
    )

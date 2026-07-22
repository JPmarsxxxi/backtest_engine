"""Calendar-period breakdown helpers (year-by-year, median annual return)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.metrics.core import sharpe_ratio
from backtest.metrics.path import max_drawdown


def _returns_with_period_index(returns: pd.Series) -> pd.Series:
    """Normalize to monthly PeriodIndex for grouping by calendar year."""
    r = returns.dropna().astype(float)
    if r.empty:
        return r
    if isinstance(r.index, pd.PeriodIndex):
        if r.index.freq is None:
            r = r.copy()
            r.index = r.index.asfreq("M")
        return r
    if isinstance(r.index, pd.DatetimeIndex):
        out = r.copy()
        out.index = out.index.to_period("M")
        return out
    raise TypeError(
        "yearly metrics require returns indexed by DatetimeIndex or PeriodIndex; "
        f"got {type(r.index).__name__}"
    )


def calendar_year_returns(returns: pd.Series) -> pd.Series:
    """Compound return per calendar year (decimal), indexed by int year."""
    r = _returns_with_period_index(returns)
    if r.empty:
        return pd.Series(dtype=float)
    return r.groupby(r.index.year).apply(lambda s: float((1.0 + s).prod() - 1.0))


def median_calendar_year_return(returns: pd.Series) -> float:
    """Median of calendar-year compound returns (decimal)."""
    yr = calendar_year_returns(returns)
    return float(yr.median()) if len(yr) else float("nan")


def median_arith_annual_return(returns: pd.Series, ann_factor: int = 12) -> float:
    """Median of within-calendar-year arithmetic annualized returns (mean * ann_factor)."""
    r = _returns_with_period_index(returns)
    if r.empty:
        return float("nan")
    vals = [float(g.mean() * ann_factor) for _, g in r.groupby(r.index.year)]
    return float(np.median(vals)) if vals else float("nan")


def yearly_metrics(
    returns: pd.Series,
    equity: pd.Series | None = None,
    ann_factor: int = 12,
) -> pd.DataFrame:
    """
    Per-calendar-year table: compound return, arithmetic ann., Sharpe, max DD.

    Parameters
    ----------
    returns : pd.Series
        Bar returns with DatetimeIndex or PeriodIndex (monthly for PMM notebooks).
    equity : pd.Series, optional
        Equity curve aligned to ``returns`` (after dropna). Used for within-year MDD.
    ann_factor : int
        Bars per year for Sharpe / arithmetic annualization (12 for monthly).
    """
    r = _returns_with_period_index(returns)
    if r.empty:
        return pd.DataFrame(
            columns=["n", "total_return", "arith_ann", "sharpe", "max_drawdown"]
        )

    eq = None
    if equity is not None:
        eq = equity.reindex(returns.index).dropna()
        if isinstance(eq.index, pd.DatetimeIndex):
            eq = eq.copy()
            eq.index = eq.index.to_period("M")

    rows = []
    for year, g in r.groupby(r.index.year):
        arr = g.to_numpy(dtype=float)
        n = len(arr)
        tot = float((1.0 + arr).prod() - 1.0)
        arith = float(g.mean() * ann_factor)
        vol = float(g.std(ddof=1))
        sr = float((g.mean() * ann_factor) / (vol * np.sqrt(ann_factor))) if n > 1 and vol > 0 else float("nan")
        if eq is not None and year in eq.index.year:
            eq_y = eq.loc[eq.index.year == year].to_numpy(dtype=float)
            mdd = float(max_drawdown(eq_y)) if len(eq_y) >= 2 else float("nan")
        else:
            cum = (1.0 + arr).cumprod()
            mdd = float(max_drawdown(cum)) if len(cum) >= 2 else float("nan")
        rows.append(
            {
                "year": int(year),
                "n": n,
                "total_return": tot,
                "arith_ann": arith,
                "sharpe": sr,
                "max_drawdown": mdd,
            }
        )
    return pd.DataFrame(rows).set_index("year")

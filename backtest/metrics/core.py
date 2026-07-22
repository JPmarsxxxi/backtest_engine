from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

DEFAULT_ANN_FACTOR = 252


def _to_array(returns) -> np.ndarray:
    if isinstance(returns, pd.Series):
        arr = returns.dropna().to_numpy(dtype=float)
    else:
        arr = np.asarray(returns, dtype=float)
        arr = arr[np.isfinite(arr)]
    return arr


def sharpe_ratio(
    returns,
    ann_factor: int = DEFAULT_ANN_FACTOR,
    rf: float = 0.0,
) -> float:
    r = _to_array(returns)
    if len(r) < 2:
        return float("nan")
    excess = r - rf / ann_factor
    sd = excess.std(ddof=1)
    if not np.isfinite(sd) or sd < 1e-12:
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(ann_factor))


def sortino_ratio(
    returns,
    ann_factor: int = DEFAULT_ANN_FACTOR,
    rf: float = 0.0,
) -> float:
    r = _to_array(returns)
    if len(r) < 2:
        return float("nan")
    excess = r - rf / ann_factor
    downside = excess[excess < 0]
    if len(downside) < 2:
        return float("nan")
    dd_std = float(np.sqrt(np.mean(downside ** 2)))
    if dd_std == 0:
        return float("nan")
    return float(excess.mean() / dd_std * np.sqrt(ann_factor))


def calmar_ratio(
    returns,
    equity,
    ann_factor: int = DEFAULT_ANN_FACTOR,
) -> float:
    from backtest.metrics.path import max_drawdown

    r = _to_array(returns)
    if len(r) < 2:
        return float("nan")
    eq = np.ascontiguousarray(
        equity.to_numpy() if isinstance(equity, pd.Series) else np.asarray(equity, float),
        dtype=np.float64,
    )
    mdd = max_drawdown(eq)
    if mdd == 0 or not np.isfinite(mdd):
        return float("nan")
    return float(r.mean() * ann_factor / mdd)


def value_at_risk(returns, alpha: float = 0.05) -> float:
    r = _to_array(returns)
    if len(r) == 0:
        return float("nan")
    return float(np.quantile(r, alpha))


def conditional_value_at_risk(returns, alpha: float = 0.05) -> float:
    r = _to_array(returns)
    if len(r) == 0:
        return float("nan")
    var = np.quantile(r, alpha)
    tail = r[r <= var]
    if len(tail) == 0:
        return float("nan")
    return float(tail.mean())


def modified_sharpe(
    returns,
    alpha: float = 0.05,
    ann_factor: int = DEFAULT_ANN_FACTOR,
) -> float:
    r = _to_array(returns)
    if len(r) < 2:
        return float("nan")
    var = value_at_risk(r, alpha)
    if not np.isfinite(var) or var >= 0:
        return float("nan")
    return float(r.mean() / abs(var) * ann_factor)


def _per_bar_sr(r: np.ndarray) -> float:
    if len(r) < 2:
        return float("nan")
    sd = r.std(ddof=1)
    if not np.isfinite(sd) or sd < 1e-12:
        return float("nan")
    return float(r.mean() / sd)


def sharpe_var_term(r: np.ndarray, sr_pb: float) -> float:
    """Per-bar asymptotic variance factor of the SR estimator (paper Eq. 2).

    Equivalent to 1 + 0.5*SR^2 - g3*SR + (g4_excess)/4 * SR^2 with g4 in raw form.
    Var(SR_hat_per_bar) = sharpe_var_term / (T - 1).
    """
    g3 = float(stats.skew(r, bias=False))
    g4 = float(stats.kurtosis(r, fisher=False, bias=False))
    return 1.0 - g3 * sr_pb + (g4 - 1.0) / 4.0 * sr_pb ** 2


def sharpe_distribution(
    returns,
    ann_factor: int = DEFAULT_ANN_FACTOR,
) -> tuple[float, float]:
    """Asymptotic distribution of the annualized Sharpe ratio estimator.

    Returns (mean, std) where mean is the sample annualized SR and std is the
    annualized standard error from paper Eq. 2. Both NaN if insufficient data.
    """
    r = _to_array(returns)
    sr_pb = _per_bar_sr(r)
    if not np.isfinite(sr_pb):
        return float("nan"), float("nan")
    v = sharpe_var_term(r, sr_pb)
    T = len(r)
    if v <= 0 or T < 2:
        return float(sr_pb * np.sqrt(ann_factor)), float("nan")
    std_pb = np.sqrt(v / (T - 1))
    return float(sr_pb * np.sqrt(ann_factor)), float(std_pb * np.sqrt(ann_factor))


def psr(
    returns,
    sr_star: float = 0.0,
    ann_factor: int = DEFAULT_ANN_FACTOR,
) -> float:
    """Probabilistic Sharpe Ratio. sr_star is annualized."""
    r = _to_array(returns)
    sr_pb = _per_bar_sr(r)
    if not np.isfinite(sr_pb):
        return float("nan")
    sr_star_pb = sr_star / np.sqrt(ann_factor)
    var_term = sharpe_var_term(r, sr_pb)
    if var_term <= 0:
        return float("nan")
    T = len(r)
    z = (sr_pb - sr_star_pb) * np.sqrt(T - 1) / np.sqrt(var_term)
    return float(norm.cdf(z))


def min_trl(
    returns,
    sr_star: float = 0.0,
    alpha: float = 0.05,
    ann_factor: int = DEFAULT_ANN_FACTOR,
) -> float:
    """Minimum Track Record Length (in bars). sr_star is annualized."""
    r = _to_array(returns)
    sr_pb = _per_bar_sr(r)
    if not np.isfinite(sr_pb):
        return float("nan")
    sr_star_pb = sr_star / np.sqrt(ann_factor)
    if sr_pb <= sr_star_pb:
        return float("nan")
    var_term = sharpe_var_term(r, sr_pb)
    if var_term <= 0:
        return float("nan")
    z_alpha = float(norm.ppf(1.0 - alpha))
    return float(1.0 + var_term * (z_alpha / (sr_pb - sr_star_pb)) ** 2)


def hit_rate(returns) -> float:
    """Fraction of bars with strictly positive return. NaN if no observations."""
    r = _to_array(returns)
    if len(r) == 0:
        return float("nan")
    return float((r > 0).sum() / len(r))


def turnover(trades, equity) -> float:
    """Total two-way turnover: sum(|trades|) / mean(equity) over the period.

    A turnover of 2.0 means the portfolio traded twice its average book value
    in total over the backtest. Two-way (a 100% buy + 100% sell = 2.0).
    """
    if isinstance(trades, pd.DataFrame):
        traded = float(np.nansum(np.abs(trades.to_numpy(dtype=float))))
        n_bars = len(trades)
    else:
        t_arr = np.asarray(trades, dtype=float)
        traded = float(np.nansum(np.abs(t_arr)))
        n_bars = t_arr.shape[0] if t_arr.ndim > 0 else 0
    if isinstance(equity, pd.Series):
        eq_arr = equity.to_numpy(dtype=float)
    else:
        eq_arr = np.asarray(equity, dtype=float)
    if n_bars == 0:
        return float("nan")
    eq_mean = float(np.nanmean(eq_arr))
    if not np.isfinite(eq_mean) or eq_mean <= 0:
        return float("nan")
    return traded / eq_mean


def margin(total_pnl: float, trades) -> float:
    """Net PnL per dollar traded: total_pnl / sum(|trades|).

    A margin of 0.01 means the strategy earned 1 cent of net PnL per dollar
    of two-way volume. NaN if trades is None, total traded is zero, or
    total_pnl is not finite.
    """
    if isinstance(trades, pd.DataFrame):
        traded = float(np.nansum(np.abs(trades.to_numpy(dtype=float))))
    else:
        t_arr = np.asarray(trades, dtype=float)
        traded = float(np.nansum(np.abs(t_arr)))
    if not np.isfinite(total_pnl) or traded <= 0:
        return float("nan")
    return total_pnl / traded


def _align_pair(a, b) -> tuple[np.ndarray, np.ndarray]:
    """Align two return series for paired metrics (IR, beta).

    Series with overlapping indices → intersection, drop joint NaNs.
    Arrays → truncate to the shorter, drop joint non-finite.
    """
    if isinstance(a, pd.Series) and isinstance(b, pd.Series):
        common = a.index.intersection(b.index)
        a = a.reindex(common)
        b = b.reindex(common)
        mask = a.notna() & b.notna()
        return a[mask].to_numpy(dtype=float), b[mask].to_numpy(dtype=float)
    a_arr = np.asarray(a, dtype=float).ravel()
    b_arr = np.asarray(b, dtype=float).ravel()
    n = min(len(a_arr), len(b_arr))
    a_arr, b_arr = a_arr[:n], b_arr[:n]
    mask = np.isfinite(a_arr) & np.isfinite(b_arr)
    return a_arr[mask], b_arr[mask]


def information_ratio(
    returns,
    benchmark,
    ann_factor: int = DEFAULT_ANN_FACTOR,
) -> float:
    """Annualized Sharpe of (returns - benchmark) over the aligned overlap."""
    r, b = _align_pair(returns, benchmark)
    if len(r) < 2:
        return float("nan")
    excess = r - b
    sd = excess.std(ddof=1)
    if not np.isfinite(sd) or sd < 1e-12:
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(ann_factor))


def beta(returns, benchmark) -> float:
    """OLS slope of returns regressed on benchmark over the aligned overlap."""
    r, b = _align_pair(returns, benchmark)
    if len(r) < 2:
        return float("nan")
    var_b = float(np.var(b, ddof=1))
    if not np.isfinite(var_b) or var_b < 1e-24:
        return float("nan")
    cov_rb = float(np.cov(r, b, ddof=1)[0, 1])
    return cov_rb / var_b

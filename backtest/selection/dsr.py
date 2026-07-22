from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from backtest.metrics.core import sharpe_var_term

EULER_MASCHERONI = 0.5772156649015329


def expected_max_sr(K: int, var_sr: float) -> float:
    """Expected max Sharpe across K independent trials with per-trial SR variance var_sr.

    From the False Strategy Theorem (Lopez de Prado):
        SR_0 = sqrt(var_sr) * ((1 - gamma) * Z^-1(1 - 1/K) + gamma * Z^-1(1 - 1/(K*e)))
    """
    if K < 1:
        raise ValueError("K must be >= 1")
    if var_sr < 0:
        raise ValueError("var_sr must be >= 0")
    if K == 1 or var_sr == 0:
        return 0.0
    z1 = norm.ppf(1.0 - 1.0 / K)
    z2 = norm.ppf(1.0 - 1.0 / (K * np.e))
    sigma = np.sqrt(var_sr)
    return float(sigma * ((1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2))


def dsr(
    returns,
    K: int,
    sr_variance: Optional[float] = None,
) -> float:
    """Deflated Sharpe Ratio.

    Returns probability that the true Sharpe exceeds the expected luck-max
    given K trials. K=1 reduces to PSR vs zero.

    sr_variance: optional per-bar variance of trial SRs (e.g., sample variance
        across observed trials). If None, estimated from the asymptotic variance
        of the single-trial SR estimator.
    """
    if K < 1:
        raise ValueError("K must be >= 1")
    r = _to_array(returns)
    if len(r) < 2:
        return float("nan")
    sd = r.std(ddof=1)
    if not np.isfinite(sd) or sd < 1e-12:
        return float("nan")
    sr_pb = float(r.mean() / sd)
    T = len(r)
    var_term = sharpe_var_term(r, sr_pb)
    if var_term <= 0:
        return float("nan")

    if sr_variance is None:
        sr_variance = var_term / T

    sr_0 = expected_max_sr(K, sr_variance)
    z = (sr_pb - sr_0) * np.sqrt(T - 1) / np.sqrt(var_term)
    return float(norm.cdf(z))


def _to_array(returns) -> np.ndarray:
    if isinstance(returns, pd.Series):
        return returns.dropna().to_numpy(dtype=float)
    arr = np.asarray(returns, dtype=float)
    return arr[np.isfinite(arr)]

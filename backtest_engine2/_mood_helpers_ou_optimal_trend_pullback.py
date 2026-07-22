"""Mood changepoint helpers for `ou_optimal_trend_pullback.ipynb`.

This module is kept separate from the existing `_mood_helpers.py` so the
notebook can reference a dedicated helper file without changing the original
VIX change-point implementation.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def _mood_d_max_tau(x, n):
    """Return the normalized Mood-style max deviation statistic and argmax."""
    if n < 2:
        return 0.0, 0.0

    mu = 0.0
    for i in range(n):
        mu += x[i]
    mu /= n

    var = 0.0
    for i in range(n):
        diff = x[i] - mu
        var += diff * diff
    var /= n

    if var < 1e-12:
        return 0.0, 0.0

    sigma = np.sqrt(var)
    s = np.empty(n + 1, dtype=np.float64)
    s[0] = 0.0

    for i in range(n):
        s[i + 1] = s[i] + (x[i] - mu) / sigma

    dmax = 0.0
    tau = 0.0
    sn = s[n]

    for k in range(1, n):
        interp = sn * k / n
        dist = abs(s[k] - interp)
        if dist > dmax:
            dmax = dist
            tau = float(k)

    return dmax / np.sqrt(n), tau


def calibrate_mood_threshold(arl=10000, n_sim=500, max_n=30000, seed=42):
    """Estimate the null threshold for the Mood-style statistic."""
    rng = np.random.default_rng(seed)
    stats = np.empty(n_sim, dtype=np.float64)

    for i in range(n_sim):
        x = rng.standard_normal(max_n).astype(np.float64)
        stat, _ = _mood_d_max_tau(x, max_n)
        stats[i] = stat

    q = 1.0 - 1.0 / arl
    return float(np.quantile(stats, q))


MOOD_THRESHOLD = calibrate_mood_threshold()

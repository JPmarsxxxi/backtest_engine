from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def max_drawdown(equity: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown as a positive fraction of peak."""
    n = equity.shape[0]
    if n == 0:
        return 0.0
    peak = equity[0]
    max_dd = 0.0
    for i in range(n):
        if equity[i] > peak:
            peak = equity[i]
        if peak > 0:
            dd = (peak - equity[i]) / peak
            if dd > max_dd:
                max_dd = dd
    return max_dd


@njit(cache=True)
def time_under_water(equity: np.ndarray) -> tuple:
    """Returns (fraction_of_bars_underwater, longest_underwater_run_in_bars)."""
    n = equity.shape[0]
    if n == 0:
        return 0.0, 0
    peak = equity[0]
    underwater_bars = 0
    longest = 0
    current = 0
    for i in range(n):
        if equity[i] >= peak:
            peak = equity[i]
            current = 0
        else:
            underwater_bars += 1
            current += 1
            if current > longest:
                longest = current
    return underwater_bars / n, longest

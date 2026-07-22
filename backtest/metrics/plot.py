from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.stats import norm

from backtest.metrics.core import DEFAULT_ANN_FACTOR, sharpe_distribution


def plot_sharpe_distribution(
    returns,
    ann_factor: int = DEFAULT_ANN_FACTOR,
    ci_level: float = 0.95,
    sr_null: float = 0.0,
    ax=None,
):
    """Plot the asymptotic Normal distribution of the SR estimator (paper Eq. 2).

    Curve: PDF over annualized SR.
    Vertical line: estimated SR (point estimate).
    Shaded band: confidence interval at ci_level.
    Dashed line: sr_null (default 0) for reference.

    Returns the matplotlib Axes.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install backtest[viz]"
        ) from e

    if not 0 < ci_level < 1:
        raise ValueError("ci_level must be in (0, 1)")

    sr_mean, sr_std = sharpe_distribution(returns, ann_factor=ann_factor)
    if not np.isfinite(sr_mean) or not np.isfinite(sr_std) or sr_std <= 0:
        raise ValueError("insufficient data for Sharpe distribution")

    z = float(norm.ppf(0.5 + ci_level / 2.0))
    ci_low, ci_high = sr_mean - z * sr_std, sr_mean + z * sr_std

    span = max(4.0 * sr_std, abs(sr_mean - sr_null) * 1.5 + sr_std)
    xs = np.linspace(sr_mean - span, sr_mean + span, 400)
    pdf = norm.pdf(xs, loc=sr_mean, scale=sr_std)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(xs, pdf, color="C0", lw=1.5, label="asymptotic SR distribution")
    band_mask = (xs >= ci_low) & (xs <= ci_high)
    ax.fill_between(
        xs[band_mask], 0, pdf[band_mask],
        color="C0", alpha=0.20, label=f"{int(round(ci_level * 100))}% CI",
    )
    ax.axvline(
        sr_mean, color="C3", lw=2.0,
        label=f"estimate = {sr_mean:.3f}",
    )
    ax.axvline(
        sr_null, color="gray", lw=1.0, linestyle="--",
        label=f"null = {sr_null:.2f}",
    )

    ax.set_xlabel("annualized Sharpe ratio")
    ax.set_ylabel("density")
    ax.set_title(
        f"SR ~ N({sr_mean:.3f}, {sr_std:.3f}²),  "
        f"95% CI [{ci_low:.3f}, {ci_high:.3f}]"
    )
    ax.legend(loc="best", frameon=False)
    ax.set_ylim(bottom=0)
    return ax

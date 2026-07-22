from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def effective_k(
    trial_returns: pd.DataFrame,
    threshold: float = 0.5,
    method: str = "single",
) -> int:
    """Estimate effective number of independent trials by clustering returns.

    Computes pairwise correlations across columns (each column = one trial's
    returns), converts to a distance metric d = sqrt(0.5 * (1 - r)), runs
    hierarchical linkage, and cuts at the distance corresponding to the given
    correlation threshold. The number of resulting clusters is the effective K.

    threshold: correlation above which trials are considered the same cluster.
    method: linkage method ('single', 'average', 'complete', 'ward').
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    n = trial_returns.shape[1]
    if n <= 1:
        return n

    corr = trial_returns.corr().to_numpy()
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -1.0, 1.0)
    dist = np.sqrt(0.5 * (1.0 - corr))
    np.fill_diagonal(dist, 0.0)

    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method=method)

    dist_threshold = float(np.sqrt(0.5 * (1.0 - threshold)))
    clusters = fcluster(Z, t=dist_threshold, criterion="distance")
    return int(len(np.unique(clusters)))

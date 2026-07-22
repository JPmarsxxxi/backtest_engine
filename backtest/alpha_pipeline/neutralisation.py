"""Skill 4 - Neutralisation.

Remove group-level exposure so the portfolio is balanced within each group
(Finding Alphas, Ch. 5: "Sum(alpha value within same industry) = 0").

Math:
    neutralised(i, t) = signal(i, t) - mean_{j in group(i)} signal(j, t)

    applied cross-sectionally (per timestamp). Subtracting the group mean makes
    every group's signals sum to zero on each bar. For categorical groups this
    is identical to OLS residualisation on group-membership dummies, i.e. it is
    the Ch. 5 neutralisation and a special case of Ch. 13 factor neutralisation.

Grouping options:
    market            - one group = all instruments (dollar-neutral book). No metadata.
    <static column>   - group by any static metadata column (e.g. 'industry',
                        'sector', 'subindustry', 'country'). Pass metadata=.
    dynamic labels    - group by a per-bar (date x asset) label frame, so the
                        grouping can change over time (e.g. volatility deciles,
                        liquidity buckets). Pass labels=. Build these with
                        ``bucket()`` from any numeric field.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def bucket(field: pd.DataFrame, n_buckets: int = 5) -> pd.DataFrame:
    """Turn a numeric field into per-bar integer group labels (0..n_buckets-1).

    Cross-sectional quantile binning per timestamp: rank each bar to a
    percentile, then cut into ``n_buckets`` equal-count groups. Feed the result
    to ``run(signal, labels=...)`` for custom-group neutralisation (WebSim
    ``bucket`` + ``group_neutralize``).

    Parameters
    ----------
    field : pd.DataFrame
        date x asset numeric field to group on (volatility, dollar volume, ...).
    n_buckets : int, default 5
        Number of equal-count buckets per bar (>= 2).

    Returns
    -------
    pd.DataFrame
        date x asset integer bucket labels in [0, n_buckets-1]; NaN preserved.
    """
    if n_buckets < 2:
        raise ValueError(f"n_buckets must be >= 2, got {n_buckets}")
    pct = field.rank(axis=1, pct=True)  # (0, 1] per bar, NaN preserved
    labels = np.ceil(pct * n_buckets) - 1.0
    return labels.clip(lower=0, upper=n_buckets - 1)


def _neutralise_static(signal: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    labels = labels.reindex(signal.columns)
    missing = labels.index[labels.isna()]
    if len(missing):
        raise ValueError(f"assets missing a group label: {list(missing)}")
    out = signal.copy()
    for _, members in labels.groupby(labels):
        cols = members.index
        out[cols] = signal[cols].sub(signal[cols].mean(axis=1), axis=0)
    return out


def _neutralise_dynamic(signal: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Per-bar demean within time-varying groups (date x asset label frame)."""
    labels = labels.reindex(index=signal.index, columns=signal.columns)
    out = signal.copy().astype(float)
    for dt in signal.index:
        row = signal.loc[dt]
        grp = labels.loc[dt].fillna("__nan__")  # NaN-labelled assets form their own group
        out.loc[dt] = row - row.groupby(grp).transform("mean")
    return out


def run(
    signal: pd.DataFrame,
    by: Optional[str] = None,
    metadata: Optional[pd.DataFrame] = None,
    labels: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Demean a signal within groups, per timestamp.

    Parameters
    ----------
    signal : pd.DataFrame
        Raw signal vector, date x asset.
    by : str, optional
        'market' demeans across all instruments (no metadata needed). Any other
        string demeans within that STATIC metadata column (e.g. 'industry',
        'sector', 'subindustry', 'country'); requires ``metadata``.
    metadata : pd.DataFrame, optional
        Static table indexed by asset with the column named by ``by``.
    labels : pd.DataFrame, optional
        date x asset frame of DYNAMIC (time-varying) group labels. If given,
        ``by``/``metadata`` are ignored and neutralisation is done per bar within
        each label group. Build with ``bucket()``.

    Returns
    -------
    pd.DataFrame
        Neutralised signal, date x asset. Each group sums to ~0 per bar.
    """
    if labels is not None:
        return _neutralise_dynamic(signal, labels)

    if by is None:
        raise ValueError("provide `by` (static column / 'market') or `labels` (dynamic)")

    if by == "market":
        return signal.sub(signal.mean(axis=1), axis=0)

    if metadata is None:
        raise ValueError(f"{by!r} neutralisation requires metadata")
    if by not in metadata.columns:
        raise ValueError(f"metadata is missing column {by!r}")

    return _neutralise_static(signal, metadata[by])


def quick_test() -> None:
    """Synthetic smoke test: market, arbitrary static column, dynamic buckets."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    assets = ["BTC", "ETH", "SOL", "ADA", "XRP", "DOGE"]
    signal = pd.DataFrame(rng.normal(0, 1, (5, 6)), index=dates, columns=assets)
    metadata = pd.DataFrame(
        {
            "sector": ["L1", "L1", "L1", "L1", "payments", "meme"],
            "subindustry": ["pow", "pos", "pos", "pos", "pay", "meme"],  # arbitrary column
        },
        index=assets,
    )

    # market: whole-row demean -> each bar's cross-sectional mean is 0
    mkt = run(signal, "market")
    assert np.allclose(mkt.mean(axis=1).values, 0.0)

    # sector (static): each sector group sums to ~0 per bar
    sec = run(signal, "sector", metadata)
    assert np.allclose(sec[["BTC", "ETH", "SOL", "ADA"]].sum(axis=1).values, 0.0)

    # ARBITRARY static column ('subindustry') now works (was rejected before)
    sub = run(signal, "subindustry", metadata)
    assert np.allclose(sub[["ETH", "SOL", "ADA"]].sum(axis=1).values, 0.0)  # 'pos' group
    assert np.allclose(sub[["BTC"]].values, 0.0)  # singleton 'pow'

    # DYNAMIC buckets: group by per-bar quantiles of an arbitrary field, demean within
    field = pd.DataFrame(rng.normal(0, 1, (5, 6)), index=dates, columns=assets)
    labs = bucket(field, n_buckets=2)
    dyn = run(signal, labels=labs)
    # within each bar, each bucket group sums to ~0
    for dt in dates:
        for g, members in labs.loc[dt].groupby(labs.loc[dt]):
            cols = members.index
            assert np.isclose(dyn.loc[dt, cols].sum(), 0.0)

    # guards
    for bad in [dict(), dict(by="sector")]:  # nothing / static-without-metadata
        try:
            run(signal, **bad)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass

    print("neutralisation.quick_test passed")
    print("\nsubindustry-neutral signal, first 2 rows:")
    print(sub.head(2).round(4).to_string())


if __name__ == "__main__":
    quick_test()

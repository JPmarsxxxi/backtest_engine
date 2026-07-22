"""Skill 1 - Universe construction.

Filter a raw price/volume panel down to the tradable universe.

Math (Finding Alphas, Ch. 4 "Alpha Universe"; Ch. 31 universe definition):
    The universe is "the top most-liquid stocks ... determined by the highest
    average daily dollar volume traded."

        dollar_volume[t, i] = price[t, i] * volume[t, i]
        adv[t, i]           = mean of dollar_volume[t-w+1 .. t, i]   (w = 30)
        keep, at each t, the top-N instruments by adv[t, .]

    Categorical pre-filters (asset class / region / sector; Ch. 4 lists these
    as the universe restriction dimensions) restrict the eligible set *before*
    the liquidity ranking, so "top N" is taken within the chosen slice.

Output: boolean DataFrame (date x asset). True == in-universe at that bar.
This is exactly the shape consumed by backtest.data.DataPanel(universe=...).
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

_DEFAULT_ADV_WINDOW = 30

CategoryFilter = Optional[Union[str, Sequence[str]]]


def _eligible_by_category(
    assets: pd.Index,
    metadata: Optional[pd.DataFrame],
    asset_class: CategoryFilter,
    region: CategoryFilter,
    sector: CategoryFilter,
) -> pd.Index:
    """Return the subset of ``assets`` passing the static categorical filters."""
    filters = {"asset_class": asset_class, "region": region, "sector": sector}
    active = {k: v for k, v in filters.items() if v is not None}
    if not active:
        return assets
    if metadata is None:
        raise ValueError(
            f"categorical filter(s) {list(active)} requested but metadata is None"
        )
    mask = pd.Series(True, index=metadata.index)
    for col, val in active.items():
        if col not in metadata.columns:
            raise ValueError(f"metadata is missing column '{col}'")
        wanted = [val] if isinstance(val, str) else list(val)
        mask &= metadata[col].isin(wanted)
    return assets.intersection(metadata.index[mask])


def run(
    prices: pd.DataFrame,
    volume: pd.DataFrame,
    top_n: int,
    metadata: Optional[pd.DataFrame] = None,
    asset_class: CategoryFilter = None,
    region: CategoryFilter = None,
    sector: CategoryFilter = None,
    adv_window: int = _DEFAULT_ADV_WINDOW,
) -> pd.DataFrame:
    """Build the boolean in-universe mask by liquidity and category.

    Parameters
    ----------
    prices, volume : pd.DataFrame
        date x asset, identical index and columns. ``volume`` is share/coin
        volume; dollar volume is computed internally as ``prices * volume``.
    top_n : int
        Number of most-liquid instruments to keep at each bar. Ties at the
        boundary are kept (the mask may exceed ``top_n`` on tie bars).
    metadata : pd.DataFrame, optional
        Static table indexed by asset with optional columns 'asset_class',
        'region', 'sector'. Required only if a categorical filter is passed.
    asset_class, region, sector : str | sequence of str, optional
        Restrict the eligible set before ranking by liquidity.
    adv_window : int, default 30
        Rolling window (in bars) for average dollar volume. ``min_periods=1``,
        so early bars rank on a partial window rather than being dropped.

    Returns
    -------
    pd.DataFrame
        Boolean date x asset mask. True == in-universe.
    """
    if not prices.index.equals(volume.index):
        raise ValueError("prices and volume must share the same index")
    if not prices.columns.equals(volume.columns):
        raise ValueError("prices and volume must share the same columns")
    if top_n < 1:
        raise ValueError("top_n must be >= 1")

    dollar_volume = prices * volume
    adv = dollar_volume.rolling(adv_window, min_periods=1).mean()

    eligible = _eligible_by_category(
        prices.columns, metadata, asset_class, region, sector
    )
    if len(eligible) < len(prices.columns):
        ineligible = prices.columns.difference(eligible)
        adv = adv.copy()
        adv[ineligible] = np.nan

    # Rank descending; method="min" so tied liquidity shares the lower rank and
    # both sides of a boundary tie are retained.
    ranks = adv.rank(axis=1, ascending=False, method="min")
    mask = (ranks <= top_n) & adv.notna()
    return mask.fillna(False).astype(bool)


def quick_test() -> None:
    """Synthetic smoke test: 6 assets, 2 sectors, top_n=3."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    assets = ["BTC", "ETH", "SOL", "ADA", "XRP", "DOGE"]

    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, (60, 6)), axis=0)),
        index=dates, columns=assets,
    )
    # First three assets get structurally higher volume.
    base = np.array([1e6, 8e5, 6e5, 1e4, 8e3, 5e3])
    volume = pd.DataFrame(
        base * (1 + rng.normal(0, 0.05, (60, 6))).clip(0.1),
        index=dates, columns=assets,
    )
    metadata = pd.DataFrame(
        {"sector": ["L1", "L1", "L1", "L1", "payments", "meme"]},
        index=assets,
    )

    mask = run(prices, volume, top_n=3)
    assert mask.shape == prices.shape
    assert mask.dtypes.eq(bool).all()
    last = mask.iloc[-1]
    assert last[["BTC", "ETH", "SOL"]].all()
    assert not last[["ADA", "XRP", "DOGE"]].any()

    # Categorical filter: only L1 sector eligible -> XRP/DOGE never selected.
    masked_l1 = run(prices, volume, top_n=3, metadata=metadata, sector="L1")
    assert not masked_l1[["XRP", "DOGE"]].any().any()
    assert masked_l1.iloc[-1].sum() == 3

    print("universe.quick_test passed")
    print("\nlast-bar universe (top 3 by 30-bar $volume):")
    print(mask.iloc[-1].to_string())


if __name__ == "__main__":
    quick_test()

"""Skill 2 - Prediction frequency + PIT (delay) alignment.

Resample a date x asset panel to the chosen prediction frequency and apply the
delay convention so the value at bar ``t`` respects point-in-time visibility.

Math (Finding Alphas, Ch. 4 "Alpha Prediction Frequency"):
    Frequencies: tick, intraday, daily, weekly, monthly.
        - tick / intraday : native resolution (passthrough). You cannot
          manufacture finer detail than the data already holds.
        - daily / weekly / monthly : calendar resample (close = last bar of
          each bucket for prices; sum for volume).
    Delay:
        - delay 0 (snapshot) : bar t may use its own period's data.
        - delay 1            : "Only data available before the current trading
          day may be used" -> value at bar t is the resampled value from t-1,
          i.e. ``out.shift(1)``. No lookahead.

Note on the engine: backtest.data enforces PIT independently via
``FieldSpec(lag=...)`` (panel.py:13-25) and ``DataView.prices`` slicing
``.loc[:self.t]`` (panel.py:37-39). This module applies its own delay so the
raw-vector pipeline is self-contained; when feeding the engine, set the
adapter's FieldSpec lag to 0 so PIT is not double-applied.
"""

from __future__ import annotations

from typing import Callable, Union

import pandas as pd

_FREQUENCIES = ("tick", "intraday", "daily", "weekly", "monthly")
_PASSTHROUGH = ("tick", "intraday")

# Month-end alias changed to "ME" in pandas 2.2; fall back to "M" on older.
_PD_VERSION = tuple(int(x) for x in pd.__version__.split(".")[:2])
_MONTHLY_RULE = "ME" if _PD_VERSION >= (2, 2) else "M"
_RESAMPLE_RULE = {"daily": "D", "weekly": "W", "monthly": _MONTHLY_RULE}

Agg = Union[str, Callable]


def run(
    data: pd.DataFrame,
    freq: str,
    delay: int,
    agg: Agg = "last",
) -> pd.DataFrame:
    """Resample to ``freq`` and apply the ``delay`` PIT shift.

    Parameters
    ----------
    data : pd.DataFrame
        date x asset. For daily/weekly/monthly the index must be a
        DatetimeIndex. Call once per field (e.g. agg="last" for price,
        agg="sum" for volume).
    freq : {'tick', 'intraday', 'daily', 'weekly', 'monthly'}
        tick/intraday are passthrough (native resolution); the rest resample.
    delay : {0, 1}
        0 = snapshot (bar t sees its own period). 1 = bar t sees only data
        strictly before t (out.shift(1)); introduces a leading NaN row.
    agg : str | callable, default 'last'
        Bucket aggregation for the resample. 'last' = the close (per the PDF's
        use of close as price); 'sum' for volume.

    Returns
    -------
    pd.DataFrame
        Resampled, delay-aligned date x asset frame.
    """
    if freq not in _FREQUENCIES:
        raise ValueError(f"freq must be one of {_FREQUENCIES}, got {freq!r}")
    if delay not in (0, 1):
        raise ValueError(f"delay must be 0 or 1, got {delay!r}")

    if freq in _PASSTHROUGH:
        out = data.copy()
    else:
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError(
                f"freq={freq!r} requires a DatetimeIndex to resample"
            )
        out = data.resample(_RESAMPLE_RULE[freq]).agg(agg)

    if delay:
        out = out.shift(delay)
    return out


def quick_test() -> None:
    """Synthetic smoke test: 240 hourly bars (10 days) x 3 assets."""
    import numpy as np

    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=240, freq="h")
    assets = ["BTC", "ETH", "SOL"]
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.001, (240, 3)), axis=0)),
        index=idx, columns=assets,
    )
    volume = pd.DataFrame(
        rng.uniform(1e3, 1e4, (240, 3)), index=idx, columns=assets
    )

    # daily close = last hourly bar of each day
    daily = run(prices, "daily", delay=0)
    assert len(daily) == 10
    assert np.allclose(daily.iloc[0].values, prices.loc["2024-01-01"].iloc[-1].values)

    # delay 1 shifts everything forward one bar; first row is NaN
    daily_d1 = run(prices, "daily", delay=1)
    assert daily_d1.iloc[0].isna().all()
    assert np.allclose(daily_d1.iloc[1].values, daily.iloc[0].values)

    # volume aggregates by sum
    vol_daily = run(volume, "daily", delay=0, agg="sum")
    assert np.allclose(vol_daily.iloc[0].values, volume.loc["2024-01-01"].sum().values)

    # tick/intraday are passthrough (native resolution preserved)
    tick = run(prices, "tick", delay=0)
    assert tick.shape == prices.shape
    assert np.allclose(tick.values, prices.values)

    # weekly and monthly collapse correctly (10 days -> 2 weeks, 1 month)
    assert len(run(prices, "monthly", delay=0)) == 1
    assert 1 <= len(run(prices, "weekly", delay=0)) <= 3

    print("frequency.quick_test passed")
    print("\ndaily close (delay 0), first 3 rows:")
    print(daily.head(3).round(2).to_string())


if __name__ == "__main__":
    quick_test()

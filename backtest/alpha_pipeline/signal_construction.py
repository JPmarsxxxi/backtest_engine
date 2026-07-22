"""Skill 3 - Signal construction.

Turn a resampled date x asset frame into a raw signal vector (one value per
instrument per bar; + = long, - = short). Named presets cover the common
ideas; ``custom`` runs an arbitrary user/Claude-supplied function so any signal
can be expressed without changing this module.

Math (Finding Alphas):
    mean_reversion (Ch. 5 Alpha1):  signal_t = -sum of log returns over lookback
                                              = -log(close_t / close_{t-L})
        Positive -> long (price fell, expect bounce); negative -> short.
        Using log returns makes "sum of per-period returns" exactly equal the
        book's total-window-return formula -(close_t - close_{t-L})/close_{t-L}
        in log space (no approximation).
    momentum (Ch. 21):              signal_t = +sum of log returns over lookback
        The sign flip of mean_reversion: past winners keep winning.
    fundamental (Ch. 19):           signal_t = field_t - field_{t-L}
        Rate of change of a fundamental field (pass the field as ``data``).
    custom:                         signal = fn(data, **fn_kwargs)
        Arbitrary expression; fn must return a date x asset frame.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

_SIGNAL_TYPES = ("mean_reversion", "momentum", "fundamental", "custom")


def _log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Per-period log returns; one leading NaN row."""
    return np.log(prices).diff()


def run(
    data: pd.DataFrame,
    signal_type: str,
    lookback: Optional[int] = None,
    fn: Optional[Callable[..., pd.DataFrame]] = None,
    **fn_kwargs,
) -> pd.DataFrame:
    """Construct a raw signal vector.

    Parameters
    ----------
    data : pd.DataFrame
        date x asset. Prices for mean_reversion/momentum; the fundamental field
        for fundamental; whatever ``fn`` expects for custom.
    signal_type : {'mean_reversion', 'momentum', 'fundamental', 'custom'}
    lookback : int, optional
        Window in bars. Required for mean_reversion, momentum, fundamental.
    fn : callable, optional
        Required for signal_type='custom'. Called as ``fn(data, **fn_kwargs)``
        and must return a date x asset frame matching ``data``'s shape.
    **fn_kwargs
        Forwarded to ``fn`` for custom signals.

    Returns
    -------
    pd.DataFrame
        Raw signal vector, date x asset. Leading rows are NaN (warmup).
    """
    if signal_type not in _SIGNAL_TYPES:
        raise ValueError(f"signal_type must be one of {_SIGNAL_TYPES}, got {signal_type!r}")

    if signal_type == "custom":
        if fn is None:
            raise ValueError("signal_type='custom' requires fn=<callable>")
        out = fn(data, **fn_kwargs)
        if not isinstance(out, pd.DataFrame):
            raise TypeError("custom fn must return a pd.DataFrame")
        if out.shape != data.shape:
            raise ValueError(
                f"custom fn returned shape {out.shape}, expected {data.shape}"
            )
        return out

    if lookback is None or lookback < 1:
        raise ValueError(f"signal_type={signal_type!r} requires lookback >= 1")

    if signal_type == "mean_reversion":
        return -_log_returns(data).rolling(lookback).sum()
    if signal_type == "momentum":
        return _log_returns(data).rolling(lookback).sum()
    # fundamental: rate of change of the supplied field over the lookback
    return data.diff(lookback)


def quick_test() -> None:
    """Synthetic smoke test: presets + a custom signal."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=50, freq="D")
    assets = ["BTC", "ETH", "SOL"]
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, (50, 3)), axis=0)),
        index=dates, columns=assets,
    )

    mr = run(prices, "mean_reversion", lookback=5)
    mom = run(prices, "momentum", lookback=5)

    # warmup: first 5 rows NaN (need 5 returns -> 6 prices)
    assert mr.iloc[:5].isna().all().all()
    assert mr.iloc[5:].notna().all().all()
    # momentum is exactly the sign flip of mean_reversion
    assert np.allclose(mr.add(mom).dropna().values, 0.0)
    # mean_reversion equals -log(close_t / close_{t-5})
    expected = -np.log(prices / prices.shift(5))
    assert np.allclose(mr.dropna().values, expected.dropna().values)

    # fundamental: change over lookback of a supplied field
    fund = pd.DataFrame(
        rng.normal(0, 1, (50, 3)).cumsum(axis=0), index=dates, columns=assets
    )
    fsig = run(fund, "fundamental", lookback=4)
    assert np.allclose(fsig.values, fund.diff(4).values, equal_nan=True)

    # custom: arbitrary expression, written live (e.g. 3-bar % change momentum)
    out = run(prices, "custom", fn=lambda d, n: d.pct_change(n), n=3)
    assert np.allclose(out.values, prices.pct_change(3).values, equal_nan=True)

    print("signal_construction.quick_test passed")
    print("\nmean_reversion (lookback 5), last 3 rows:")
    print(mr.tail(3).round(4).to_string())


if __name__ == "__main__":
    quick_test()

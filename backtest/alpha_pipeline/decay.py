"""Skill 7 - Decay (smoothing).

Dedicated smoothing pass, applied after clamp/hump (Finding Alphas, Ch. 7:
"smoothed (decayed) ... by using an exponential moving average ... or a simple
or weighted moving average"). One method is chosen per run; the choice depends
on the alpha's horizon (Ch. 7).

Math (per-asset, time-axis; uniform `window` parameter):
    sma    : out_t = (1/N) * sum_{k=0..N-1} signal_{t-k}    (equal-weight MA)
    linear : out_t = sum_{k=0..d-1} (d-k)*signal_{t-k} / [d(d+1)/2]
             linearly-decaying weighted MA (reuses Skill 6 `linear`, d=window).
    ema    : out_t = lam*signal_t + (1-lam)*out_{t-1},  lam = 2/(window+1)
             EMA with the standard span<->alpha bridge (reuses Skill 6 `ema`),
             i.e. ewm(span=window).

`linear` and `ema` delegate to turnover_control so there is a single source of
truth; only `sma` is implemented here.
"""

from __future__ import annotations

import pandas as pd

from backtest.alpha_pipeline.turnover_control import _ema, _linear

_METHODS = ("linear", "ema", "sma")


def run(signal: pd.DataFrame, method: str, window: int) -> pd.DataFrame:
    """Smooth a signal with the chosen decay method.

    Parameters
    ----------
    signal : pd.DataFrame
        date x asset signal vector (typically post clamp/hump).
    method : {'linear', 'ema', 'sma'}
    window : int
        Number of bars. For 'ema' it sets lam = 2/(window+1) (span=window).

    Returns
    -------
    pd.DataFrame
        Smoothed signal, date x asset.
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    if method == "sma":
        return signal.rolling(window, min_periods=window).mean()
    if method == "linear":
        return _linear(signal, d=window)
    return _ema(signal, lam=2.0 / (window + 1))


def quick_test() -> None:
    """Synthetic smoke test: exact-value checks per method."""
    import numpy as np

    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    sig = pd.DataFrame({"A": [10.0, 20.0, 30.0, 40.0]}, index=idx)

    # sma window=3: full-window warmup, then equal-weight means
    sma = run(sig, "sma", window=3)
    assert sma["A"].iloc[:2].isna().all()
    assert np.allclose(sma["A"].iloc[2:].values, [20.0, 30.0])

    # linear window=3: matches Skill 6 linear (140/6, 200/6)
    lin = run(sig, "linear", window=3)
    assert np.allclose(lin["A"].iloc[2:].values, [140 / 6, 200 / 6])

    # ema window=3 -> lam=0.5; equals ewm(span=3) and Skill 6 ema(lam=0.5)
    sig2 = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0]}, index=idx)
    em = run(sig2, "ema", window=3)
    assert np.allclose(em["A"].values, [1.0, 1.5, 2.25, 3.125])
    assert np.allclose(em["A"].values, sig2["A"].ewm(span=3, adjust=False).mean().values)

    print("decay.quick_test passed")
    print("\nsma / linear (window 3) on [10,20,30,40]:")
    print(pd.concat([sma["A"].rename("sma"), lin["A"].rename("linear")], axis=1).round(3).to_string())


if __name__ == "__main__":
    quick_test()

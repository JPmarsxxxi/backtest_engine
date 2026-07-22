"""Skill 5b - Time-series transforms.

Per-instrument (column-wise) rolling transforms that standardise a raw signal
against its OWN recent history, rather than cross-sectionally against peers
(that is Skill 5 rank / Skill 4 neutralisation). WebSim analogues:
``ts_zscore``, ``ts_rank``, ``ts_scale``, ``ts_quantile``.

Math (per asset, rolling window d, current bar t):
    zscore   z_t   = (x_t - mean(x[t-d+1 .. t])) / std(x[t-d+1 .. t])   (ddof=1)
    rank     r_t   = #{k in window : x_k <= x_t} / d                    in (0, 1]
    scale    s_t   = (x_t - min(window)) / (max(window) - min(window))  in [0, 1]
    quantile q_t   = Q_driver( ts_rank )   -- reshape the rank through an inverse
                     CDF (gaussian / uniform / cauchy) to tame / spread the tails.

Use when a signal's level is only meaningful relative to its own trailing
distribution (most mean-reversion signals): standardise first, then rank /
neutralise cross-sectionally.
"""

from __future__ import annotations

from statistics import NormalDist
import math

import numpy as np
import pandas as pd

_METHODS = ("zscore", "rank", "scale", "quantile")
_DRIVERS = ("gaussian", "uniform", "cauchy")
_NORM = NormalDist()


def _zscore(x: pd.DataFrame, d: int) -> pd.DataFrame:
    mu = x.rolling(d, min_periods=d).mean()
    sd = x.rolling(d, min_periods=d).std()  # ddof=1
    return (x - mu) / sd.replace(0.0, np.nan)


def _rank(x: pd.DataFrame, d: int) -> pd.DataFrame:
    # fraction of window values <= current value, in (0, 1]
    def _last_rank(w: np.ndarray) -> float:
        last = w[-1]
        return float(np.sum(w <= last)) / len(w)
    return x.rolling(d, min_periods=d).apply(_last_rank, raw=True)


def _scale(x: pd.DataFrame, d: int) -> pd.DataFrame:
    lo = x.rolling(d, min_periods=d).min()
    hi = x.rolling(d, min_periods=d).max()
    span = (hi - lo).replace(0.0, np.nan)
    return (x - lo) / span


def _q_driver(p: float, driver: str) -> float:
    if np.isnan(p):
        return p
    if driver == "gaussian":
        return _NORM.inv_cdf(p)
    if driver == "uniform":
        return 2.0 * p - 1.0
    # cauchy: inverse CDF = tan(pi*(p - 0.5))
    return math.tan(math.pi * (p - 0.5))


def _quantile(x: pd.DataFrame, d: int, driver: str = "gaussian", eps: float = 1e-6) -> pd.DataFrame:
    if driver not in _DRIVERS:
        raise ValueError(f"driver must be one of {_DRIVERS}, got {driver!r}")
    r = _rank(x, d).clip(lower=eps, upper=1.0 - eps)
    return r.apply(lambda col: col.map(lambda v: _q_driver(v, driver)))


def run(signal: pd.DataFrame, method: str, d: int, **params) -> pd.DataFrame:
    """Apply a per-asset rolling time-series transform.

    Parameters
    ----------
    signal : pd.DataFrame
        date x asset signal (or raw field) vector. NaNs preserved.
    method : {'zscore', 'rank', 'scale', 'quantile'}
    d : int
        Rolling window length in bars (d >= 2). First d-1 rows are NaN (warmup).
    **params
        quantile -> driver ('gaussian' [default] / 'uniform' / 'cauchy')

    Returns
    -------
    pd.DataFrame
        Transformed signal, date x asset; NaNs preserved; warmup bars NaN.
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    if d < 2:
        raise ValueError(f"d must be >= 2, got {d}")
    if method == "zscore":
        return _zscore(signal, d)
    if method == "rank":
        return _rank(signal, d)
    if method == "scale":
        return _scale(signal, d)
    return _quantile(signal, d, driver=params.get("driver", "gaussian"))


def quick_test() -> None:
    """Synthetic smoke test: exact-value checks per method."""
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    sig = pd.DataFrame({"A": [1.0, 3.0, 2.0, 5.0, 4.0]}, index=idx)

    # zscore d=3: warmup 2 NaN, then equals the reference rolling formula
    z = run(sig, "zscore", d=3)
    assert z["A"].iloc[:2].isna().all()
    ref = (sig["A"] - sig["A"].rolling(3).mean()) / sig["A"].rolling(3).std()
    assert np.allclose(z["A"].iloc[2:].values, ref.iloc[2:].values)
    # window [1,3,2] -> (2-2)/1 = 0.0
    assert np.isclose(z["A"].iloc[2], 0.0)

    # rank d=3: window[1,3,2] last=2 -> 2 of 3 <= 2 -> 2/3
    r = run(sig, "rank", d=3)
    assert np.isclose(r["A"].iloc[2], 2 / 3)
    assert (r["A"].iloc[2:] > 0).all() and (r["A"].iloc[2:] <= 1).all()

    # scale d=3: window[1,3,2] last=2 -> (2-1)/(3-1) = 0.5
    s = run(sig, "scale", d=3)
    assert np.isclose(s["A"].iloc[2], 0.5)

    # quantile gaussian: rank 0.5 -> N^-1(0.5) = 0; monotone in rank
    q = run(sig, "quantile", d=3, driver="gaussian")
    # bar where rank == 2/3 maps to a positive z
    assert q["A"].iloc[2] > 0
    # uniform driver: rank r -> 2r-1
    qu = run(sig, "quantile", d=3, driver="uniform")
    assert np.isclose(qu["A"].iloc[2], 2 * (2 / 3) - 1)

    # guards
    for bad in [("zscore", 1), ("nope", 3)]:
        try:
            run(sig, bad[0], d=bad[1])
            raise AssertionError("expected ValueError")
        except ValueError:
            pass

    print("ts_transform.quick_test passed")
    print("\nzscore / rank / scale (d=3) on [1,3,2,5,4]:")
    print(pd.concat([z["A"].rename("zscore"), r["A"].rename("rank"),
                     s["A"].rename("scale")], axis=1).round(4).to_string())


if __name__ == "__main__":
    quick_test()

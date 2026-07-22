"""Spread source layer — produce a per-bar half-spread (bps) frame to feed RealizedSpread.

Two families, both returning a date x asset frame of HALF-spread in basis points
(same units/convention as Spread(half_bps=...) and the RealizedSpread cost model):

  from_bidask(bid, ask)              -- MEASURED: real quotes (FX / crypto / paid
                                        equity feeds). Exact, point-in-time.
  estimate(high, low, close, open_,  -- ESTIMATED: no quotes available (e.g. equities
           method, window)             via yfinance). Rolling estimator from OHLC.

Estimators (proportional full spread S -> reported as half = S/2, in bps):
  "edge"           Ardia, Guidotti & Kroencke (2024, JFE). State of the art from
                   OHLC. Requires the `bidask` package (lazy import). MOST ACCURATE.
  "abdi_ranaldo"   Abdi & Ranaldo (2017). Close + mid-range (high, low). Robust,
                   few negatives. Best ZERO-DEPENDENCY option (default).
  "corwin_schultz" Corwin & Schultz (2012). Two-day high-low ratios.
  "roll"           Roll (1984). Serial covariance of log-price changes. Noisy;
                   undefined (-> 0) when autocovariance is positive.

NOTE: OHLC estimators are DAILY and noisy per name/day -> use a rolling window
(default 21) and treat the level as approximate. Intraday equity strategies need
real quotes; there is no accurate OHLC spread estimate at intraday frequency.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

_BPS = 10_000.0
_METHODS = ("edge", "abdi_ranaldo", "corwin_schultz", "roll")
_CONST = 3.0 - 2.0 * np.sqrt(2.0)  # Corwin-Schultz constant


def from_bidask(bid: pd.DataFrame, ask: pd.DataFrame) -> pd.DataFrame:
    """MEASURED half-spread (bps) from real bid/ask. half_bp = (ask-bid)/mid * 1e4 / 2."""
    mid = (ask + bid) / 2.0
    return (ask - bid).div(mid).mul(_BPS / 2.0)


def _corwin_schultz(high: pd.DataFrame, low: pd.DataFrame, window: int) -> pd.DataFrame:
    hi_t, hi_t1 = high, high.shift(-1)
    lo_t, lo_t1 = low, low.shift(-1)
    himax = np.maximum(hi_t, hi_t1)
    lomin = np.minimum(lo_t, lo_t1)
    beta = np.log(hi_t / lo_t) ** 2 + np.log(hi_t1 / lo_t1) ** 2
    gamma = np.log(himax / lomin) ** 2
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / _CONST - np.sqrt(gamma / _CONST)
    s = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))  # proportional full spread
    s = s.clip(lower=0.0)
    return s.rolling(window, min_periods=max(2, window // 2)).mean()


def _abdi_ranaldo(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, window: int) -> pd.DataFrame:
    eta = (np.log(high) + np.log(low)) / 2.0           # log mid-range
    c = np.log(close)
    x = 4.0 * (c - eta) * (c - eta.shift(-1))          # per-day s^2 term
    s2 = x.rolling(window, min_periods=max(2, window // 2)).mean().clip(lower=0.0)
    return np.sqrt(s2)                                  # proportional full spread


def _roll(close: pd.DataFrame, window: int) -> pd.DataFrame:
    dp = np.log(close).diff()
    x, y = dp, dp.shift(1)
    cov = (x * y).rolling(window, min_periods=max(2, window // 2)).mean() \
        - x.rolling(window, min_periods=max(2, window // 2)).mean() \
        * y.rolling(window, min_periods=max(2, window // 2)).mean()
    return 2.0 * np.sqrt((-cov).clip(lower=0.0))        # proportional full spread


def _edge(open_, high, low, close, window):
    try:
        from bidask import edge_rolling
    except ImportError as e:  # pragma: no cover
        raise ImportError("method='edge' requires the bidask package: pip install bidask") from e
    out = {}
    for col in close.columns:
        df = pd.DataFrame({"open": open_[col], "high": high[col],
                           "low": low[col], "close": close[col]})
        out[col] = edge_rolling(df, window=window)      # proportional full spread series
    return pd.DataFrame(out).reindex(index=close.index, columns=close.columns).clip(lower=0.0)


def estimate(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    open_: Optional[pd.DataFrame] = None,
    method: str = "abdi_ranaldo",
    window: int = 21,
) -> pd.DataFrame:
    """ESTIMATED rolling half-spread (bps) from OHLC. Returns date x asset.

    method: 'edge' (needs bidask; most accurate) | 'abdi_ranaldo' (default, no dep)
            | 'corwin_schultz' | 'roll'. window: rolling bars (>= 2).
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    if method == "edge":
        if open_ is None:
            raise ValueError("method='edge' requires open_ (OHLC).")
        s = _edge(open_, high, low, close, window)
    elif method == "abdi_ranaldo":
        s = _abdi_ranaldo(high, low, close, window)
    elif method == "corwin_schultz":
        s = _corwin_schultz(high, low, window)
    else:
        s = _roll(close, window)
    return s.mul(_BPS / 2.0)  # proportional full spread -> HALF-spread in bps


def quick_test() -> None:
    """Synthetic: exact recovery from bid/ask; order-of-magnitude recovery from OHLC."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", periods=400, freq="B")
    cols = ["X", "Y"]

    # --- from_bidask: exact ---
    mid = pd.DataFrame(100.0, index=dates, columns=cols)
    half_frac = 0.0005  # 5 bps half-spread
    ask = mid * (1 + half_frac)
    bid = mid * (1 - half_frac)
    hb = from_bidask(bid, ask)
    assert np.allclose(hb.values, 5.0, atol=1e-6)  # 5 bps everywhere

    # --- estimators: monotonic in the true spread, non-negative, plausible units ---
    # Clean microstructure sim: high/low = mid +/- half-spread (range ~ the spread),
    # close/open bounce +/- half-spread around a low-vol random-walk mid.
    def _ohlc(h, seed):
        r = np.random.default_rng(seed)
        n = 500
        dts = pd.date_range("2020-01-01", periods=n, freq="B")
        lm = np.log(100.0) + np.cumsum(r.normal(0, 0.003, (n, 2)), axis=0)
        mm = pd.DataFrame(np.exp(lm), index=dts, columns=cols)
        sgn = pd.DataFrame(r.choice([-1.0, 1.0], (n, 2)), index=dts, columns=cols)
        sgn2 = pd.DataFrame(r.choice([-1.0, 1.0], (n, 2)), index=dts, columns=cols)
        return mm * (1 + h), mm * (1 - h), mm * (1 + h * sgn), mm * (1 + h * sgn2)

    small = _ohlc(0.0005, 1)   # 5 bp half-spread
    large = _ohlc(0.0050, 2)   # 50 bp half-spread
    for method in ("abdi_ranaldo", "corwin_schultz", "roll"):
        med_s = estimate(*small, method=method, window=60).stack().median()
        med_l = estimate(*large, method=method, window=60).stack().median()
        assert med_s >= 0 and np.isfinite(med_l), f"{method}: bad values"
        assert med_l > med_s, f"{method}: not monotonic in spread ({med_l:.1f} !> {med_s:.1f})"
        assert 0.1 < med_l < 1000, f"{method}: median {med_l:.1f}bp implausible (unit error?)"

    # edge is optional (needs bidask); only assert non-negative if importable
    try:
        import bidask  # noqa: F401
        est = estimate(*large, method="edge", window=60)
        assert (est.stack().dropna() >= 0).all()
        edge_note = f"edge median {est.stack().median():.1f}bp"
    except ImportError:
        edge_note = "edge skipped (bidask not installed)"

    # guards
    for bad in [("nope", 21, large[3]), ("edge", 21, None), ("roll", 1, None)]:
        try:
            estimate(large[0], large[1], large[2], bad[2], method=bad[0], window=bad[1])
            raise AssertionError("expected ValueError/ImportError")
        except (ValueError, ImportError):
            pass

    print("spread_source.quick_test passed")
    print(f"from_bidask exact 5bp OK | estimators monotonic in spread | {edge_note}")


if __name__ == "__main__":
    quick_test()

"""Skill 6 - Turnover control.

Reduce how fast a signal changes (and thus trading turnover) using the four
methods from Finding Alphas, Ch. 7 "Controlling Turnover". All operate
per-asset along the time axis.

Math:
    clamp  : out_t = clip(signal_t, mu_t - k*sd_t, mu_t + k*sd_t)
             mu, sd are rolling over `window` bars (Ch. 7: bounds as a number
             of standard deviations). Outlier suppression / winsorising.
    hump   : carry-forward (Ch. 7: "values remain unchanged until the threshold
             is crossed"). out_t = out_{t-1} if |signal_t - out_{t-1}| < threshold
             else signal_t.
    ema    : out_t = lam*signal_t + (1-lam)*out_{t-1}   (Ch. 7 EMA verbatim).
    linear : out_t = sum_{k=0..d-1} (d-k)*signal_{t-k} / [d(d+1)/2]
             linearly-decaying weighted MA; most recent bar weighted highest.
    trade_when : event-gated (WebSim `trade_when`). Update the held value to the
             current signal only on bars where `trigger` is true; carry the
             previous value forward otherwise; set NaN (flatten) where `exit` is
             true. Exit takes precedence over trigger. Slashes turnover by only
             trading on chosen events instead of every bar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_METHODS = ("clamp", "hump", "ema", "linear", "trade_when")


def _require(params: dict, name: str, method: str):
    if name not in params:
        raise ValueError(f"method={method!r} requires parameter {name!r}")
    return params[name]


def _clamp(signal: pd.DataFrame, k: float, window: int) -> pd.DataFrame:
    mu = signal.rolling(window, min_periods=window).mean()
    sd = signal.rolling(window, min_periods=window).std()
    # NaN bounds (warmup) leave the value unclipped.
    return signal.clip(lower=mu - k * sd, upper=mu + k * sd)


def _hump(signal: pd.DataFrame, threshold: float) -> pd.DataFrame:
    vals = signal.to_numpy(dtype=float)
    n_rows, n_cols = vals.shape
    held = np.full(n_cols, np.nan)
    out = np.full_like(vals, np.nan)
    for t in range(n_rows):
        cur = vals[t]
        # start tracking each series at its first valid value
        init = np.isnan(held) & ~np.isnan(cur)
        held = np.where(init, cur, held)
        # update the held value only when the move clears the threshold
        big = ~np.isnan(cur) & ~np.isnan(held) & (np.abs(cur - held) >= threshold)
        held = np.where(big, cur, held)
        out[t] = np.where(np.isnan(cur), np.nan, held)
    return pd.DataFrame(out, index=signal.index, columns=signal.columns)


def _ema(signal: pd.DataFrame, lam: float) -> pd.DataFrame:
    if not 0 < lam <= 1:
        raise ValueError(f"lam must be in (0, 1], got {lam}")
    return signal.ewm(alpha=lam, adjust=False).mean()


def _linear(signal: pd.DataFrame, d: int) -> pd.DataFrame:
    if d < 1:
        raise ValueError(f"d must be >= 1, got {d}")
    weights = np.arange(1, d + 1, dtype=float)  # oldest..newest -> 1..d
    denom = d * (d + 1) / 2.0                    # == weights.sum()
    return signal.rolling(d, min_periods=d).apply(
        lambda x: np.dot(weights, x) / denom, raw=True
    )


def _trade_when(signal: pd.DataFrame, trigger: pd.DataFrame, exit_cond: pd.DataFrame) -> pd.DataFrame:
    trig = (trigger.reindex_like(signal).fillna(0).to_numpy() != 0)
    ex = (exit_cond.reindex_like(signal).fillna(0).to_numpy() != 0)
    vals = signal.to_numpy(dtype=float)
    n_rows, n_cols = vals.shape
    held = np.full(n_cols, np.nan)
    out = np.full_like(vals, np.nan)
    for t in range(n_rows):
        held = np.where(ex[t], np.nan, held)          # exit: flatten (precedence)
        upd = trig[t] & ~ex[t]                          # trigger: adopt current signal
        held = np.where(upd, vals[t], held)
        out[t] = held                                   # else: carry previous forward
    return pd.DataFrame(out, index=signal.index, columns=signal.columns)


def run(signal: pd.DataFrame, method: str, **params) -> pd.DataFrame:
    """Apply a turnover-control transform to a signal.

    Parameters
    ----------
    signal : pd.DataFrame
        date x asset signal vector.
    method : {'clamp', 'hump', 'ema', 'linear', 'trade_when'}
    **params
        clamp      -> k (float), window (int)
        hump       -> threshold (float)
        ema        -> lam (float in (0, 1])
        linear     -> d (int)
        trade_when -> trigger (date x asset), exit (date x asset); nonzero = true

    Returns
    -------
    pd.DataFrame
        Turnover-controlled signal, date x asset.
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    if method == "clamp":
        return _clamp(signal, k=_require(params, "k", method),
                      window=_require(params, "window", method))
    if method == "hump":
        return _hump(signal, threshold=_require(params, "threshold", method))
    if method == "ema":
        return _ema(signal, lam=_require(params, "lam", method))
    if method == "linear":
        return _linear(signal, d=_require(params, "d", method))
    return _trade_when(signal,
                       trigger=_require(params, "trigger", method),
                       exit_cond=_require(params, "exit", method))


def quick_test() -> None:
    """Synthetic smoke test: exact-value checks per method."""
    idx = pd.date_range("2024-01-01", periods=5, freq="D")

    # hump (carry-forward): small wiggles held, big move passes through
    sig = pd.DataFrame({"A": [1.0, 1.1, 1.15, 1.0, 5.0]}, index=idx)
    humped = run(sig, "hump", threshold=0.5)
    assert np.allclose(humped["A"].values, [1.0, 1.0, 1.0, 1.0, 5.0])

    # ema: out_t = lam*x_t + (1-lam)*out_{t-1}
    sig2 = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0]}, index=idx[:4])
    em = run(sig2, "ema", lam=0.5)
    assert np.allclose(em["A"].values, [1.0, 1.5, 2.25, 3.125])

    # linear: d=3, weights 1,2,3 over [oldest..newest], denom 6
    sig3 = pd.DataFrame({"A": [10.0, 20.0, 30.0, 40.0]}, index=idx[:4])
    lin = run(sig3, "linear", d=3)
    assert lin["A"].iloc[:2].isna().all()
    assert np.allclose(lin["A"].iloc[2:].values, [140 / 6, 200 / 6])

    # clamp: a spike is pulled inside mu +/- k*sd
    spike = pd.DataFrame({"A": [0, 0, 0, 0, 10.0, 0, 0, 0, 0, 0]},
                         index=pd.date_range("2024-01-01", periods=10))
    cl = run(spike, "clamp", k=1.0, window=5)
    mu = spike.rolling(5, min_periods=5).mean()
    sd = spike.rolling(5, min_periods=5).std()
    hi, lo = mu + sd, mu - sd
    v = hi.notna().values
    assert (cl.values[v] <= hi.values[v] + 1e-9).all()
    assert (cl.values[v] >= lo.values[v] - 1e-9).all()
    assert cl["A"].iloc[4] < spike["A"].iloc[4]  # spike reduced

    # trade_when: only trade on trigger bars, hold otherwise, flatten on exit
    tw_sig = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)
    trig = pd.DataFrame({"A": [1, 0, 1, 0, 0]}, index=idx)   # enter/update on bars 0 and 2
    ex = pd.DataFrame({"A": [0, 0, 0, 1, 0]}, index=idx)     # exit on bar 3
    tw = run(tw_sig, "trade_when", trigger=trig, exit=ex)
    # bar0 trigger->1 ; bar1 hold->1 ; bar2 trigger->3 ; bar3 exit->NaN ; bar4 hold->NaN
    assert tw["A"].iloc[0] == 1.0
    assert tw["A"].iloc[1] == 1.0
    assert tw["A"].iloc[2] == 3.0
    assert np.isnan(tw["A"].iloc[3])
    assert np.isnan(tw["A"].iloc[4])

    print("turnover_control.quick_test passed")
    print("\nhump (threshold 0.5):")
    print(pd.concat([sig["A"].rename("in"), humped["A"].rename("out")], axis=1).to_string())


if __name__ == "__main__":
    quick_test()

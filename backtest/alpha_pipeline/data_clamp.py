"""Skill 2.5 - Data-quality clamp (rolling-band winsorisation on raw data).

Applied to **raw data** before signal construction. Purpose: suppress data-
feed anomalies (bad ticks, corporate-action discontinuities, exchange
glitches) so they do not corrupt the signal computation downstream.

Distinct from the signal-level clamp in turnover_control.py: same math
(rolling mean +/- k*std), different intent and different point in the
pipeline. data_clamp operates on the price/return time series; the
turnover_control clamp operates on the signal vector after rank.

Math:
    out_t = clip(data_t, mu_{t-1} - k*sd_{t-1}, mu_{t-1} + k*sd_{t-1})
    mu, sd: rolling over `window` bars along the time axis, per asset,
    shifted by 1 (trailing) so the current bar does not inflate its own
    band -- the key reason a single-bar outlier still gets caught.

Set trailing=False to use the same-bar band (matches turnover_control._clamp).
For data hygiene, trailing=True is the right default.
"""

from __future__ import annotations

import pandas as pd


def run(
    data: pd.DataFrame,
    k: float = 5.0,
    window: int = 24,
    trailing: bool = True,
) -> pd.DataFrame:
    """Clip a per-asset time series to a rolling +/- k*sigma band.

    Parameters
    ----------
    data : pd.DataFrame
        date x asset (prices, log-returns, volume, etc.) -- the raw data field
        to clean. Call once per field if you have several.
    k : float, default 5.0
        Width of the band in standard deviations. Larger k = more permissive.
        k=5 is loose enough to leave normal moves alone and only catch
        feed-level anomalies (>= 5 sigma in the trailing window).
    window : int, default 24
        Rolling window length in bars. For hourly data, 24 = one day of
        context; for daily data, 24 = a trading month.
    trailing : bool, default True
        If True (recommended for data cleaning), mu and sd are computed on
        the prior window (shifted by 1 bar), so an outlier cannot inflate
        its own band and escape clipping. If False, current bar is included
        (matches turnover_control._clamp semantics).

    Returns
    -------
    pd.DataFrame
        Same date x asset shape; NaNs preserved; warmup bars (first ~window
        rows, depending on trailing) left unclipped where mu/sd are NaN.
    """
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")

    mu = data.rolling(window, min_periods=window).mean()
    sd = data.rolling(window, min_periods=window).std()
    if trailing:
        mu = mu.shift(1)
        sd = sd.shift(1)
    # NaN bounds (warmup) leave the value unclipped via pandas clip behaviour.
    return data.clip(lower=mu - k * sd, upper=mu + k * sd)


def quick_test() -> None:
    """Smoke test: smooth random walk + one injected bad tick, see it clipped."""
    import numpy as np

    rng = np.random.default_rng(0)
    n = 60
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    base = 100.0 + np.cumsum(rng.standard_normal(n) * 0.1)
    base[40] = 100_000.0  # injected bad tick, ~6 orders of magnitude above context
    data = pd.DataFrame({"BTC": base}, index=idx)

    cleaned = run(data, k=5.0, window=24, trailing=True)

    # Warmup rows (first window rows under trailing): mu/sd not yet defined,
    # values left unclipped.
    warmup_n = 24  # window + 0 (the shift means first usable mu/sd is at idx 24)
    assert (cleaned["BTC"].iloc[:warmup_n].equals(data["BTC"].iloc[:warmup_n])), (
        "warmup bars should be unclipped"
    )

    # Bad tick at idx 40: with trailing band over [16..39] (no outlier inside),
    # mu ~ 100, sd ~ 0.1, band ~ [99.5, 100.5]. Tick of 100000 clipped to ~100.5.
    bad_raw = data["BTC"].iloc[40]
    bad_clean = cleaned["BTC"].iloc[40]
    assert bad_raw > 1000, "sanity: raw bad tick is huge"
    assert bad_clean < 110, f"bad tick should be clipped to near 100, got {bad_clean}"
    assert bad_clean > 0, "but stay near the legitimate price level"

    # Surrounding normal bars (idx 39 and idx 41) should be unchanged (within band).
    assert cleaned["BTC"].iloc[39] == data["BTC"].iloc[39], "normal bar untouched"
    # Note: idx 41's trailing band uses [17..40] which INCLUDES the (raw) bad tick.
    # The band becomes huge, so the normal value at 41 stays inside -> unchanged.
    assert cleaned["BTC"].iloc[41] == data["BTC"].iloc[41], "normal bar after spike untouched"

    # Compare to trailing=False: with the bad tick inside its own window, the
    # band inflates and the tick may NOT get clipped. Demonstrates why trailing
    # is the right default.
    cleaned_notrail = run(data, k=5.0, window=24, trailing=False)
    bad_notrail = cleaned_notrail["BTC"].iloc[40]
    assert bad_notrail > bad_clean, (
        "without trailing, the band is inflated by the outlier itself, so the "
        "tick is less aggressively clipped"
    )

    print("data_clamp.quick_test passed")
    print(f"\nBad tick at idx 40:")
    print(f"  raw                  : {bad_raw:12.2f}")
    print(f"  cleaned (trailing)   : {bad_clean:12.4f}")
    print(f"  cleaned (no-trailing): {bad_notrail:12.4f}")
    print(f"\nFirst clipped neighbourhood (idx 38..42):")
    cmp = pd.DataFrame({
        "raw": data["BTC"].iloc[38:43].round(3),
        "cleaned": cleaned["BTC"].iloc[38:43].round(3),
    })
    print(cmp.to_string())


if __name__ == "__main__":
    quick_test()

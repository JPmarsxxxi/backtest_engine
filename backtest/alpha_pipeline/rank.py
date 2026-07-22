"""Skill 5 - Cross-sectional winsorize + rescale.

Replace each bar's signal values with winsorized values, rescaled to [-1, +1].
Unlike pure ordinal rank, this preserves *relative magnitudes* inside the
winsorized band [P_low, P_high] -- only the tails get clipped. Values evenly
spaced in the input come out evenly spaced in [-1, +1]; large outliers no
longer dominate but their distance from the band is still encoded as the
boundary value (-1 or +1).

Math:
    For each bar t with N_t valid (non-NaN) values:
        P_low(t)  = quantile_{low_pct}(signal_t)
        P_high(t) = quantile_{high_pct}(signal_t)
        clipped(i, t) = clip(signal(i, t), P_low(t), P_high(t))
        out(i, t) = 2 * (clipped(i, t) - P_low(t)) / (P_high(t) - P_low(t)) - 1

Defaults: low_pct=0.05, high_pct=0.95. Standard winsorisation tails.

Edge cases:
    N_t == 1                : 0.0 (lone instrument; no relative magnitude).
    N_t == 0                : all NaN.
    P_low(t) == P_high(t)   : 0.0 for all valid entries (degenerate band).
    NaN signal values       : preserved as NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def run(
    signal: pd.DataFrame,
    low_pct: float = 0.05,
    high_pct: float = 0.95,
) -> pd.DataFrame:
    """Winsorise cross-sectionally to the [low_pct, high_pct] band and rescale to [-1, +1].

    Magnitudes inside the band are preserved (linear rescale); values outside
    are pinned to +/-1.

    Parameters
    ----------
    signal : pd.DataFrame
        date x asset signal vector (NaNs allowed).
    low_pct : float, default 0.05
        Lower percentile in (0, 1). Values at or below this percentile map to -1.
    high_pct : float, default 0.95
        Upper percentile in (0, 1), must be > low_pct. Values at or above map to +1.

    Returns
    -------
    pd.DataFrame
        date x asset in [-1, +1]; NaNs preserved; singleton or degenerate bars -> 0.0.
    """
    if not (0.0 <= low_pct < high_pct <= 1.0):
        raise ValueError(
            f"require 0 <= low_pct < high_pct <= 1, got low_pct={low_pct}, high_pct={high_pct}"
        )

    p_low = signal.quantile(low_pct, axis=1)
    p_high = signal.quantile(high_pct, axis=1)
    span = (p_high - p_low).to_numpy()

    vals = signal.to_numpy(dtype=float)
    lo = p_low.to_numpy()
    hi = p_high.to_numpy()

    # Clip and rescale on bars where span > 0.
    clipped = np.minimum(np.maximum(vals, lo[:, None]), hi[:, None])
    with np.errstate(invalid="ignore", divide="ignore"):
        rescaled = 2.0 * (clipped - lo[:, None]) / span[:, None] - 1.0

    # Where span == 0 (all-tied bar), valid entries -> 0.0, NaN preserved.
    degenerate = (span == 0) | np.isnan(span)
    if degenerate.any():
        tied = np.where(np.isnan(vals), np.nan, 0.0)
        rescaled = np.where(degenerate[:, None], tied, rescaled)

    out = pd.DataFrame(rescaled, index=signal.index, columns=signal.columns)

    # Singleton bars (N_t == 1): the lone valid value is neutral (0.0).
    n = signal.notna().sum(axis=1)
    single = n == 1
    if single.any():
        single_mask = pd.DataFrame(False, index=signal.index, columns=signal.columns)
        single_mask.loc[single] = signal.loc[single].notna()
        out = out.mask(single_mask, 0.0)

    # Empty bars (N_t == 0): preserve all NaN.
    empty = n == 0
    if empty.any():
        out.loc[empty] = np.nan

    return out


def quick_test() -> None:
    """Synthetic smoke test: magnitude preservation, ordering, ties, singleton, bounds."""
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    assets = [f"A{i}" for i in range(11)]

    # Row 0: 11 evenly-spaced values [0, 1, ..., 10]. With 5/95 pct on N=11,
    #        quantile(0.05) = 0.5, quantile(0.95) = 9.5. Span = 9.0.
    #        Magnitudes preserved -- evenly-spaced in, evenly-spaced out.
    # Row 1: outlier. [1]*10 + [1000]. quantile(0.05) = 1, quantile(0.95) = 500.5.
    #        Outlier clipped to 500.5 -> +1; all 1s at the lower bound -> -1.
    # Row 2: all tied at 7. Span = 0 -> all 0.0.
    # Row 3: singleton (only A5 valid) -> A5 = 0.0, rest NaN.
    # Row 4: empty -> all NaN.
    # Row 5: 11 values with two-decimal magnitudes [0.1, 0.2, ... 1.1]. Sanity
    #        check that small absolute values rescale to span [-1, +1] correctly.
    sig = pd.DataFrame(
        [
            list(range(11)),
            [1.0] * 10 + [1000.0],
            [7.0] * 11,
            [np.nan] * 5 + [42.0] + [np.nan] * 5,
            [np.nan] * 11,
            [round(0.1 * (i + 1), 2) for i in range(11)],
        ],
        index=dates,
        columns=assets,
        dtype=float,
    )
    out = run(sig, low_pct=0.05, high_pct=0.95)

    # Row 0: evenly spaced; smallest -> -1, largest -> +1, middle near 0.
    # With default 5/95 pct, endpoints get clipped, so the outermost diffs
    # are pinched but the interior diffs (between unclipped values) are equal.
    row0 = out.iloc[0].to_numpy()
    assert row0[0] == -1.0
    assert row0[-1] == 1.0
    assert abs(row0[5]) < 1e-9  # exact middle
    interior_diffs = np.diff(row0[1:-1])  # diffs among unclipped points
    assert np.allclose(interior_diffs, interior_diffs[0], atol=1e-9), (
        "magnitudes inside the band should be preserved (evenly-spaced in -> evenly-spaced out)"
    )

    # Row 0 without winsorisation (low=0, high=1): full magnitude preservation,
    # all diffs equal.
    out_full = run(sig.iloc[[0]], low_pct=0.0, high_pct=1.0)
    full_diffs = np.diff(out_full.iloc[0].to_numpy())
    assert np.allclose(full_diffs, full_diffs[0], atol=1e-9), (
        "with low_pct=0, high_pct=1 (no winsorisation), magnitudes fully preserved"
    )

    # Row 1: outlier clipped; all 1s at -1, the 1000 at +1.
    row1 = out.iloc[1].to_numpy()
    assert np.allclose(row1[:10], -1.0)
    assert row1[10] == 1.0

    # Row 2: all tied -> all 0.0.
    assert np.allclose(out.iloc[2].to_numpy(), 0.0)

    # Row 3: singleton neutral.
    assert out.iloc[3]["A5"] == 0.0
    assert out.iloc[3].drop("A5").isna().all()

    # Row 4: empty -> all NaN.
    assert out.iloc[4].isna().all()

    # Row 5: same shape behavior as row 0 (different scale, same outcome).
    # Scale-invariance: the output should be identical to row 0's output.
    row5 = out.iloc[5].to_numpy()
    assert np.allclose(row5, row0, atol=1e-9), "scale-invariance: shape only depends on rank/value distribution"

    # Global bounds.
    valid = out.dropna(how="all")
    assert (valid.stack().between(-1, 1)).all()

    print("rank.quick_test passed")
    print("\nwinsorised + rescaled, all rows:")
    print(out.round(3).to_string())


if __name__ == "__main__":
    quick_test()

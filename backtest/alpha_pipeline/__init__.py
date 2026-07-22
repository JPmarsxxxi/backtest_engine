"""Modular alpha-generation engine (Finding Alphas, Part II).

These are the computational primitives of the alpha pipeline. The PROTOCOL
cell-loop and the markdown skills call into them; each module is standalone
with a ``run()`` entry point and a ``quick_test()`` at the bottom.

Skills 1-7 pass *raw alpha vectors* (WebSim convention: per-instrument signal
proportional to dollars to hold); a final adapter converts to engine target
weights before evaluation/backtest.

Skill catalog (numbered per the original spec):
    1.   universe            - liquidity + categorical filter -> boolean mask
    2.   frequency           - resample + delay (PIT) alignment
    2.5  data_clamp          - rolling-band winsorise raw data (data hygiene)
    3.   signal_construction - raw signal vector
    3b.  ts_transform        - per-asset rolling zscore/rank/scale/quantile
                               (standardise a signal against its OWN history)
    4.   neutralisation      - demean within group (market / any static column /
                               dynamic per-bar bucket labels; see bucket())
    5.   rank                - cross-sectional winsor+scale to [-1, 1]
    6.   turnover_control    - clamp / hump / ema / linear / trade_when (on signal)
    7.   decay               - smoothing (applied after clamp/hump)
    8.   evaluate            - IR, annual return, max drawdown, turnover, margin
    9.   alpha_correlation   - pool correlation (pearson/temporal/weekly/sign/...)

Note: data_clamp (2.5) and turnover_control's clamp share the same math but
serve different purposes -- data_clamp suppresses raw-data feed anomalies
before signal construction; turnover_control's clamp caps signal-level
outliers after rank. Keep both; they are not redundant.

Execution order: rank (Skill 5) is applied *before* neutralisation (Skill 4),
per Ch. 5 ("Alpha3 = rank(Alpha1)", then "Sum(Alpha3 within industry) = 0").
The modules are pure/order-agnostic; the runtime sequence is:
    universe -> frequency -> data_clamp -> signal_construction -> rank
    -> neutralisation -> turnover_control -> decay -> evaluate
    (-> alpha_correlation).
"""

from backtest.alpha_pipeline import (
    alpha_correlation,
    data_clamp,
    decay,
    evaluate,
    frequency,
    neutralisation,
    rank,
    signal_construction,
    ts_transform,
    turnover_control,
    universe,
)

__all__ = [
    "universe",
    "frequency",
    "data_clamp",
    "signal_construction",
    "ts_transform",
    "neutralisation",
    "rank",
    "turnover_control",
    "decay",
    "evaluate",
    "alpha_correlation",
]

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.data.panel import DataView


class LiquidityCap:
    """Caps each trade at cap_pct * ADV_notional. Returns (executed, unfilled) in dollars.

    ADV is the rolling median of volume in shares over adv_lookback bars; converted to
    dollars using the most recent price.
    """

    def __init__(self, cap_pct: float, adv_lookback: int = 20):
        if not 0 < cap_pct <= 1:
            raise ValueError("cap_pct must be in (0, 1]")
        self.cap_pct = cap_pct
        self.adv_lookback = adv_lookback

    def apply(
        self,
        trades: pd.Series,
        view: DataView,
    ) -> tuple[pd.Series, pd.Series]:
        if view.volume is None:
            return trades.copy(), pd.Series(0.0, index=trades.index)

        prices = view.prices.iloc[-1]
        adv_shares = view.volume.tail(self.adv_lookback).median()
        adv_notional = (adv_shares * prices).reindex(trades.index)
        cap = (self.cap_pct * adv_notional).fillna(np.inf)

        over = trades.abs() > cap
        executed = trades.copy()
        executed.loc[over] = np.sign(trades.loc[over]) * cap.loc[over]
        unfilled = trades - executed
        return executed, unfilled

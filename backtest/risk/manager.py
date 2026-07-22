from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from backtest.data.panel import DataView
from backtest.risk.config import RiskConfig

_TRADING_DAYS = 252


class RiskManager:
    """Applies a RiskConfig to proposed weights to produce final weights."""

    def __init__(self, config: RiskConfig):
        self.config = config

    def apply(
        self,
        proposed: pd.Series,
        state: Mapping,
        data: DataView,
    ) -> pd.Series:
        w = proposed.astype(float).copy()
        cfg = self.config

        if cfg.target_vol is not None:
            w = self._scale_to_target_vol(w, data, cfg.target_vol, cfg.vol_lookback)

        if cfg.max_position is not None:
            w = w.clip(lower=-cfg.max_position, upper=cfg.max_position)

        net = float(w.sum())
        if cfg.max_net is not None and abs(net) > cfg.max_net and abs(net) > 0:
            w = w * (cfg.max_net / abs(net))

        gross_cap = cfg.max_leverage if cfg.max_leverage is not None else cfg.max_gross
        gross = float(w.abs().sum())
        if gross_cap is not None and gross > gross_cap and gross > 0:
            w = w * (gross_cap / gross)

        return w

    @staticmethod
    def _scale_to_target_vol(
        w: pd.Series,
        data: DataView,
        target_vol: float,
        lookback: int,
    ) -> pd.Series:
        rets = data.prices.pct_change().tail(lookback)
        if len(rets) < 2:
            return w
        cov = rets.cov().to_numpy() * _TRADING_DAYS
        cols = rets.columns
        w_vec = w.reindex(cols).fillna(0.0).to_numpy()
        port_var = float(w_vec @ cov @ w_vec)
        if not np.isfinite(port_var) or port_var <= 0:
            return w
        port_vol = np.sqrt(port_var)
        scale = target_vol / port_vol
        return w * scale

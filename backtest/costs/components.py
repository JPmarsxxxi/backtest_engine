from __future__ import annotations

from typing import Mapping, Union

import numpy as np
import pandas as pd

from backtest.costs.base import CostModel
from backtest.data.panel import DataView

_BPS = 10_000.0
_TRADING_DAYS = 252


class Commission(CostModel):
    """Flat commission charged in bps of notional or per share traded.

    Pass exactly one of bps or per_share. Per-share derives share count from
    |trade_dollars| / last_price.
    """

    def __init__(self, bps: float = None, per_share: float = None):
        if (bps is None) == (per_share is None):
            raise ValueError("specify exactly one of bps or per_share")
        self.bps = bps
        self.per_share = per_share

    def trade_cost(self, trades: pd.Series, view: DataView) -> float:
        if self.bps is not None:
            return float(trades.abs().sum() * self.bps / _BPS)
        prices = view.prices.iloc[-1].reindex(trades.index)
        shares = (trades.abs() / prices).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return float(shares.sum() * self.per_share)


class Spread(CostModel):
    """Half-spread in bps. Scalar applies to all assets; mapping is per-asset (missing -> 0)."""

    def __init__(self, half_bps: Union[float, Mapping[str, float], pd.Series]):
        self.half_bps = half_bps

    def trade_cost(self, trades: pd.Series, view: DataView) -> float:
        if isinstance(self.half_bps, (int, float)):
            return float(trades.abs().sum() * self.half_bps / _BPS)
        rates = pd.Series(self.half_bps).reindex(trades.index).fillna(0.0) / _BPS
        return float((trades.abs() * rates).sum())


class RealizedSpread(CostModel):
    """Time-varying half-spread (bps) charged per bar and per asset.

    The point-in-time analogue of ``Spread``: instead of a constant ``half_bps``,
    the rate varies each bar. The spread is sourced either from a panel FEATURE
    (read PIT via the DataView) or from a supplied ``date x asset`` frame.
    Cost = |trade_dollars| * half_bp(t) / 1e4  (half-spread convention). Build the
    input frame with ``backtest.costs.spread_source`` (from_bidask / estimate).

    Parameters
    ----------
    field : str, default 'spread'
        Panel feature name to read per bar (used when ``half_bps`` is None). The
        feature must be a date x asset frame of half-spread in bps.
    half_bps : pd.DataFrame, optional
        date x asset half-spread (bps) frame, used directly (PIT: last row <= t).
        If given, ``field`` is ignored.
    """

    def __init__(self, field: str = "spread", half_bps: pd.DataFrame = None):
        self.field = field
        self._frame = half_bps

    def trade_cost(self, trades: pd.Series, view: DataView) -> float:
        if self._frame is not None:
            rates = self._frame.loc[:view.t].iloc[-1].reindex(trades.index).fillna(0.0)
        else:
            rates = view.feature(self.field).iloc[-1].reindex(trades.index).fillna(0.0)
        return float((trades.abs() * rates / _BPS).sum())


class MarketImpact(CostModel):
    """Per-trade impact as a function of trade-size / ADV.

    sqrt:   impact_bps = k * sqrt(|trade| / ADV_notional)
    linear: impact_bps = k * (|trade| / ADV_notional)

    Volume on the panel is interpreted as shares; ADV_notional = median(volume) * last_price.
    """

    def __init__(self, k: float, kind: str = "sqrt", adv_lookback: int = 20):
        if kind not in ("sqrt", "linear"):
            raise ValueError(f"unknown impact kind: {kind!r}")
        self.k = k
        self.kind = kind
        self.adv_lookback = adv_lookback

    def trade_cost(self, trades: pd.Series, view: DataView) -> float:
        if view.volume is None:
            raise ValueError("MarketImpact requires volume on the data panel")
        prices = view.prices.iloc[-1]
        adv_shares = view.volume.tail(self.adv_lookback).median()
        adv_notional = (adv_shares * prices).reindex(trades.index)
        ratio = (trades.abs() / adv_notional).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if self.kind == "sqrt":
            impact_bps = self.k * np.sqrt(ratio)
        else:
            impact_bps = self.k * ratio
        return float((trades.abs() * impact_bps / _BPS).sum())


class ShortBorrow(CostModel):
    """Annualized borrow rate in bps, charged per bar held on negative positions."""

    def __init__(self, annual_bps: float, trading_days: int = _TRADING_DAYS):
        self.daily_rate = annual_bps / _BPS / trading_days

    def trade_cost(self, trades: pd.Series, view: DataView) -> float:
        return 0.0

    def holding_cost(self, positions: pd.Series, view: DataView) -> float:
        shorts = positions.clip(upper=0.0).abs()
        return float(shorts.sum() * self.daily_rate)

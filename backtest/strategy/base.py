from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Mapping, Optional, Union

import numpy as np
import pandas as pd

from backtest.data.panel import DataView
from backtest.risk.config import DEFAULT_RISK_CONFIG, RiskConfig
from backtest.risk.manager import RiskManager


RebalanceSpec = Union[str, Callable[[pd.DatetimeIndex], pd.DatetimeIndex]]


class Strategy(ABC):
    """Base class. Subclasses must set rebalance_frequency and implement generate_weights.

    rebalance_frequency: one of {"daily","weekly","monthly","quarterly","yearly"},
        a pandas frequency alias (e.g. "W-FRI", "BMS"), or a callable
        (dates -> rebalance_dates).
    risk: RiskConfig. Strategy may also override apply_risk for custom logic.
    """

    rebalance_frequency: RebalanceSpec = None
    risk: RiskConfig = DEFAULT_RISK_CONFIG

    def required_data(self) -> dict:
        return {"prices": None}

    @abstractmethod
    def generate_weights(self, data: DataView, t: pd.Timestamp) -> pd.Series: ...

    def fit(self, data: DataView) -> None:
        return None

    def apply_risk(
        self,
        proposed: pd.Series,
        state: Mapping,
        data: DataView,
    ) -> pd.Series:
        return RiskManager(self.risk).apply(proposed, state, data)

    def rebalance_dates(self, dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
        freq = self.rebalance_frequency
        if freq is None:
            raise ValueError(
                f"{type(self).__name__}.rebalance_frequency is not set"
            )
        if callable(freq):
            return pd.DatetimeIndex(freq(dates))
        return _resolve_frequency(dates, freq)


_BUILTIN_KEYS = {
    "daily": None,
    "weekly": "week",
    "monthly": "month",
    "quarterly": "quarter",
    "yearly": "year",
}


def _resolve_frequency(dates: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    if freq in ("daily", "D", "B"):
        return dates
    if freq in _BUILTIN_KEYS:
        keys = _bucket_keys(dates, _BUILTIN_KEYS[freq])
        mask = np.r_[True, keys[1:] != keys[:-1]]
        return dates[mask]
    grid = pd.date_range(dates[0], dates[-1], freq=freq)
    pos = dates.searchsorted(grid)
    pos = pos[pos < len(dates)]
    if len(pos) == 0:
        return dates[:0]
    return dates[np.unique(pos)]


def _bucket_keys(dates: pd.DatetimeIndex, kind: str) -> np.ndarray:
    if kind == "week":
        iso = dates.isocalendar()
        return (iso.year.values * 100 + iso.week.values).astype(np.int64)
    if kind == "month":
        return (dates.year.values * 12 + dates.month.values).astype(np.int64)
    if kind == "quarter":
        return (dates.year.values * 4 + dates.quarter.values).astype(np.int64)
    if kind == "year":
        return dates.year.values.astype(np.int64)
    raise ValueError(f"unknown bucket: {kind}")

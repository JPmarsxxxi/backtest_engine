from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import pandas as pd

from backtest.data.panel import DataView


class CostModel(ABC):
    """Cost interface. Trades are signed dollar amounts; positions are signed dollar exposures."""

    @abstractmethod
    def trade_cost(self, trades: pd.Series, view: DataView) -> float: ...

    def holding_cost(self, positions: pd.Series, view: DataView) -> float:
        return 0.0


class CompositeCostModel(CostModel):
    """Sum of multiple cost models."""

    def __init__(self, models: Iterable[CostModel]):
        self.models = list(models)

    def trade_cost(self, trades: pd.Series, view: DataView) -> float:
        return float(sum(m.trade_cost(trades, view) for m in self.models))

    def holding_cost(self, positions: pd.Series, view: DataView) -> float:
        return float(sum(m.holding_cost(positions, view) for m in self.models))

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
import pandas as pd

from backtest.data import DataPanel


class PathGenerator(ABC):
    """Produces synthetic return tensors for Monte Carlo backtesting.

    sample() returns shape (n_paths, n_steps, n_assets) of simple returns.
    The MC engine materializes each path into a DataPanel via panel_from_returns.
    """

    @abstractmethod
    def sample(
        self,
        n_paths: int,
        n_steps: int,
        n_assets: int,
        seed: int,
    ) -> np.ndarray: ...

    @abstractmethod
    def config(self) -> dict: ...


def panel_from_returns(
    returns: np.ndarray,
    assets: Sequence[str],
    dates: pd.DatetimeIndex,
    init_price: float = 100.0,
) -> DataPanel:
    """Materialize one path of simple returns into a DataPanel.

    returns shape: (n_steps, n_assets). Prices seed at init_price and cumprod.
    """
    if returns.ndim != 2:
        raise ValueError(
            f"returns must be 2-D (n_steps, n_assets), got {returns.shape}"
        )
    if returns.shape[0] != len(dates):
        raise ValueError(
            f"returns has {returns.shape[0]} steps, dates has {len(dates)}"
        )
    if returns.shape[1] != len(assets):
        raise ValueError(
            f"returns has {returns.shape[1]} assets, assets has {len(assets)}"
        )
    prices = init_price * np.cumprod(1.0 + returns, axis=0)
    df = pd.DataFrame(prices, index=pd.DatetimeIndex(dates), columns=list(assets))
    # check_outliers=False: synthetic data shouldn't trigger real-data warnings.
    return DataPanel(df, check_outliers=False)

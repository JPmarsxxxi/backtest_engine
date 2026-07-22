from __future__ import annotations

import hashlib

import pandas as pd

from backtest.data.panel import DataPanel
from backtest.strategy.base import Strategy


def strategy_hash(strategy: Strategy) -> str:
    """Stable 16-char hash from strategy class + repr(strategy).

    Relies on the strategy having a stable __repr__. Default object repr is
    instance-address-based and will produce different hashes for identical
    configurations - implement __repr__ on your strategy classes for proper
    deduplication.
    """
    cls = type(strategy)
    rep = f"{cls.__module__}.{cls.__name__}|{repr(strategy)}"
    return hashlib.sha256(rep.encode()).hexdigest()[:16]


def dataset_id_for(panel: DataPanel) -> str:
    """Auto-derived dataset ID from panel shape, dates, and sampled price sums."""
    cols = ",".join(map(str, panel.assets_all))
    start = pd.Timestamp(panel.dates[0]).isoformat()
    end = pd.Timestamp(panel.dates[-1]).isoformat()
    n = len(panel.dates)
    head_sum = float(panel._prices.iloc[: min(5, n)].sum().sum())
    tail_sum = float(panel._prices.iloc[-min(5, n) :].sum().sum())
    rep = f"{cols}|{start}|{end}|{n}|{head_sum:.4f}|{tail_sum:.4f}"
    return hashlib.sha256(rep.encode()).hexdigest()[:16]

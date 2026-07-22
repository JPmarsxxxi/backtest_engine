from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np

from backtest.costs.base import CostModel
from backtest.engine.engine import Engine
from backtest.metrics import compute_metrics
from backtest.registry.hashing import strategy_hash
from backtest.simulation.probes import ProbeBattery
from backtest.strategy.base import Strategy

StrategyOrFactory = Union[Strategy, Callable[[], Strategy]]


def fingerprint(
    strategy: StrategyOrFactory,
    battery: Optional[ProbeBattery] = None,
    warmup_bars: int = 60,
    cache: Optional["FingerprintCache"] = None,
    costs: Optional[CostModel] = None,
    initial_capital: float = 1_000_000.0,
) -> dict[str, float]:
    """Score a strategy on each probe; return delta-Sharpe vs the IID baseline.

    Caches by (strategy_hash, battery_id). Strategies must implement __repr__
    on __init__ args for cache stability (see registry/hashing.py).
    """
    if battery is None:
        battery = ProbeBattery()
    is_factory = callable(strategy) and not isinstance(strategy, Strategy)
    strat_for_hash = strategy() if is_factory else strategy

    bid = battery.battery_id()
    if cache is not None:
        cached = cache.get(strat_for_hash, bid)
        if cached is not None:
            return cached

    sharpes: dict[str, float] = {}
    for probe in battery.probes():
        s = strategy() if is_factory else copy.deepcopy(strategy)
        engine = Engine(
            data=probe.panel,
            strategy=s,
            costs=costs,
            initial_capital=initial_capital,
        )
        start = probe.panel.dates[warmup_bars] if warmup_bars > 0 else None
        result = engine.run(start=start)
        rep = compute_metrics(result.returns.dropna(), result.equity_curve.dropna())
        sharpes[probe.name] = float(rep.sharpe) if np.isfinite(rep.sharpe) else 0.0

    baseline_sharpe = sharpes[battery.baseline().name]
    scores = {name: float(s - baseline_sharpe) for name, s in sharpes.items()}

    if cache is not None:
        cache.put(strat_for_hash, bid, scores)
    return scores


class FingerprintCache:
    """JSON-file cache keyed by strategy hash + battery id."""

    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, strategy: Strategy, battery_id: str) -> Path:
        return self.cache_dir / f"{strategy_hash(strategy)}_{battery_id}.json"

    def get(self, strategy: Strategy, battery_id: str) -> Optional[dict[str, float]]:
        p = self._path(strategy, battery_id)
        if not p.exists():
            return None
        return json.loads(p.read_text())

    def put(self, strategy: Strategy, battery_id: str, scores: dict[str, float]) -> None:
        self._path(strategy, battery_id).write_text(json.dumps(scores))

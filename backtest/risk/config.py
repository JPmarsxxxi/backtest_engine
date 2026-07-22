from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RiskConfig:
    """Declarative portfolio-level risk constraints.

    Constraints are applied in order in RiskManager:
        1. target_vol (best-effort scale)
        2. max_position (hard clip)
        3. max_net (proportional rescale)
        4. max_gross / max_leverage (proportional rescale)

    Later constraints win when they conflict.

    Note: this engine intentionally has no drawdown kill switch. Backtests
    are meant to reveal whether a strategy recovers from drawdowns; a
    kill switch would silently truncate that information. If you want
    drawdown-conditional sizing for a *live* deployment, implement it in
    a custom `Strategy.apply_risk` override and keep it out of the
    backtest config.
    """
    max_position: Optional[float] = 0.20
    max_gross: float = 1.0
    max_net: float = 1.0
    max_leverage: Optional[float] = None
    target_vol: Optional[float] = None
    vol_lookback: int = 60


DEFAULT_RISK_CONFIG = RiskConfig()

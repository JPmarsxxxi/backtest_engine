# Cell 6 — Costs: custom time-varying spread cost model (#027)
from backtest.costs import CompositeCostModel
from backtest.costs.base import CostModel


class TimeVaryingSpread(CostModel):
    """Charge the REAL per-bar half-spread on each executed trade: |trades| * half_bp(t) / 1e4.
    Reads the 'spread' feature (per-pair half-spread in bp) PIT via the DataView. holding_cost=0
    (swap-free by the rollover flatten). Time-varying analogue of the built-in Spread (components.py:36-46)."""

    def trade_cost(self, trades, view):
        half_bp = view.feature("spread").iloc[-1].reindex(trades.index).fillna(0.0)
        return float((trades.abs() * half_bp / 1e4).sum())


costs = CompositeCostModel([TimeVaryingSpread()])
print("Cost components:", [type(m).__name__ for m in costs.models])
print("panel has 'spread' field:", panel.has_field("spread"))
print("panel has 'volume' field:", panel.has_field("volume"), "(no MarketImpact/LiquidityCap -> fine)")

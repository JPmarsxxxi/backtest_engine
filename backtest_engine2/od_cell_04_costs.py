# Cell 4 — Stage 3: spread-only cost stack (FTMO index CFD).
# No commission (FTMO indices are commission-free). No MarketImpact/LiquidityCap (no real
# volume data; trade size trivial vs ES depth — decided Cell 1). half_bps=1.0 ~= 0.5pt at
# S&P 5000; 0.5/2.0bp sensitivity at Stage 7 covers the points-vs-bps drift.
from backtest.costs import Spread, CompositeCostModel

costs = CompositeCostModel([Spread(half_bps=1.0)])
liq = None

print("Cost components:", [type(m).__name__ for m in costs.models])
print("Round-trip cost: ~2.0 bp/night vs paper drift ~1.4 bp/night (the gate under test)")

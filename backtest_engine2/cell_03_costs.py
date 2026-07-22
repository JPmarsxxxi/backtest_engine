# Cell 3 — Costs + liquidity (realistic S&P 100 large-cap stack)
from backtest.costs import (
    Commission, Spread, MarketImpact, ShortBorrow,
    CompositeCostModel, LiquidityCap,
)

costs = CompositeCostModel([
    Commission(bps=1),                                  # 1 bp institutional commission
    Spread(half_bps=1),                                 # 1 bp half-spread (2 bp full)
    MarketImpact(k=10, kind="sqrt", adv_lookback=20),   # sqrt impact, 20-bar ADV
    ShortBorrow(annual_bps=50, trading_days=252),       # 0.5%/yr easy-to-borrow
])
liq = LiquidityCap(cap_pct=0.10, adv_lookback=20)       # 10% of ADV

print("Cost components:", [type(m).__name__ for m in costs.models])
print(f"LiquidityCap: cap_pct={liq.cap_pct}, adv_lookback={liq.adv_lookback}")
print(f"Panel has volume: {panel.has_field('volume')}")

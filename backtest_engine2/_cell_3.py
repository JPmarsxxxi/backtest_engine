# Cell 4 â€” Costs + Liquidity
from backtest.costs import (
    Commission, Spread, MarketImpact, ShortBorrow,
    CompositeCostModel, LiquidityCap,
)

costs = CompositeCostModel([
    Commission(bps=10),                                        # 10 bps per executed trade
    Spread(half_bps=5),                                        # 5 bps half-spread
    MarketImpact(k=10, kind="sqrt", adv_lookback=20),          # sqrt Almgren-style, 20-bar ADV
    ShortBorrow(annual_bps=300, trading_days=6048),             # 6048 = 252d Ã— 24h; long-only â†’ zero charge
])

liq = LiquidityCap(cap_pct=0.10, adv_lookback=20)              # cap each trade to 10% of 20-bar ADV

print("Cost components :", [type(m).__name__ for m in costs.models])
print(f"LiquidityCap    : cap_pct={liq.cap_pct}, adv_lookback={liq.adv_lookback}")
print(f"Panel has volume: {panel.has_field('volume')}")

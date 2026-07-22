# Cell 9 — Costs: mid-cap realistic stack
from backtest.costs import (
    Commission, Spread, MarketImpact, ShortBorrow,
    CompositeCostModel, LiquidityCap,
)

costs_mid = CompositeCostModel([
    Commission(bps=1.5),                                # 1.5 bp commission
    Spread(half_bps=6),                                 # 6 bp half-spread (12 bp full) — mid-cap
    MarketImpact(k=12, kind="sqrt", adv_lookback=20),   # higher impact than large-cap
    ShortBorrow(annual_bps=150, trading_days=252),      # 1.5%/yr borrow
])
liq_mid = LiquidityCap(cap_pct=0.08, adv_lookback=20)   # 8% of ADV

print("Cost components:", [type(m).__name__ for m in costs_mid.models])
print(f"LiquidityCap: cap_pct={liq_mid.cap_pct}, adv_lookback={liq_mid.adv_lookback}")
print(f"panel_mid has volume: {panel_mid.has_field('volume')}")

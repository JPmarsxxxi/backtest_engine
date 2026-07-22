# Cell 3 — FTMO-calibrated crypto CFD cost stack
from backtest.costs import (
    Commission, Spread, MarketImpact, ShortBorrow,
    CompositeCostModel, LiquidityCap,
)

# Symmetric CFD swap S=5%/yr per side; dollar-neutral book (shorts = gross/2)
# => model both legs exactly as ShortBorrow(2*S) on the short notional.
costs_ftmo = CompositeCostModel([
    Commission(bps=3.25),                               # FTMO published 0.0325%/side
    Spread(half_bps=5),                                 # blended majors/alts (as #005/#006)
    MarketImpact(k=12, kind="sqrt", adv_lookback=20),
    ShortBorrow(annual_bps=1000, trading_days=365),     # 2*S, S=500bp/side baseline
])
liq_ftmo = LiquidityCap(cap_pct=0.05, adv_lookback=20)

print("Cost components:", [type(m).__name__ for m in costs_ftmo.models])
print(f"LiquidityCap: cap_pct={liq_ftmo.cap_pct}, adv_lookback={liq_ftmo.adv_lookback}")
print(f"panel_skew has volume: {panel_skew.has_field('volume')}")

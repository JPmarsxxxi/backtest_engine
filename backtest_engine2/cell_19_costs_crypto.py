# Cell 19 — Crypto cost stack
from backtest.costs import (
    Commission, Spread, MarketImpact, ShortBorrow,
    CompositeCostModel, LiquidityCap,
)
costs_crypto = CompositeCostModel([
    Commission(bps=5),                                  # ~5 bp/side (Binance VIP-ish)
    Spread(half_bps=5),                                 # blended majors+alts
    MarketImpact(k=12, kind="sqrt", adv_lookback=20),
    ShortBorrow(annual_bps=100, trading_days=365),      # nominal; perp funding is a wildcard
])
liq_crypto = LiquidityCap(cap_pct=0.05, adv_lookback=20)
print("Cost components:", [type(m).__name__ for m in costs_crypto.models])
print(f"panel_crypto has volume: {panel_crypto.has_field('volume')}")

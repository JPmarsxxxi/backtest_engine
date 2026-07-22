# Cell 3 — Costs + liquidity (mandatory per PROTOCOL §6)
from backtest.costs import (
    Commission, Spread, MarketImpact, ShortBorrow,
    CompositeCostModel, LiquidityCap,
)

# Hourly crypto on Binance spot. trading_days=8760 (= 365*24) is the hourly annualization
# factor for ShortBorrow — the default 252 would silently inflate borrow ~35x.
HOURS_PER_YEAR = 365 * 24  # = 8760
BORROW_ANNUAL_BPS = 1500

costs = CompositeCostModel([
    Commission(bps=10),                                                # Binance spot taker fee
    Spread(half_bps=5),                                                # ~10 bps round-trip (mix of majors & alts)
    MarketImpact(k=10, kind="sqrt", adv_lookback=168),                 # sqrt impact, 1-week ADV
    ShortBorrow(annual_bps=BORROW_ANNUAL_BPS, trading_days=HOURS_PER_YEAR),  # 15% annual on shorts, hourly
])

liq = LiquidityCap(cap_pct=0.10, adv_lookback=168)                     # 10% of 1-week median hourly volume

# Pre-flight: volume MUST be on the panel (MarketImpact + LiquidityCap both require it).
print("Cost components :", [type(m).__name__ for m in costs.models])
for m in costs.models:
    if isinstance(m, Commission):
        print(f"  Commission        bps         = {m.bps}")
    elif isinstance(m, Spread):
        print(f"  Spread            half_bps    = {m.half_bps}")
    elif isinstance(m, MarketImpact):
        print(f"  MarketImpact      k={m.k}, kind={m.kind!r}, adv_lookback={m.adv_lookback}")
    elif isinstance(m, ShortBorrow):
        # ShortBorrow only stores daily_rate; back-derive the annualized rate from it.
        implied_annual_pct = m.daily_rate * HOURS_PER_YEAR * 100
        print(f"  ShortBorrow       per-bar rate={m.daily_rate:.3e}  (annualized = {implied_annual_pct:.2f}%)")

print(f"\nLiquidityCap      cap_pct={liq.cap_pct}, adv_lookback={liq.adv_lookback}")
print(f"Panel has volume  : {panel.has_field('volume')}  (must be True)")

# Smoke check: hand-compute one bar's cost on a trivial trade to make sure nothing blows up.
import pandas as pd
sample_view = panel.as_of(panel.dates[-1])
sample_trades = pd.Series({"BTCUSDT": 10_000.0, "ETHUSDT": -10_000.0}).reindex(panel.assets_all).fillna(0.0)
sample_positions = sample_trades.copy()
tc = costs.trade_cost(sample_trades, sample_view)
hc = costs.holding_cost(sample_positions, sample_view)
print(f"\nSmoke test (last bar, +$10k BTC / -$10k ETH):")
print(f"  trade_cost   = ${tc:,.2f}   ({tc / 20_000 * 1e4:.1f} bps of $20k notional)")
print(f"  holding_cost = ${hc:,.4f}   (only the $10k short ETH leg charged; annualized = {hc / 10_000 * HOURS_PER_YEAR * 100:.2f}%)")

# Cell 3 — Costs + liquidity (Stage 3, mandatory per PROTOCOL)
# Long-only crypto hourly: Commission + Spread + sqrt impact; ShortBorrow kept for engine parity.

import pandas as pd

from backtest.costs import (
    Commission,
    Spread,
    MarketImpact,
    ShortBorrow,
    CompositeCostModel,
    LiquidityCap,
)

HOURS_PER_YEAR = 365 * 24
TAKER_BPS = 10
HALF_SPREAD_BPS = 5
IMPACT_K = 10
ADV_LOOKBACK = 168
LIQ_CAP_PCT = 0.10
BORROW_ANNUAL_BPS = 1500  # only applies if strategy shorts; long-only -> ~0 holding

costs = CompositeCostModel([
    Commission(bps=TAKER_BPS),
    Spread(half_bps=HALF_SPREAD_BPS),
    MarketImpact(k=IMPACT_K, kind="sqrt", adv_lookback=ADV_LOOKBACK),
    ShortBorrow(annual_bps=BORROW_ANNUAL_BPS, trading_days=HOURS_PER_YEAR),
])

liq = LiquidityCap(cap_pct=LIQ_CAP_PCT, adv_lookback=ADV_LOOKBACK)

print("Cost components:", [type(m).__name__ for m in costs.models])
for m in costs.models:
    if isinstance(m, Commission):
        print(f"  Commission        bps         = {m.bps}")
    elif isinstance(m, Spread):
        print(f"  Spread            half_bps    = {m.half_bps}")
    elif isinstance(m, MarketImpact):
        print(f"  MarketImpact      k={m.k}, kind={m.kind!r}, adv_lookback={m.adv_lookback}")
    elif isinstance(m, ShortBorrow):
        implied_annual_pct = m.daily_rate * HOURS_PER_YEAR * 100
        print(
            f"  ShortBorrow       per-bar rate={m.daily_rate:.3e}  "
            f"(annualized = {implied_annual_pct:.2f}%)"
        )

print(f"\nLiquidityCap      cap_pct={liq.cap_pct}, adv_lookback={liq.adv_lookback}")
print(f"Panel has volume  : {panel.has_field('volume')}  (required True)")

# Long-only smoke: $10k BTC buy on last bar with valid BTC price
_btc_idx = close["BTCUSDT"].dropna().index
if len(_btc_idx) == 0:
    _probe_t = panel.dates[-1]
    _probe_sym = panel.assets_all[0]
else:
    _probe_t = _btc_idx[-1]
    _probe_sym = "BTCUSDT"

sample_view = panel.as_of(_probe_t)
sample_trades = pd.Series({_probe_sym: 10_000.0}).reindex(panel.assets_all).fillna(0.0)
tc = costs.trade_cost(sample_trades, sample_view)
hc = costs.holding_cost(sample_trades, sample_view)
print(f"\nSmoke test ({_probe_t}, +$10k {_probe_sym} long):")
print(f"  trade_cost   = ${tc:,.2f}   ({tc / 10_000 * 1e4:.1f} bps of $10k notional)")
print(f"  holding_cost = ${hc:,.4f}   (long-only -> expect ~0)")
# Cell 3 — Stage 3: costs + liquidity (spec.json §2.3).
# Linear k=0.7 bps round-trip -> 0.35 bps per trade leg (Commission).
# Sqrt impact: engine MarketImpact(k, kind="sqrt") approximates paper eta=0.02
# (different functional form — see mismatch note below).
import pandas as pd
from backtest.costs import Commission, MarketImpact, CompositeCostModel, LiquidityCap
from backtest.engine import Engine

RT_BPS = 0.7
LEG_BPS = RT_BPS / 2.0          # one rebalance leg = half a round trip
IMPACT_K = 2.0                    # sqrt-impact scale; paper eta=0.02 (not 1:1 with k)
ADV_LOOKBACK = 20

costs = CompositeCostModel([
    Commission(bps=LEG_BPS),
    MarketImpact(k=IMPACT_K, kind="sqrt", adv_lookback=ADV_LOOKBACK),
])
liq = LiquidityCap(cap_pct=0.10, adv_lookback=ADV_LOOKBACK)

print("Cost components:", [type(m).__name__ for m in costs.models])
print(f"  Commission: {LEG_BPS} bps/leg ({RT_BPS} bps round-trip)")
print(f"  MarketImpact: k={IMPACT_K}, kind=sqrt, adv_lookback={ADV_LOOKBACK}")
print(f"LiquidityCap: cap_pct={liq.cap_pct}, adv_lookback={liq.adv_lookback}")
print(f"Panel has volume: {panel.has_field('volume')}")

# Worked example: $10k rebalance on latest bar (<< 1% of $1M — typical here).
view = panel.as_of(panel.dates[-1])
px = float(view.prices[ASSET].iloc[-1])
vol_med = float(view.volume[ASSET].tail(ADV_LOOKBACK).median())
adv_usd = vol_med * px
trade = pd.Series({ASSET: 10_000.0})
comm = costs.models[0].trade_cost(trade, view)
impact = costs.models[1].trade_cost(trade, view)
ratio = 10_000.0 / adv_usd
print(f"\nSanity @ {view.t.date()}  GC=F close={px:,.0f}  "
      f"ADV~{adv_usd/1e9:.2f}B notional ({vol_med:,.0f} oz = {vol_med/GC_CONTRACT_OZ:,.0f} contracts)")
print(f"  $10k trade: commission=${comm:.2f}  impact=${impact:.2f}  "
      f"total=${comm + impact:.2f}  (participation={ratio:.2e})")

# Pre-flight: Engine must construct (LiquidityCap needs volume on panel).
Engine(panel, strat, costs=costs, liquidity=liq)
print("Engine(panel, strat, costs, liquidity) construction: OK")

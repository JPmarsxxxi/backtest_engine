# Cell 5 — Stage 3: costs + liquidity (FTMO crypto CFD intraday stack).
# ShortBorrow deliberately absent: strategy is flat overnight by construction (no swap
# is the point of #011), crypto CFD shorts pay no intraday borrow, and ShortBorrow's
# per-bar charging (components.py:79-90) mis-annualizes ~70x on 30m bars.
from backtest.costs import Commission, Spread, MarketImpact, CompositeCostModel, LiquidityCap

costs = CompositeCostModel([
    Commission(bps=3.25),                              # per side, matches #009/#010 stack
    Spread(half_bps=5),                                # baseline; 00:00 UTC widening = later sensitivity
    MarketImpact(k=12, kind="sqrt", adv_lookback=20),  # trailing 10h of 30m-bar liquidity
])
liq = LiquidityCap(cap_pct=0.05, adv_lookback=20)      # 5% of median 30m bar volume

print("Cost components:", [type(m).__name__ for m in costs.models])
print(f"LiquidityCap: cap_pct={liq.cap_pct}, adv_lookback={liq.adv_lookback}")
print(f"Panel has volume: {panel.has_field('volume')}")

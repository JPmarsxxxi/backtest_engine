# Cell 2 — #019 strategy: continuous TS reversal on US500, VOL-TARGETED (FTMO-safe sizing).
# signal = z-score of L-day log return over a trailing window, sign-flipped (buy after a drop),
# capped +/-3 sigma -> [-1,1]. Sizing: w = signal * (target_vol / realized_vol), hard cap +/-0.5
# so position SHRINKS in high vol (crisis-day protection) and a -10% day stays ~<=5% loss.
# Lookback L=5 is the pre-registered primary (past-5d return dominated the #018 horse-race, t -3.7).
# Lookback sweep {1,2,3,5,10} runs LATER as registered trials (PROTOCOL §6), not optimized here.
from backtest.strategy import Strategy
from backtest.risk import RiskConfig


class IndexReversal(Strategy):
    """Buy US500 after it falls / fade rallies; z-scored L-day reversal, vol-targeted, capped."""
    rebalance_frequency = "daily"

    def __init__(self, lookback=5, win=120, vol_span=20,
                 target_vol=0.10, cap=0.5, vol_floor=0.05):
        self.lookback = lookback
        self.win = win
        self.vol_span = vol_span
        self.target_vol = target_vol
        self.cap = cap
        self.vol_floor = vol_floor
        self.risk = RiskConfig(max_position=cap, max_gross=cap)

    def __repr__(self):
        return (f"IndexReversal(L={self.lookback}, win={self.win}, volspan={self.vol_span}, "
                f"tgt={self.target_vol}, cap={self.cap}, floor={self.vol_floor})")

    def generate_weights(self, data, t):
        px = data.prices["US500"]
        if len(px) < self.lookback + self.win + 2:
            return pd.Series(0.0, index=data.assets)
        lret = np.log(px).diff(self.lookback)               # L-day log returns (PIT, up to t)
        recent = lret.iloc[-self.win:]
        mu, sd = recent.mean(), recent.std()
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(0.0, index=data.assets)
        z = (lret.iloc[-1] - mu) / sd
        sig = float(np.clip(-z, -3.0, 3.0) / 3.0)           # reversal signal in [-1,1]
        vol = float(px.pct_change().iloc[-self.vol_span:].std() * np.sqrt(252))
        vol = max(vol, self.vol_floor)
        w = float(np.clip(sig * (self.target_vol / vol), -self.cap, self.cap))
        return pd.Series({"US500": w}).reindex(data.assets).fillna(0.0)


strat = IndexReversal()
print(strat)
rb = strat.rebalance_dates(panel.dates)
print(f"Rebalances: {len(rb)} (daily) | {rb[0].date()} -> {rb[-1].date()}")

from backtest.data import DataView
print("\ndry-run generate_weights (incl. crisis dates -> position should be bounded/shrunk):")
for ts in ["2014-06-16", "2008-10-20", "2020-03-16", "2025-04-10", "2026-05-29"]:
    view = panel.as_of(pd.Timestamp(ts))
    d = view.t
    px = view.prices["US500"]
    lret = np.log(px).diff(5)
    z = (lret.iloc[-1] - lret.iloc[-120:].mean()) / lret.iloc[-120:].std()
    vol = px.pct_change().iloc[-20:].std() * np.sqrt(252)
    w = strat.generate_weights(view, d)["US500"]
    print(f"  {d.date()}: 5d_ret {lret.iloc[-1]*100:+5.1f}% | z {z:+5.2f} | annvol {vol*100:4.0f}% | w {w:+.3f}")

# Cell 8 — Ranked + clipped reversal strategy (trial #002)
import pandas as pd
from backtest.risk import RiskConfig

class ReversalRanked(ShortTermReversal):
    """Reversal on cross-sectional RANK of clipped k-day return. Robust to data
    artifacts: daily returns winsorized to +-clip before compounding, then ranked."""
    risk = RiskConfig(max_position=0.03, max_gross=1.0, max_net=1.0)

    def __init__(self, lookback: int = 5, clip: float = 0.5):
        self.lookback = lookback
        self.clip = clip

    def __repr__(self):
        return f"ReversalRanked(lookback={self.lookback}, clip={self.clip})"

    def generate_weights(self, data, t):
        k = self.lookback
        px = data.prices
        if len(px) < k + 1:
            return pd.Series(0.0, index=data.assets)
        cols = data.assets
        dr = px[cols].pct_change().iloc[-k:].clip(-self.clip, self.clip)  # clipped daily rets
        valid = dr.notna().all()                                          # full k-window only
        ret = ((1.0 + dr).prod() - 1.0)[valid]
        if len(ret) < 2:
            return pd.Series(0.0, index=data.assets)
        r = ret.rank()                          # cross-sectional rank (1..N)
        sig = -(r - r.mean())                    # reversal: losers long, winners short
        gross = sig.abs().sum()
        if gross == 0:
            return pd.Series(0.0, index=data.assets)
        return sig / gross

strat_rank = ReversalRanked(lookback=5, clip=0.5)
print(strat_rank)
# sanity: weights at the last date (pre-engine, pre-risk-cap)
v = panel_mid.as_of(panel_mid.dates[-1])
w = strat_rank.generate_weights(v, panel_mid.dates[-1])
print(f"nonzero names: {(w!=0).sum()}  gross: {w.abs().sum():.3f}  net: {w.sum():+.3e}  max|w|: {w.abs().max():.3%}")

# Cell 2 — Strategy class (short-term cross-sectional reversal)
import pandas as pd
from backtest.strategy import Strategy

class ShortTermReversal(Strategy):
    """Daily dollar-neutral short-term reversal: short recent winners, long recent
    losers on trailing k-day return, demeaned across the cross-section."""
    rebalance_frequency = "daily"

    def __init__(self, lookback: int = 5):
        self.lookback = lookback

    def __repr__(self):
        return f"ShortTermReversal(lookback={self.lookback})"

    def required_data(self):
        return {"prices": None}

    def generate_weights(self, data, t):
        k = self.lookback
        px = data.prices                       # loc[:t] inclusive — no lookahead
        if len(px) < k + 1:
            return pd.Series(0.0, index=data.assets)
        cols = data.assets                      # eligible names at t
        ret = px[cols].iloc[-1] / px[cols].iloc[-1 - k] - 1.0
        ret = ret.dropna()
        if len(ret) < 2:
            return pd.Series(0.0, index=data.assets)
        sig = -ret                              # reversal
        sig = sig - sig.mean()                  # cross-sectional demean -> dollar-neutral
        gross = sig.abs().sum()
        if gross == 0:
            return pd.Series(0.0, index=data.assets)
        return sig / gross                      # gross exposure = 1.0

strat = ShortTermReversal(lookback=5)
print(strat)
rb = strat.rebalance_dates(panel.dates)
print(f"Rebalances: {len(rb)}  (~{len(rb) / (len(panel.dates) / 252):.0f}/yr)")

# Cell 12 — Decay + ranked + clipped reversal (trial #003)
import pandas as pd

class ReversalRankedDecay(ReversalRanked):
    """Ranked reversal with linear signal decay over `decay` days to cut turnover.
    Stateless: recomputes the ranked signal for each lag, weights it (decay-j)."""
    def __init__(self, lookback: int = 5, clip: float = 0.5, decay: int = 5):
        super().__init__(lookback, clip)
        self.decay = decay

    def __repr__(self):
        return f"ReversalRankedDecay(lookback={self.lookback}, clip={self.clip}, decay={self.decay})"

    def _ranked_sig(self, px, cols):
        k = self.lookback
        if len(px) < k + 1:
            return None
        dr = px[cols].pct_change().iloc[-k:].clip(-self.clip, self.clip)
        ret = ((1.0 + dr).prod() - 1.0)[dr.notna().all()]
        if len(ret) < 2:
            return None
        r = ret.rank()
        sig = -(r - r.mean())
        g = sig.abs().sum()
        return sig / g if g > 0 else None

    def generate_weights(self, data, t):
        px, cols = data.prices, data.assets
        acc = None
        for j in range(self.decay):
            sub = px.iloc[:len(px) - j] if j > 0 else px
            s = self._ranked_sig(sub, cols)
            if s is None:
                continue
            w = (self.decay - j) * s
            acc = w if acc is None else acc.add(w, fill_value=0.0)
        if acc is None:
            return pd.Series(0.0, index=data.assets)
        g = acc.abs().sum()
        return acc / g if g > 0 else pd.Series(0.0, index=data.assets)

strat_decay = ReversalRankedDecay(lookback=5, clip=0.5, decay=5)
print(strat_decay)

# turnover proxy: L1 weight change over one day, decay vs non-decay
v0, v1 = panel_mid.as_of(panel_mid.dates[-1]), panel_mid.as_of(panel_mid.dates[-2])
def l1(strat):
    a = strat.generate_weights(v0, panel_mid.dates[-1])
    b = strat.generate_weights(v1, panel_mid.dates[-2])
    return (a.subtract(b, fill_value=0.0)).abs().sum()
print(f"1-day L1 turnover  ranked(no decay): {l1(strat_rank):.3f}")
print(f"1-day L1 turnover  decay=5:          {l1(strat_decay):.3f}")

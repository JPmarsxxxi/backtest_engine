# Cell 16 — Sector-neutral ranked+decay reversal (trial #004); pct_change fix + neutralize flag
import pandas as pd

class ReversalNeutral(ReversalRankedDecay):
    """Adds within-sector demeaning (industry neutralization) to the decayed ranked
    signal. neutralize=False reproduces the fixed-decay baseline for clean attribution."""
    def __init__(self, lookback=5, clip=0.5, decay=5, sectors=None, neutralize=True):
        super().__init__(lookback, clip, decay)
        self.sectors = sectors or {}
        self.neutralize = neutralize

    def __repr__(self):
        return (f"ReversalNeutral(lookback={self.lookback}, clip={self.clip}, "
                f"decay={self.decay}, neutralize={self.neutralize})")

    def _ranked_sig(self, px, cols):
        k = self.lookback
        if len(px) < k + 1:
            return None
        dr = px[cols].pct_change(fill_method=None).iloc[-k:].clip(-self.clip, self.clip)
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
        if self.neutralize:
            sec = pd.Series({n: self.sectors.get(n, "UNK") for n in acc.index})
            acc = acc - acc.groupby(sec).transform("mean")   # within-sector demean
        g = acc.abs().sum()
        return acc / g if g > 0 else pd.Series(0.0, index=data.assets)

strat_decay_fix = ReversalNeutral(sectors=sector_map, neutralize=False)  # fixed baseline
strat_neutral   = ReversalNeutral(sectors=sector_map, neutralize=True)   # +sector neutral
print(strat_neutral)

# sanity: per-sector net exposure at last date, neutral vs not
v = panel_mid.as_of(panel_mid.dates[-1]); d = panel_mid.dates[-1]
sec = pd.Series({n: sector_map.get(n, "UNK") for n in panel_mid.assets_all})
for name, st in [("baseline", strat_decay_fix), ("neutral", strat_neutral)]:
    w = st.generate_weights(v, d)
    by_sec = w.groupby(sec.reindex(w.index)).sum()
    print(f"{name:>8}: max |sector net| = {by_sec.abs().max():.4f}  gross={w.abs().sum():.3f}")

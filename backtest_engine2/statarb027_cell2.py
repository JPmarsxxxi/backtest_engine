# Cell 2 — Strategy class: discrete PCA-residual s-score stat-arb (#027)
import numpy as np
import pandas as pd
from backtest.strategy import Strategy

_GAP = pd.Timedelta("20min")   # consecutive-bar gap threshold (mask weekend/holiday returns)


class StatArbResidual(Strategy):
    """Intraday cross-sectional stat-arb. Each bar: strip top-m PCA factors from standardized
    returns -> idiosyncratic residual; OU s-score on the trailing-L cumulative residual;
    DISCRETE bands per pair (open |s|>s_in, hold, close |s|<s_out), fixed 1/N sizing.
    Market-neutral, intraday (no overnight). PCA re-estimated once per calendar day (cached on self)."""

    rebalance_frequency = "daily"          # = every bar in `dates` (base.py:67-68); here every M15 bar

    def __init__(self, pca_win=288, m_factors=3, L=32, vol_win=96, s_in=2.0, s_out=0.5):
        self.pca_win = pca_win             # PCA estimation window (bars; 288 = 3 trading days)
        self.m_factors = m_factors         # common factors removed
        self.L = L                         # s-score / OU window (bars; 32*15min = 8h)
        self.vol_win = vol_win             # return-standardization window (bars; 96 = 1 day)
        self.s_in = s_in                   # entry threshold
        self.s_out = s_out                 # exit threshold
        # state (persists across generate_weights within one run; per §11)
        self._pos = None                   # current discrete position per asset in {-1,0,+1}
        self._day = None                   # calendar day of the cached PCA loadings
        self._V = None                     # cached top-m eigenvectors

    def __repr__(self):
        return (f"StatArbResidual(pca_win={self.pca_win},m={self.m_factors},L={self.L},"
                f"vol_win={self.vol_win},s_in={self.s_in},s_out={self.s_out})")

    def apply_risk(self, proposed, state, data):
        return proposed                    # identity — faithful signal replay (risk = optional Stage 5)

    def _sscore(self, px, t):
        """PIT s-score per pair at t, from prices up to t. Returns pd.Series or None if warming up."""
        need = self.pca_win + self.vol_win + 10
        if len(px) < need:
            return None
        px = px.iloc[-need:]
        rets = np.log(px).diff()
        rets[(px.index.to_series().diff() > _GAP).values] = np.nan       # mask gap returns
        vol = rets.rolling(self.vol_win, min_periods=self.vol_win // 2).std()
        z = (rets / vol).clip(-8, 8)
        # re-estimate PCA loadings once per calendar day, cache on self
        day = t.normalize()
        if self._day != day or self._V is None:
            Zw = z.iloc[-self.pca_win:].dropna()
            if len(Zw) < len(px.columns) * 5:
                return None
            C = np.nan_to_num(np.corrcoef(Zw.values.T))
            evals, evecs = np.linalg.eigh(C)
            self._V = evecs[:, np.argsort(evals)[::-1][:self.m_factors]]
            self._day = day
        P = self._V @ self._V.T                                          # factor projection
        Z = z.iloc[-2 * self.L:].fillna(0.0).values                      # last 2L bars
        R = Z - Z @ P.T                                                   # factor-neutral residual
        X = pd.DataFrame(R, columns=px.columns).rolling(self.L).sum().iloc[-self.L:]
        s = -(X.iloc[-1] - X.mean()) / X.std().replace(0, np.nan)        # OU s-score at t
        return s

    def generate_weights(self, data, t):
        px = data.prices
        assets = data.assets
        if self._pos is None:
            self._pos = pd.Series(0.0, index=px.columns)
        s = self._sscore(px, t)
        if s is None:
            return pd.Series(0.0, index=assets)                          # warmup -> flat
        # discrete state machine per pair (long when residual cheap, i.e. s > s_in)
        for a in px.columns:
            v = s.get(a, np.nan)
            if not np.isfinite(v):
                continue                                                 # hold through missing signal
            p = self._pos[a]
            if p == 0.0:
                if v > self.s_in:      self._pos[a] = 1.0
                elif v < -self.s_in:   self._pos[a] = -1.0
            elif p == 1.0 and v < self.s_out:      self._pos[a] = 0.0
            elif p == -1.0 and v > -self.s_out:    self._pos[a] = 0.0
        N = len(px.columns)                                              # fixed 1/N sizing per pair
        return (self._pos / N).reindex(assets).fillna(0.0)


strat = StatArbResidual()
print(strat)
rb = strat.rebalance_dates(panel.dates)
print(f"rebalance bars: {len(rb):,} of {len(panel.dates):,} panel bars  (every-bar: {len(rb) == len(panel.dates)})")
print(f"approx rebalances/year: {len(rb) / (len(panel.dates) / (96*252)):,.0f}  (M15: 96*252 bars/yr)")
print(f"warmup bars before first signal: ~{strat.pca_win + strat.vol_win + 10}")

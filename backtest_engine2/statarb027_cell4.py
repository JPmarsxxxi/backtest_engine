# Cell 4 — Strategy revision: add the spread gate (open only) (#027)
import numpy as np
import pandas as pd
from backtest.strategy import Strategy

_GAP = pd.Timedelta("20min")


class StatArbResidual(Strategy):
    """Discrete PCA-residual s-score stat-arb with a SPREAD GATE: open a position only when the
    pair's half-spread at t is <= spread_cap; closes are never gated. Market-neutral, intraday.
    PCA re-estimated once per calendar day (cached on self)."""

    rebalance_frequency = "daily"          # = every bar (base.py:67-68) -> every M15 bar

    def __init__(self, pca_win=288, m_factors=3, L=32, vol_win=96, s_in=2.0, s_out=0.5, spread_cap=0.20):
        self.pca_win = pca_win
        self.m_factors = m_factors
        self.L = L
        self.vol_win = vol_win
        self.s_in = s_in
        self.s_out = s_out
        self.spread_cap = spread_cap       # bp; skip an OPEN if half-spread(t) > cap
        self._pos = None
        self._day = None
        self._V = None

    def __repr__(self):
        return (f"StatArbResidual(pca_win={self.pca_win},m={self.m_factors},L={self.L},"
                f"vol_win={self.vol_win},s_in={self.s_in},s_out={self.s_out},spread_cap={self.spread_cap})")

    def required_data(self):
        return {"prices": None, "spread": None}

    def apply_risk(self, proposed, state, data):
        return proposed                    # identity — risk = optional Stage 5

    def _sscore(self, px, t):
        need = self.pca_win + self.vol_win + 10
        if len(px) < need:
            return None
        px = px.iloc[-need:]
        rets = np.log(px).diff()
        rets[(px.index.to_series().diff() > _GAP).values] = np.nan
        vol = rets.rolling(self.vol_win, min_periods=self.vol_win // 2).std()
        z = (rets / vol).clip(-8, 8)
        day = t.normalize()
        if self._day != day or self._V is None:
            Zw = z.iloc[-self.pca_win:].dropna()
            if len(Zw) < len(px.columns) * 5:
                return None
            C = np.nan_to_num(np.corrcoef(Zw.values.T))
            evals, evecs = np.linalg.eigh(C)
            self._V = evecs[:, np.argsort(evals)[::-1][:self.m_factors]]
            self._day = day
        P = self._V @ self._V.T
        Z = z.iloc[-2 * self.L:].fillna(0.0).values
        R = Z - Z @ P.T
        X = pd.DataFrame(R, columns=px.columns).rolling(self.L).sum().iloc[-self.L:]
        return -(X.iloc[-1] - X.mean()) / X.std().replace(0, np.nan)

    def generate_weights(self, data, t):
        px = data.prices
        assets = data.assets
        if self._pos is None:
            self._pos = pd.Series(0.0, index=px.columns)
        s = self._sscore(px, t)
        if s is None:
            return pd.Series(0.0, index=assets)
        sp = data.feature("spread").iloc[-1]              # per-pair half-spread(bp) at t
        for a in px.columns:
            v = s.get(a, np.nan)
            if not np.isfinite(v):
                continue                                  # hold through missing signal
            p = self._pos[a]
            if p == 0.0:                                  # OPEN — spread-gated
                spread_a = sp.get(a, np.nan)
                if np.isfinite(spread_a) and spread_a <= self.spread_cap:
                    if v > self.s_in:      self._pos[a] = 1.0
                    elif v < -self.s_in:   self._pos[a] = -1.0
            elif p == 1.0 and v < self.s_out:      self._pos[a] = 0.0    # CLOSE — never gated
            elif p == -1.0 and v > -self.s_out:    self._pos[a] = 0.0
        N = len(px.columns)
        return (self._pos / N).reindex(assets).fillna(0.0)


strat = StatArbResidual(spread_cap=0.20)
print(strat)
print("\ngate pass-rate (frac of bars with half-spread <= cap), by pair:")
print((half_bp <= strat.spread_cap).mean().sort_values(ascending=False).round(3).to_string())

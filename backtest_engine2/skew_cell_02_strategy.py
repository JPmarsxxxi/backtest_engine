# Cell 2 — SkewLottery strategy: short lottery-shaped (high 21d skew), long steady
import pandas as pd
from backtest.strategy import Strategy
from backtest.risk import RiskConfig

class SkewLottery(Strategy):
    """Cross-sectional lottery-asymmetry alpha. Signal = -rank(skew of daily returns,
    trailing `window` days), demeaned, linear-decayed over `decay` days, gross 1.0.
    Hypothesis: skew-seeking buyers overpay for lottery-like payoffs -> high skew underperforms."""
    rebalance_frequency = "daily"
    risk = RiskConfig(max_position=0.15, max_gross=1.0, max_net=1.0)

    def __init__(self, window: int = 21, decay: int = 5):
        self.window = window
        self.decay = decay

    def __repr__(self):
        return f"SkewLottery(window={self.window}, decay={self.decay})"

    def generate_weights(self, data, t):
        w, d = self.window, self.decay
        px = data.prices
        if len(px) < w + d + 1:
            return pd.Series(0.0, index=data.assets)
        dr = px[data.assets].pct_change(fill_method=None).iloc[-(w + d - 1):]
        sk = dr.rolling(w, min_periods=w).skew()                # full window only
        num = pd.Series(0.0, index=dr.columns)
        den = pd.Series(0.0, index=dr.columns)
        for k in range(d):                                      # k=0 newest, weight d-k
            row = sk.iloc[-(k + 1)].dropna()
            if len(row) < 5:
                continue
            r = row.rank()
            s = -(r - r.mean())                                 # short high skew, long low
            wt = float(d - k)
            num = num.add(s * wt, fill_value=0.0)
            den = den.add(pd.Series(wt, index=s.index), fill_value=0.0)
        valid = den > 0
        if valid.sum() < 5:
            return pd.Series(0.0, index=data.assets)
        sig = num[valid] / den[valid]
        sig = sig - sig.mean()                                  # re-demean after decay
        gross = sig.abs().sum()
        if gross == 0:
            return pd.Series(0.0, index=data.assets)
        return sig / gross

strat_skew = SkewLottery(window=21, decay=5)
print(strat_skew)
rb = strat_skew.rebalance_dates(panel_skew.dates)
yrs = (panel_skew.dates[-1] - panel_skew.dates[0]).days / 365.25
print(f"Rebalances/year: {len(rb) / yrs:.0f}")
v = panel_skew.as_of(panel_skew.dates[-1])
wts = strat_skew.generate_weights(v, panel_skew.dates[-1])
print(f"Last bar: nonzero={int((wts != 0).sum())}  gross={wts.abs().sum():.3f}  "
      f"net={wts.sum():+.3e}  max|w|={wts.abs().max():.2%}")

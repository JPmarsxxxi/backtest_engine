# Cell 7 — #006 replica (30d XS momentum) on FTMO-14, correlate vs flipped skew (Step 6)
import pandas as pd

class Momentum30(SkewLottery):
    """#006's final construction: +rank(clipped 30d return), demeaned, decay 5, gross 1.0."""
    def __init__(self, lookback: int = 30, clip: float = 1.0, decay: int = 5):
        self.lookback = lookback
        self.clip = clip
        self.decay = decay

    def __repr__(self):
        return f"Momentum30(lookback={self.lookback}, clip={self.clip}, decay={self.decay})"

    def generate_weights(self, data, t):
        k, d = self.lookback, self.decay
        px = data.prices
        if len(px) < k + d + 1:
            return pd.Series(0.0, index=data.assets)
        dr = px[data.assets].pct_change(fill_method=None).clip(-self.clip, self.clip)
        num = pd.Series(0.0, index=dr.columns)
        den = pd.Series(0.0, index=dr.columns)
        for j in range(d):                                   # j=0 newest, weight d-j
            win = dr.iloc[len(dr) - k - j: len(dr) - j]
            valid = win.notna().all()
            ret = ((1.0 + win).prod() - 1.0)[valid]
            if len(ret) < 5:
                continue
            r = ret.rank()
            s = r - r.mean()                                 # momentum: winners long
            wt = float(d - j)
            num = num.add(s * wt, fill_value=0.0)
            den = den.add(pd.Series(wt, index=s.index), fill_value=0.0)
        ok = den > 0
        if ok.sum() < 5:
            return pd.Series(0.0, index=data.assets)
        sig = num[ok] / den[ok]
        sig = sig - sig.mean()
        gross = sig.abs().sum()
        if gross == 0:
            return pd.Series(0.0, index=data.assets)
        return sig / gross

strat_m30 = Momentum30()
result_m30 = Engine(panel_skew, strat_m30, costs=costs_ftmo, liquidity=liq_ftmo,
                    initial_capital=1_000_000).run(start=panel_skew.dates[40])  # 30+5 warmup

rep_m30 = compute_metrics(result_m30.returns, result_m30.equity_curve,
                          costs=result_m30.costs, trades=result_m30.trades, ann_factor=365)
yrs_m30 = (result_m30.equity_curve.index[-1] - result_m30.equity_curve.index[0]).days / 365.25
gS_m30 = sharpe_ratio(result_m30.gross_pnl / result_m30.equity_curve.shift(1), ann_factor=365)
print(strat_m30)
print(f"#006-on-FTMO14: GROSS Sh {gS_m30:+.3f} | NET Sh {rep_m30.sharpe:+.3f} | "
      f"turnover {rep_m30.turnover/yrs_m30:.0f}x/yr | maxDD {rep_m30.max_drawdown:.0%}")
print()
both = pd.concat([result_flip.returns.rename("skew_flip"),
                  result_m30.returns.rename("mom30")], axis=1).dropna()
corr = both["skew_flip"].corr(both["mom30"])
print(f"Overlap bars: {len(both)}")
print(f"Pearson corr(flip skew, mom30): {corr:+.3f}")
print("Verdict:", "REDUNDANT (>0.7)" if corr > 0.7 else
      "ok (0.3-0.5)" if corr > 0.3 else "GOOD diversifier (<0.3)" if corr < 0.3 else "borderline")

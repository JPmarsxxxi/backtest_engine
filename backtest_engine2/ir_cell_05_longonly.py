# Cell 5 — #019 long-only variant: buy-the-dip only (drop the fade-rally SHORT leg -> flat on rallies).
# Tests whether the recent decay (2014-2026) lived in the short side (fighting drift/momentum).
from backtest.engine import Engine


class IndexReversalLong(IndexReversal):
    def __repr__(self):
        return (f"IndexReversalLong(L={self.lookback}, win={self.win}, volspan={self.vol_span}, "
                f"tgt={self.target_vol}, cap={self.cap})")

    def generate_weights(self, data, t):
        return super().generate_weights(data, t).clip(lower=0.0)   # long-only: short -> flat


strat_lo = IndexReversalLong()
res = Engine(panel, strat_lo, costs=costs, liquidity=liq, initial_capital=1_000_000).run(start=panel.dates[130])
eq, ret = res.equity_curve, res.returns.dropna()
yrs = (eq.index[-1] - eq.index[0]).days / 365.25
ppy = len(ret) / yrs
annf = np.sqrt(ppy)
ret_g = (res.gross_pnl / eq.shift(1)).dropna()
def sh(r):
    return r.mean() / r.std() * annf if r.std() > 0 else float("nan")

print(f"LONG-ONLY | net Sh {sh(ret):+.2f} (gross {sh(ret_g):+.2f}) | ann vol {ret.std()*annf:.1%} | "
      f"maxDD {(eq/eq.cummax()-1).min():.1%} | worst {ret.min():+.2%} | "
      f"avg gross {res.weights.abs().sum(axis=1).mean():.2f} | in-mkt {(res.weights.abs().sum(axis=1)>1e-6).mean():.0%}")
for lab, m in [("full 2000-26", ret.index.year >= 2000),
               ("2014-2026", ret.index.year >= 2014),
               ("2022-2025", (ret.index.year >= 2022) & (ret.index.year <= 2025))]:
    r = ret[m]
    pos = sum(1 for y in set(r.index.year) if r[r.index.year == y].mean() > 0)
    print(f"  {lab:>12}: net Sh {sh(r):+.2f} | ann ret {r.mean()*ppy*100:+.1f}% | pos yrs {pos}/{r.index.year.nunique()}")
print(f"\n{'yr':>5} {'net ret':>8} {'netSh':>6} {'worst d':>8}")
for yr, g in ret.groupby(ret.index.year):
    e = (1 + g).cumprod()
    print(f"{yr:>5} {e.iloc[-1]-1:>+8.1%} {g.mean()/g.std()*annf if g.std()>0 else float('nan'):>+6.2f} {g.min():>+8.2%}")

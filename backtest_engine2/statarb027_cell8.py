# Cell 8 — TRIAL 2 (best-net shot): high conviction + tight gate + swap-free (#027)
from backtest.engine import Engine

strat2 = StatArbResidual(s_in=3.0, s_out=0.75, spread_cap=0.18)
res2 = Engine(panel, strat2, costs=costs, initial_capital=1_000_000).run(start=panel.dates[500])

r = res2.returns.dropna()
gross_r = (res2.gross_pnl / res2.equity_curve.shift(1)).dropna()
ann = (96 * 252) ** 0.5
def sh(x): return x.mean() / x.std() * ann if x.std() > 0 else 0.0

print(strat2)
print(f"NET   total {res2.equity_curve.iloc[-1]/1e6 - 1:+.2%}  ann {r.mean()*96*252:+.2%}  Sharpe {sh(r):+.2f}")
print(f"GROSS total {res2.gross_pnl.sum()/1e6:+.2%}  ann {gross_r.mean()*96*252:+.2%}  Sharpe {sh(gross_r):+.2f}")
print(f"costs {res2.costs.sum()/1e6:.2%}  |  max gross exposure {res2.weights.abs().sum(axis=1).max():.2%}")
yr = r.groupby(r.index.year).apply(lambda x: sh(x))
print("per-year NET Sharpe:", {int(y): round(float(v), 2) for y, v in yr.items()})
print("DONE")

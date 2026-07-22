# Cell 5 — Risk overlay: 3% per-name cap (new trial), compared vs raw
from backtest.risk import RiskConfig

class ShortTermReversalCapped(ShortTermReversal):
    risk = RiskConfig(max_position=0.03, max_gross=1.0, max_net=1.0)
    def __repr__(self):
        return f"ShortTermReversalCapped(lookback={self.lookback}, max_position=0.03)"

strat_cap = ShortTermReversalCapped(lookback=5)
result_cap = Engine(panel, strat_cap, costs=costs, liquidity=liq,
                    initial_capital=1_000_000).run(start=panel.dates[WARMUP_BARS])

def line(name, r):
    eq0 = r.metadata["initial_capital"]
    print(f"{name:<8} | maxpos {r.weights.abs().max().max():>6.2%} | "
          f"gross {r.weights.abs().sum(axis=1).max():>6.2%} | "
          f"mean|w| {r.weights.abs().replace(0,float('nan')).mean().mean():>5.2%} | "
          f"net {(r.equity_curve.iloc[-1]/eq0-1):>+7.2%} | "
          f"gross {(r.gross_pnl.sum()/eq0):>+7.2%} | "
          f"costs ${r.costs.sum():>9,.0f}")

print("trial    | max/name | gross-exp | avg pos | NET ret | GROSS ret | costs")
line("raw20%", result)
line("cap3%",  result_cap)

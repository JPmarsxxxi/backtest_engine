# Cell 6 — #014 trial 2: net-of-financing trend (swap inside the signal, zero new params).
# edge_long = r252 + 1y long swap; edge_short = -r252 + 1y short swap.
# Long if edge_long > max(0, edge_short); short if edge_short > max(0, edge_long); else FLAT
# (the trailing trend doesn't cover its own financing). Sizing identical to t1: 0.4/sigma,
# split by data-alive count (not position count), so per-position size matches t1 and the
# book simply drops unprofitable holdings -> lower gross, same per-name risk.
class TSMOMNetCarry(Strategy):
    """Trend signal on net-of-swap trailing return; trade only self-financing sides."""
    rebalance_frequency = staticmethod(_month_end_bars)

    def __init__(self, long_rate: dict, short_rate: dict):
        self.rl = pd.Series(long_rate)
        self.rs = pd.Series(short_rate)
        self.risk = RiskConfig(max_position=0.25, max_gross=6.0)

    def __repr__(self):
        return "TSMOMNetCarry(sign 252d net of per-side swap, 0.40/sigma, monthly)"

    def generate_weights(self, data, t):
        px = data.prices
        if len(px) < LOOKBACK + 1:
            return pd.Series(0.0, index=data.assets)
        r252 = px.iloc[-1] / px.iloc[-(LOOKBACK + 1)] - 1.0
        el = r252 + self.rl.reindex(r252.index)
        es = -r252 + self.rs.reindex(r252.index)
        sig = pd.Series(0.0, index=r252.index)
        sig[(el > 0) & (el >= es)] = 1.0
        sig[(es > 0) & (es > el)] = -1.0
        vol = (px.pct_change(fill_method=None).ewm(span=VOL_SPAN).std().iloc[-1]
               * np.sqrt(252)).clip(lower=VOL_FLOOR)
        n_alive = int(r252.notna().sum())
        if n_alive == 0:
            return pd.Series(0.0, index=data.assets)
        w = (sig * SCALE / vol / n_alive).fillna(0.0)
        return w.reindex(data.assets).fillna(0.0)


strat2 = TSMOMNetCarry(swp_l, swp_s)
print(strat2)
engine2 = Engine(panel, strat2, costs=costs, liquidity=None, initial_capital=1_000_000)
result2 = engine2.run(start=START)

eq2 = result2.equity_curve
ret2 = result2.returns.dropna()
sh2 = ret2.mean() / ret2.std() * np.sqrt(252)
dd2 = (eq2 / eq2.cummax() - 1).min()
pos2 = result2.positions.fillna(0.0)
swap2 = (-(pos2.clip(lower=0) * pd.Series(swp_l).reindex(pos2.columns)
           + pos2.clip(upper=0).abs() * pd.Series(swp_s).reindex(pos2.columns)) / 252).sum(axis=1)
print(f"Final equity ${eq2.iloc[-1]:,.0f} ({eq2.iloc[-1] / 1e6 - 1:+.1%}) | "
      f"net Sharpe {sh2:+.2f} | vol {ret2.std() * np.sqrt(252):.1%} | maxDD {dd2:.1%}")
print(f"Costs ${result2.costs.sum():,.0f} (swap ${swap2.sum():,.0f}) | "
      f"swap drag {252 * (swap2 / eq2.shift(1)).mean():.2%}/yr | "
      f"avg gross {result2.weights.abs().sum(axis=1).mean():.2f} (t1: 4.5)")

cost_frac2 = (result2.costs / eq2.shift(1)).reindex(ret2.index).fillna(0.0)
gross2 = ret2 + cost_frac2
print(f"GROSS Sharpe {gross2.mean() / gross2.std() * np.sqrt(252):+.2f} (t1 gross +0.31)")
print(f"\n{'year':>5} {'net':>8} {'netSh':>6} | {'t1 net':>8}")
t1net = result.returns.dropna()
for yr, g in ret2.groupby(ret2.index.year):
    n1 = t1net[t1net.index.year == yr]
    print(f"{yr:>5} {(1 + g).prod() - 1:>+8.1%} "
          f"{g.mean() / g.std() * np.sqrt(252) if g.std() > 0 else float('nan'):>+6.2f} | "
          f"{(1 + n1).prod() - 1:>+8.1%}")

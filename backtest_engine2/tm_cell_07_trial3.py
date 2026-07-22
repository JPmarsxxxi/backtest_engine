# Cell 7 — #014 trial 3 (originally pre-registered): lookback-blend TSMOM.
# signal = sign(mean[sign(63d), sign(126d), sign(252d)]) per instrument; same 0.4/sigma
# sizing, same real FTMO costs. Robustness check on signal speed (does faster trend help?).
LBS = [63, 126, 252]


class TSMOMBlend(Strategy):
    rebalance_frequency = staticmethod(_month_end_bars)

    def __init__(self):
        self.risk = RiskConfig(max_position=0.25, max_gross=6.0)

    def __repr__(self):
        return "TSMOMBlend(mean sign[63,126,252], 0.40/sigma, monthly)"

    def generate_weights(self, data, t):
        px = data.prices
        if len(px) < max(LBS) + 1:
            return pd.Series(0.0, index=data.assets)
        sig = np.sign(sum(np.sign(px.iloc[-1] / px.iloc[-(lb + 1)] - 1.0) for lb in LBS))
        vol = (px.pct_change(fill_method=None).ewm(span=VOL_SPAN).std().iloc[-1]
               * np.sqrt(252)).clip(lower=VOL_FLOOR)
        n_alive = int((px.iloc[-1] / px.iloc[-(max(LBS) + 1)]).notna().sum())
        if n_alive == 0:
            return pd.Series(0.0, index=data.assets)
        return (sig * SCALE / vol / n_alive).fillna(0.0).reindex(data.assets).fillna(0.0)


strat3 = TSMOMBlend()
print(strat3)
engine3 = Engine(panel, strat3, costs=costs, liquidity=None, initial_capital=1_000_000)
result3 = engine3.run(start=START)
eq3 = result3.equity_curve
ret3 = result3.returns.dropna()
cost_frac3 = (result3.costs / eq3.shift(1)).reindex(ret3.index).fillna(0.0)
gross3 = ret3 + cost_frac3
print(f"Final ${eq3.iloc[-1]:,.0f} ({eq3.iloc[-1] / 1e6 - 1:+.1%}) | "
      f"net Sharpe {ret3.mean() / ret3.std() * np.sqrt(252):+.2f} | "
      f"gross Sharpe {gross3.mean() / gross3.std() * np.sqrt(252):+.2f} | "
      f"maxDD {(eq3 / eq3.cummax() - 1).min():.1%} | costs ${result3.costs.sum():,.0f}")

print("\nSession trial summary (net / gross Sharpe):")
print(f"  t1 MOP-exact 252d:      net -0.34 | gross +0.31")
print(f"  t2 net-of-financing:    net -0.17 | gross +0.17")
print(f"  t3 blend[63,126,252]:   net {ret3.mean() / ret3.std() * np.sqrt(252):+.2f} | "
      f"gross {gross3.mean() / gross3.std() * np.sqrt(252):+.2f}")

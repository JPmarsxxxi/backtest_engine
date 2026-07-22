# Cell 5 — #014 diagnostic: gross/cost decomposition + 2022 autopsy.
# Gross = net + costs (per bar, as fraction of equity). Swap recomputed analytically
# from positions x per-side rates; spread = |trades| x half_bp (residual check vs engine).
eq = result.equity_curve
ret_net = result.returns.dropna()
cost_frac = (result.costs / eq.shift(1)).reindex(ret_net.index).fillna(0.0)
ret_gross = ret_net + cost_frac

# Analytic cost split.
pos = result.positions.fillna(0.0)
trades = pos.diff().fillna(pos.iloc[0])
hb = pd.Series(half_bp)
spread_d = (trades.abs() * hb.reindex(trades.columns) / 1e4).sum(axis=1)
rl, rs = pd.Series(swp_l), pd.Series(swp_s)
swap_d = (-(pos.clip(lower=0) * rl.reindex(pos.columns)
            + pos.clip(upper=0).abs() * rs.reindex(pos.columns)) / 252).sum(axis=1)
print(f"Engine costs ${result.costs.sum():,.0f} | analytic: spread ${spread_d.sum():,.0f} "
      f"+ swap ${swap_d.sum():,.0f} = ${spread_d.sum() + swap_d.sum():,.0f}")
gsh = ret_gross.mean() / ret_gross.std() * np.sqrt(252)
print(f"GROSS Sharpe {gsh:+.2f} (net {ret_net.mean() / ret_net.std() * np.sqrt(252):+.2f}) | "
      f"swap drag {252 * (swap_d / eq.shift(1)).mean():.2%}/yr | "
      f"spread drag {252 * (spread_d / eq.shift(1)).mean():.2%}/yr")

print(f"\n{'year':>5} {'gross':>8} {'gSh':>6} {'net':>8} {'swap$':>10} {'spread$':>9}")
for yr, g in ret_gross.groupby(ret_gross.index.year):
    n = ret_net[ret_net.index.year == yr]
    print(f"{yr:>5} {(1 + g).prod() - 1:>+8.1%} "
          f"{g.mean() / g.std() * np.sqrt(252) if g.std() > 0 else float('nan'):>+6.2f} "
          f"{(1 + n).prod() - 1:>+8.1%} {swap_d[swap_d.index.year == yr].sum():>10,.0f} "
          f"{spread_d[spread_d.index.year == yr].sum():>9,.0f}")

# 2022 autopsy: month-end book vs what trend "should" have held.
KEY = ["EURUSD", "USDJPY", "GBPUSD", "US500", "US100", "GER40", "USOIL", "UKOIL", "XAUUSD"]
w22 = result.weights.loc["2022", KEY]
me = w22.groupby([w22.index.year, w22.index.month]).last()
print("\n2022 month-end weights (sign matters: trend wanted short EUR/GBP, long USDJPY, "
      "short indices, long oil):")
print((me * 100).round(1).to_string())

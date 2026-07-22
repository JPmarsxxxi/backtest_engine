# Cell 28 — 30d momentum on FTMO-13 vs full 59-coin universe (same strat/costs/warmup)
strat_ftmo  = MomentumRanked(lookback=30, clip=1.0, decay=5, neutralize=False)
result_ftmo = Engine(panel_crypto_ftmo, strat_ftmo, costs=costs_crypto, liquidity=liq_crypto,
                     initial_capital=1_000_000).run(start=panel_crypto_ftmo.dates[40])

def stats(r):
    rep = compute_metrics(r.returns, r.equity_curve, costs=r.costs, trades=r.trades, ann_factor=365)
    yrs = (r.equity_curve.index[-1] - r.equity_curve.index[0]).days / 365.25
    gS  = sharpe_ratio(r.gross_pnl / r.equity_curve.shift(1), ann_factor=365)
    return rep, gS, yrs

rep_b, gS_b, yrs_b = stats(result_best)   # full 59-coin benchmark
rep_f, gS_f, yrs_f = stats(result_ftmo)   # FTMO-13

print("universe   | nNames | netSh | grossSh | netRet | vol   | maxDD | turn/yr | PSR")
def row(name, n, rep, gS, yrs, r):
    print(f"{name:<10} |  {n:>4}  | {rep.sharpe:>+5.2f} | {gS:>+6.2f}  | "
          f"{r.equity_curve.iloc[-1]/1e6-1:>+6.1%} | {rep.ann_vol:>5.1%} | {rep.max_drawdown:>5.1%} | "
          f"{rep.turnover/yrs:>5.0f}x  | {rep.psr:.2f}")
row("full 59", 59, rep_b, gS_b, yrs_b, result_best)
row("FTMO 13", 13, rep_f, gS_f, yrs_f, result_ftmo)
print()
print(f"net Sharpe retention: {rep_f.sharpe/rep_b.sharpe:5.0%} of full-universe edge")

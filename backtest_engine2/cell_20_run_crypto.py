# Cell 20 — Crypto reversal: run + metrics (ann_factor=365)
strat_crypto = ReversalNeutral(lookback=5, clip=1.0, decay=5, neutralize=False)
result_crypto = Engine(panel_crypto, strat_crypto, costs=costs_crypto, liquidity=liq_crypto,
                       initial_capital=1_000_000).run(start=panel_crypto.dates[WARMUP_BARS])

rep_c = compute_metrics(result_crypto.returns, result_crypto.equity_curve,
                        costs=result_crypto.costs, trades=result_crypto.trades, ann_factor=365)
yrs_c = (result_crypto.equity_curve.index[-1] - result_crypto.equity_curve.index[0]).days / 365.25
gS_c = sharpe_ratio(result_crypto.gross_pnl / result_crypto.equity_curve.shift(1), ann_factor=365)

print(rep_c)
print()
print(f"NET   ret:    {result_crypto.equity_curve.iloc[-1]/1e6-1:+.2%}   over {yrs_c:.1f}y")
print(f"NET   Sharpe: {rep_c.sharpe:+.3f}  (95% CI [{rep_c.sharpe_ci_low:+.2f}, {rep_c.sharpe_ci_high:+.2f}])")
print(f"GROSS Sharpe: {gS_c:+.3f}   <- is reversal alive in crypto?")
print(f"Turnover:     {rep_c.turnover/yrs_c:.0f}x per year")
print(f"Margin:       {rep_c.margin*1e4:+.2f} bps net per $ traded")
print(f"PSR(0):       {rep_c.psr:.3f}   {'SIG' if rep_c.psr>0.95 else 'not sig'} at 95%")

# Cell 8 — Trial 3: flipped skew with window=30 (horizon aligned to #006's 30d plateau)
strat_flip30 = SkewLotteryFlipped(window=30, decay=5)
result_flip30 = Engine(panel_skew, strat_flip30, costs=costs_ftmo, liquidity=liq_ftmo,
                       initial_capital=1_000_000).run(start=panel_skew.dates[40])

rep_f30 = compute_metrics(result_flip30.returns, result_flip30.equity_curve,
                          costs=result_flip30.costs, trades=result_flip30.trades,
                          ann_factor=365)
yrs_f30 = (result_flip30.equity_curve.index[-1] - result_flip30.equity_curve.index[0]).days / 365.25
gS_f30 = sharpe_ratio(result_flip30.gross_pnl / result_flip30.equity_curve.shift(1), ann_factor=365)

print(strat_flip30)
print(f"NET   ret:    {result_flip30.equity_curve.iloc[-1]/1e6-1:+.2%}   over {yrs_f30:.1f}y")
print(f"NET   Sharpe: {rep_f30.sharpe:+.3f}  (95% CI [{rep_f30.sharpe_ci_low:+.2f}, {rep_f30.sharpe_ci_high:+.2f}])")
print(f"GROSS Sharpe: {gS_f30:+.3f}")
print(f"Turnover:     {rep_f30.turnover/yrs_f30:.0f}x/yr   Margin: {rep_f30.margin*1e4:+.2f} bps/$")
print(f"Max DD:       {rep_f30.max_drawdown:.1%}   Ann vol: {rep_f30.ann_vol:.1%}")
print(f"PSR(0):       {rep_f30.psr:.3f}")

both30 = pd.concat([result_flip30.returns.rename("skew30"),
                    result_m30.returns.rename("mom30")], axis=1).dropna()
print(f"corr(skew30, mom30): {both30['skew30'].corr(both30['mom30']):+.3f}")

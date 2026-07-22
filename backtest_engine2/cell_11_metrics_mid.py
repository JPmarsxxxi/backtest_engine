# Cell 11 — Metrics on trial #002 (mid-cap ranked reversal)
rep_mid = compute_metrics(
    result_mid.returns, result_mid.equity_curve,
    costs=result_mid.costs, trades=result_mid.trades,
    ann_factor=252, ci_level=0.95,
)
years_m = (result_mid.equity_curve.index[-1] - result_mid.equity_curve.index[0]).days / 365.25
gross_ret_m = result_mid.gross_pnl / result_mid.equity_curve.shift(1)
gross_sharpe_m = sharpe_ratio(gross_ret_m, ann_factor=252)

print(rep_mid)
print()
print(f"NET   Sharpe:  {rep_mid.sharpe:+.3f}  (95% CI [{rep_mid.sharpe_ci_low:+.2f}, {rep_mid.sharpe_ci_high:+.2f}])")
print(f"GROSS Sharpe:  {gross_sharpe_m:+.3f}   <- is the edge real?")
print(f"Turnover:      {rep_mid.turnover:.0f}x total -> {rep_mid.turnover/years_m:.0f}x per year")
print(f"Margin:        {rep_mid.margin*1e4:+.2f} bps net PnL per $ traded")
print(f"PSR(0):        {rep_mid.psr:.3f}   GROSS would need cost cut to realize")

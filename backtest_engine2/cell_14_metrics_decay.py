# Cell 14 — Metrics on trial #003 (decay)
rep_dec = compute_metrics(
    result_decay.returns, result_decay.equity_curve,
    costs=result_decay.costs, trades=result_decay.trades,
    ann_factor=252, ci_level=0.95,
)
years_d = (result_decay.equity_curve.index[-1] - result_decay.equity_curve.index[0]).days / 365.25
gross_sharpe_d = sharpe_ratio(result_decay.gross_pnl / result_decay.equity_curve.shift(1), ann_factor=252)

print(rep_dec)
print()
print(f"NET   Sharpe:  {rep_dec.sharpe:+.3f}  (95% CI [{rep_dec.sharpe_ci_low:+.2f}, {rep_dec.sharpe_ci_high:+.2f}])")
print(f"GROSS Sharpe:  {gross_sharpe_d:+.3f}")
print(f"Turnover:      {rep_dec.turnover/years_d:.0f}x per year  (was 143x)")
print(f"Margin:        {rep_dec.margin*1e4:+.2f} bps net PnL per $ traded")
print(f"PSR(0):        {rep_dec.psr:.3f}   {'SIG' if rep_dec.psr>0.95 else 'not sig'} at 95%")
print(f"MinTRL(0):     {rep_dec.min_trl:.0f} bars vs sample {rep_dec.n_obs}")

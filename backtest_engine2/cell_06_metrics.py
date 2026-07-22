# Cell 6 — Metrics report (Stage 7) on the 3%-capped trial
from backtest.metrics import compute_metrics, sharpe_ratio

rep = compute_metrics(
    result_cap.returns, result_cap.equity_curve,
    costs=result_cap.costs, trades=result_cap.trades,
    ann_factor=252, ci_level=0.95,
)
years = (result_cap.equity_curve.index[-1] - result_cap.equity_curve.index[0]).days / 365.25
gross_ret = result_cap.gross_pnl / result_cap.equity_curve.shift(1)
gross_sharpe = sharpe_ratio(gross_ret, ann_factor=252)

print(rep)
print()
print(f"NET  Sharpe:   {rep.sharpe:+.3f}  (95% CI [{rep.sharpe_ci_low:+.2f}, {rep.sharpe_ci_high:+.2f}])")
print(f"GROSS Sharpe:  {gross_sharpe:+.3f}   <- edge before costs")
print(f"Turnover:      {rep.turnover:.1f}x total  ->  {rep.turnover/years:.1f}x per year")
print(f"Margin:        {rep.margin*1e4:+.2f} bps net PnL per $ traded")
print(f"PSR(0):        {rep.psr:.3f}   {'sig' if rep.psr>0.95 else 'NOT sig'} at 95%")
print(f"MinTRL(0):     {rep.min_trl:.0f} bars vs sample {rep.n_obs}  "
      f"({'powered' if rep.min_trl<rep.n_obs else 'underpowered'})")

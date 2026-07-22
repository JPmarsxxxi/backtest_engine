# Cell 5 — Metrics report for the short-lottery run
from backtest.metrics import compute_metrics, sharpe_ratio

rep_skew = compute_metrics(result_skew.returns, result_skew.equity_curve,
                           costs=result_skew.costs, trades=result_skew.trades,
                           ann_factor=365)
yrs_s = (result_skew.equity_curve.index[-1] - result_skew.equity_curve.index[0]).days / 365.25
gS_s = sharpe_ratio(result_skew.gross_pnl / result_skew.equity_curve.shift(1), ann_factor=365)

print(rep_skew)
print()
print(f"NET   ret:    {result_skew.equity_curve.iloc[-1]/1e6-1:+.2%}   over {yrs_s:.1f}y")
print(f"NET   Sharpe: {rep_skew.sharpe:+.3f}  (95% CI [{rep_skew.sharpe_ci_low:+.2f}, {rep_skew.sharpe_ci_high:+.2f}])")
print(f"GROSS Sharpe: {gS_s:+.3f}")
print(f"Turnover:     {rep_skew.turnover/yrs_s:.0f}x/yr   Margin: {rep_skew.margin*1e4:+.2f} bps/$")
print(f"Max DD:       {rep_skew.max_drawdown:.1%}   Ann vol: {rep_skew.ann_vol:.1%}")
print(f"PSR(0):       {rep_skew.psr:.3f}")

# Cell 6 — Trial 2: flipped sign = LONG lottery-skew, short steady (user's original hypothesis)
class SkewLotteryFlipped(SkewLottery):
    """Long high trailing skew (lottery-shaped), short low. Underreaction-to-surges story."""
    def __repr__(self):
        return f"SkewLotteryFlipped(window={self.window}, decay={self.decay})"
    def generate_weights(self, data, t):
        return -super().generate_weights(data, t)

strat_flip = SkewLotteryFlipped(window=21, decay=5)
result_flip = Engine(panel_skew, strat_flip, costs=costs_ftmo, liquidity=liq_ftmo,
                     initial_capital=1_000_000).run(start=panel_skew.dates[WARMUP_BARS])

rep_flip = compute_metrics(result_flip.returns, result_flip.equity_curve,
                           costs=result_flip.costs, trades=result_flip.trades,
                           ann_factor=365)
yrs_f = (result_flip.equity_curve.index[-1] - result_flip.equity_curve.index[0]).days / 365.25
gS_f = sharpe_ratio(result_flip.gross_pnl / result_flip.equity_curve.shift(1), ann_factor=365)

print(strat_flip)
print(f"NET   ret:    {result_flip.equity_curve.iloc[-1]/1e6-1:+.2%}   over {yrs_f:.1f}y")
print(f"NET   Sharpe: {rep_flip.sharpe:+.3f}  (95% CI [{rep_flip.sharpe_ci_low:+.2f}, {rep_flip.sharpe_ci_high:+.2f}])")
print(f"GROSS Sharpe: {gS_f:+.3f}")
print(f"Turnover:     {rep_flip.turnover/yrs_f:.0f}x/yr   Margin: {rep_flip.margin*1e4:+.2f} bps/$")
print(f"Max DD:       {rep_flip.max_drawdown:.1%}   Ann vol: {rep_flip.ann_vol:.1%}")
print(f"PSR(0):       {rep_flip.psr:.3f}")

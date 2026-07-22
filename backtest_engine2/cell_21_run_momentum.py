# Cell 21 — Crypto MOMENTUM = flipped reversal (trial #006)
class MomentumRanked(ReversalNeutral):
    """Long recent winners, short recent losers = negation of the reversal signal."""
    def __repr__(self):
        return (f"MomentumRanked(lookback={self.lookback}, clip={self.clip}, decay={self.decay})")
    def generate_weights(self, data, t):
        return -super().generate_weights(data, t)   # flip reversal -> momentum

strat_mom = MomentumRanked(lookback=5, clip=1.0, decay=5, neutralize=False)
result_mom = Engine(panel_crypto, strat_mom, costs=costs_crypto, liquidity=liq_crypto,
                    initial_capital=1_000_000).run(start=panel_crypto.dates[WARMUP_BARS])

rep_m = compute_metrics(result_mom.returns, result_mom.equity_curve,
                        costs=result_mom.costs, trades=result_mom.trades, ann_factor=365)
yrs_m = (result_mom.equity_curve.index[-1] - result_mom.equity_curve.index[0]).days / 365.25
gS_m = sharpe_ratio(result_mom.gross_pnl / result_mom.equity_curve.shift(1), ann_factor=365)

print(rep_m)
print()
print(f"NET   ret:    {result_mom.equity_curve.iloc[-1]/1e6-1:+.2%}   over {yrs_m:.1f}y")
print(f"NET   Sharpe: {rep_m.sharpe:+.3f}  (95% CI [{rep_m.sharpe_ci_low:+.2f}, {rep_m.sharpe_ci_high:+.2f}])")
print(f"GROSS Sharpe: {gS_m:+.3f}")
print(f"Turnover:     {rep_m.turnover/yrs_m:.0f}x/yr   Margin: {rep_m.margin*1e4:+.2f} bps/$")
print(f"PSR(0):       {rep_m.psr:.3f}   {'SIG at 95%' if rep_m.psr>0.95 else 'not sig'}")

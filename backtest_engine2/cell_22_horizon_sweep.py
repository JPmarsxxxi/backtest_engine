# Cell 22 — Crypto momentum: formation-horizon sweep (sensitivity, not cherry-pick)
SWEEP_WARMUP = 40                      # >= max lookback (30) + decay (5) for fair comparison
print("lookback | net Sh | gross Sh | net ret | turn/yr | PSR")
for lb in [3, 5, 10, 15, 20, 30]:
    s = MomentumRanked(lookback=lb, clip=1.0, decay=5, neutralize=False)
    r = Engine(panel_crypto, s, costs=costs_crypto, liquidity=liq_crypto,
               initial_capital=1_000_000).run(start=panel_crypto.dates[SWEEP_WARMUP])
    rep = compute_metrics(r.returns, r.equity_curve, costs=r.costs, trades=r.trades, ann_factor=365)
    yrs = (r.equity_curve.index[-1] - r.equity_curve.index[0]).days / 365.25
    gS = sharpe_ratio(r.gross_pnl / r.equity_curve.shift(1), ann_factor=365)
    print(f"  {lb:>4}d  | {rep.sharpe:>+5.2f}  | {gS:>+6.2f}   | "
          f"{r.equity_curve.iloc[-1]/1e6-1:>+6.1%} | {rep.turnover/yrs:>5.0f}x  | {rep.psr:.2f}")

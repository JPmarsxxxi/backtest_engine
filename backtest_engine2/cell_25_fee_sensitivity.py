# Cell 25 — Fee sensitivity for 30d crypto momentum
scenarios = {
    "institutional": (2, 3),    # (commission_bps, spread_half_bps)
    "base/VIP":      (5, 5),
    "retail":        (10, 10),
}
print("scenario      | comm/spread | net Sh | net ret | margin bps/$")
for name, (cbp, sbp) in scenarios.items():
    cm = CompositeCostModel([
        Commission(bps=cbp), Spread(half_bps=sbp),
        MarketImpact(k=12, kind="sqrt", adv_lookback=20),
        ShortBorrow(annual_bps=100, trading_days=365),
    ])
    r = Engine(panel_crypto, MomentumRanked(lookback=30, clip=1.0, decay=5, neutralize=False),
               costs=cm, liquidity=liq_crypto, initial_capital=1_000_000).run(start=panel_crypto.dates[40])
    rep = compute_metrics(r.returns, r.equity_curve, costs=r.costs, trades=r.trades, ann_factor=365)
    print(f"{name:<13} | {cbp:>2}bp/{sbp:>2}bp   | {rep.sharpe:>+5.2f}  | "
          f"{r.equity_curve.iloc[-1]/1e6-1:>+7.1%} | {rep.margin*1e4:>+6.2f}")

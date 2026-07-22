# Cell 9 — Swap sensitivity for flip30: S = 0 / 5 / 15 %/yr per side (cost scenarios, not new trials)
for label, sbps in [("swap-free (0%/side)", 0), ("baseline (5%/side)", 1000), ("pessimistic (15%/side)", 3000)]:
    comps = [Commission(bps=3.25), Spread(half_bps=5), MarketImpact(k=12, kind="sqrt", adv_lookback=20)]
    if sbps > 0:
        comps.append(ShortBorrow(annual_bps=sbps, trading_days=365))   # 2*S on shorts = S on both legs
    c = CompositeCostModel(comps)
    res = Engine(panel_skew, SkewLotteryFlipped(window=30, decay=5), costs=c, liquidity=liq_ftmo,
                 initial_capital=1_000_000).run(start=panel_skew.dates[40])
    rep = compute_metrics(res.returns, res.equity_curve, costs=res.costs, trades=res.trades,
                          ann_factor=365)
    yrs = (res.equity_curve.index[-1] - res.equity_curve.index[0]).days / 365.25
    print(f"{label:24s} NET Sh {rep.sharpe:+.3f}  net ret {res.equity_curve.iloc[-1]/1e6-1:+8.1%}  "
          f"margin {rep.margin*1e4:+6.2f} bps/$  maxDD {rep.max_drawdown:.0%}  PSR {rep.psr:.3f}")

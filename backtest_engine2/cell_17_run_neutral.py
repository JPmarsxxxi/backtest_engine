# Cell 17 — Run fixed-decay baseline + sector-neutral; full comparison
result_decay_fix = Engine(panel_mid, strat_decay_fix, costs=costs_mid, liquidity=liq_mid,
                          initial_capital=1_000_000).run(start=panel_mid.dates[WARMUP_BARS])
result_neutral = Engine(panel_mid, strat_neutral, costs=costs_mid, liquidity=liq_mid,
                        initial_capital=1_000_000).run(start=panel_mid.dates[WARMUP_BARS])

def summ(name, r):
    rep = compute_metrics(r.returns, r.equity_curve, costs=r.costs, trades=r.trades, ann_factor=252)
    yrs = (r.equity_curve.index[-1] - r.equity_curve.index[0]).days / 365.25
    gS = sharpe_ratio(r.gross_pnl / r.equity_curve.shift(1), ann_factor=252)
    print(f"{name:<14} | NETret {r.equity_curve.iloc[-1]/1e6-1:>+7.2%} | "
          f"netSh {rep.sharpe:>+5.2f} | grossSh {gS:>+5.2f} | "
          f"vol {rep.ann_vol:>5.1%} | maxDD {rep.max_drawdown:>5.1%} | "
          f"turn {rep.turnover/yrs:>4.0f}x | PSR {rep.psr:.2f}")

print("trial          | net ret  | netSh | grossSh | vol   | maxDD | turn | PSR")
summ("003 decay(old)", result_decay)       # original, pad-fill bug
summ("003f decay-fix", result_decay_fix)   # fill_method=None
summ("004 neutral",    result_neutral)     # + sector neutral

# Cell 10 — Engine run: ranked reversal on mid-caps (trial #002)
result_mid = Engine(panel_mid, strat_rank, costs=costs_mid, liquidity=liq_mid,
                    initial_capital=1_000_000).run(start=panel_mid.dates[WARMUP_BARS])

print("trial       | max/name | gross-exp | avg pos | NET ret | GROSS ret | costs")
line("001 cap3%", result_cap)   # S&P100 raw-magnitude, large-cap costs
line("002 rank",  result_mid)   # S&P400 ranked+clipped, mid-cap costs

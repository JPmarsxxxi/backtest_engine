# Cell 13 — Engine run: decay variant (trial #003)
result_decay = Engine(panel_mid, strat_decay, costs=costs_mid, liquidity=liq_mid,
                      initial_capital=1_000_000).run(start=panel_mid.dates[WARMUP_BARS])

print("trial        | max/name | gross-exp | avg pos | NET ret | GROSS ret | costs")
line("002 rank",  result_mid)
line("003 decay", result_decay)

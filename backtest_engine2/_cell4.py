# Cell 4 — Single-path Engine run
import time
from backtest.engine import Engine

WARMUP_BARS = 5400   # coint_lookback (4320) + hedge (720) + tw (168) + ~8d buffer

engine = Engine(
    panel,
    strat,
    costs=costs,
    liquidity=liq,
    initial_capital=1_000_000.0,
)

print(f"Engine constructed. Starting backtest from {panel.dates[WARMUP_BARS]} ...")
print(f"Expected runtime: 25-35 min (60k hourly bars × 20 pairs × per-bar rolling signal)\n")

t0 = time.time()
result = engine.run(start=panel.dates[WARMUP_BARS])
elapsed = time.time() - t0
print(f"...done in {elapsed/60:.1f} min.\n")

# Validation block (skills/05-engine-single-path.md §9)
md = result.metadata
init_cap = md["initial_capital"]
final_eq = result.equity_curve.iloc[-1]
total_ret = final_eq / init_cap - 1

print(f"Bars                 : {len(result.equity_curve):,}")
print(f"Rebalances           : {md['n_rebalances']:,}  (expected = bars, hourly)")
print(f"Date range           : {md['start']} -> {md['end']}")
print(f"Initial capital      : ${init_cap:>14,.0f}")
print(f"Final equity         : ${final_eq:>14,.0f}")
print(f"Total return         : {total_ret:+.2%}")
print(f"Total costs paid     : ${result.costs.sum():>14,.0f}")
print(f"Max |position weight|: {result.weights.abs().max().max():.2%}    (cap = 20% by DEFAULT_RISK_CONFIG)")
print(f"Max gross exposure   : {result.weights.abs().sum(axis=1).max():.2%}    (cap = 100% by DEFAULT_RISK_CONFIG)")
print(f"Max net exposure     : {result.weights.sum(axis=1).abs().max():.2%}    (cap = 100%, paper-style $-neutral expects ~0%)")
print(f"Open trades at end   : {len(strat.open_trades)}")
print(f"Unfilled (sum |$|)   : ${result.unfilled.abs().sum().sum():>14,.0f}    (= LiquidityCap throttling)")

# Tiny sanity: how many bars had any non-zero exposure?
non_idle = (result.weights.abs().sum(axis=1) > 1e-6).sum()
print(f"Bars with exposure   : {non_idle:,} / {len(result.equity_curve):,}  ({non_idle/len(result.equity_curve):.1%})")

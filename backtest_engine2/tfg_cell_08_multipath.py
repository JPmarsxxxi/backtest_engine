# Cell 8 — Stage 6: MultiPathEngine run (skills/07-multipath.md).
# 15 splits -> 5 stitched paths over eval_dates; vol_only @ paper caps via the
# _make_strat factory (fresh instance per split). Costs + liquidity from Stage 3.
# No summary()/aggregate_metrics() here — measuring is Stage 7 (skills/08-metrics.md).
from backtest.engine import MultiPathEngine

mp = MultiPathEngine(
    panel,
    _make_strat,          # zero-arg factory: vol_only, paper W_max caps (Cell 6)
    splitter,             # CPCV(6, 2, purge=50, embargo=0.01) from Cell 7
    costs=costs,
    liquidity=liq,
    initial_capital=1_000_000,
)
mp_res = mp.run(dates=eval_dates, n_jobs=-1)

print(f"Splits run:           {mp_res.n_splits}")
print(f"Paths assembled:      {mp_res.n_paths}")
print(f"path_returns shape:   {mp_res.path_returns.shape}")
print(f"path_equity shape:    {mp_res.path_equity.shape}")
print(f"Date range:           {mp_res.path_returns.index[0].date()} -> "
      f"{mp_res.path_returns.index[-1].date()}")
print(f"Per-path NaN count:   min={mp_res.path_returns.isna().sum().min()}, "
      f"max={mp_res.path_returns.isna().sum().max()}")
total_cost_all_splits = sum(
    float(seg.costs.sum()) for segs in mp_res.split_results for seg in segs
)
print(f"Total costs (15 splits): ${total_cost_all_splits:,.0f}")
print("\nFinal equity per path:")
print(mp_res.path_equity.iloc[-1].rename("final_equity").map(lambda v: f"${v:,.0f}"))

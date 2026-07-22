# Cell 9 — Stage 7: metrics on the CPCV multi-path result (skills/08-metrics.md).
# summary() = aggregate table + per-path equity overlay + Sharpe histogram
# (it calls aggregate_metrics/metrics_per_path internally — single entry point).
# PSR here is single-trial evidence; DSR deflation is Stage 9, NOT this cell.
agg = mp_res.summary()

per_path = mp_res.metrics_per_path()
pp = pd.DataFrame({
    f"path_{i}": {
        "sharpe": r.sharpe,
        "sortino": r.sortino,
        "psr": r.psr,
        "max_dd": r.max_drawdown,
        "ann_return": r.ann_return,
        "ann_vol": r.ann_vol,
        "min_trl_bars": r.min_trl,
        "n_obs": r.n_obs,
    }
    for i, r in enumerate(per_path)
}).T

print("\nPer-path metrics:")
print(pp.to_string(float_format=lambda v: f"{v:,.3f}"))
print(f"\nSharpe across {mp_res.n_paths} paths: mean={agg['sharpe_mean']:.3f} "
      f"std={agg['sharpe_std']:.3f} min={agg['sharpe_min']:.3f} max={agg['sharpe_max']:.3f}")
print(f"PSR(0) across paths:    mean={agg['psr_mean']:.3f} min={agg['psr_min']:.3f}")
print(f"MaxDD across paths:     mean={agg['max_drawdown_mean']:.2%} worst={agg['max_drawdown_max']:.2%}")
powered = all(r.min_trl < r.n_obs for r in per_path if pd.notna(r.min_trl))
print(f"MinTRL vs sample:       all paths powered = {powered}")

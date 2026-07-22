# Cell 7 — Metrics
# ann_factor = 8760 (24h × 365d): crypto trades 24/7, hourly bars

ANN_FACTOR = 8760

# ── Single-path report + dashboard ───────────────────────────────────────────
print("=" * 60)
print("SINGLE-PATH (full sample)")
print("=" * 60)
single_rep = result.summary(ann_factor=ANN_FACTOR)

# ── Multi-path aggregate ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"CPCV ({res.n_paths} paths)  —  aggregate metrics")
print("=" * 60)
agg      = res.aggregate_metrics()
per_path = res.metrics_per_path()

_rows = ["sharpe", "sortino", "calmar", "max_drawdown", "psr"]
_cols = ["mean", "std", "min", "max"]
_hdr  = f"{'metric':<20}" + "".join(f"{c:>10}" for c in _cols)
print(_hdr)
print("-" * len(_hdr))
for row in _rows:
    vals = [agg.get(f"{row}_{c}", float("nan")) for c in _cols]
    print(f"{row:<20}" + "".join(f"{v:>10.3f}" if not (v != v) else f"{'NaN':>10}" for v in vals))

print()
print(f"{'path':<10}{'sharpe':>10}{'ann_ret':>10}{'max_dd':>10}{'psr':>10}")
print("-" * 42)
for i, rep in enumerate(per_path):
    print(f"{'path_'+str(i):<10}{rep.sharpe:>10.3f}{rep.ann_return:>10.2%}"
          f"{rep.max_drawdown:>10.2%}{rep.psr:>10.3f}")

print()
print(f"Sharpe mean ± std : {agg['sharpe_mean']:.3f} ± {agg['sharpe_std']:.3f}")
print(f"PSR(0) mean       : {agg['psr_mean']:.3f}  "
      f"({'all paths significant' if agg['psr_min'] > 0.95 else 'not all paths significant'} at 95%)")
print(f"Max drawdown mean : {agg['max_drawdown_mean']:.2%}")

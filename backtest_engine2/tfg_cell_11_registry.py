# Cell 11 — Stage 10: trial registry (skills/11-registry.md).
# Log the K=2 committed trials (full-Kelly, vol_only) under one family with ONE
# shared evaluation procedure: single-path, fit once 2005-2014, test 2015-2025-10-31
# (reg.run is the canonical entry; family must not mix WF/CPCV procedures).
# Causal graph: diagrams/forecast_to_fill_gold_causal.png (spec Sec 14.1 hypothesis).
# NOTE: do NOT re-run this cell — each execution logs new trials and inflates K.
from backtest.registry import TrialRegistry

reg = TrialRegistry("./trial_store")
FAMILY = "forecast_to_fill_gold"
GRAPH = "diagrams/forecast_to_fill_gold_causal.png"

train_dates_reg = panel.dates[(panel.dates >= "2005-01-01") & (panel.dates <= "2014-12-31")]
test_dates_reg = panel.dates[(panel.dates >= "2015-01-01") & (panel.dates <= "2025-10-31")]

logged = {}
for name, make in [("full_kelly", ForecastToFillGoldStrategy), ("vol_only", _make_strat)]:
    eng = Engine(panel, make(), costs=costs, liquidity=liq, initial_capital=1_000_000)
    res = reg.run(eng, causal_graph_path=GRAPH, family=FAMILY,
                  train_dates=train_dates_reg, test_dates=test_dates_reg)
    r = res.returns.dropna()
    logged[name] = r.mean() / r.std() * np.sqrt(P["tdays"]) if r.std() > 0 else float("nan")

trials = reg.list(family=FAMILY, include_exploratory=False)
K_count = reg.k(family=FAMILY, method="count")
K_eff = reg.k(family=FAMILY, method="effective", threshold=0.5)
dsr_last = reg.dsr_for(trials[-1]["id"], family=FAMILY, method="effective", threshold=0.5)

print(f"Family: {FAMILY} | store: {reg.directory}")
for nm, sh in logged.items():
    print(f"  logged {nm:11s} single-path OOS Sharpe {sh:+.3f}")
print(f"Trials in family:  {len(trials)}")
print(f"K (count):         {K_count} | K (effective@0.5): {K_eff}")
print(f"DSR of last trial (vol_only, K_eff from registry): {dsr_last:.4f}")
for t in trials:
    print(f"  id={t['id']} hash={t['strategy_hash']} exploratory={t['exploratory']} "
          f"sharpe={t['sharpe']:+.3f} psr={t['psr']:.3f} "
          f"test={t['test_start'][:10]} -> {t['test_end'][:10]}")

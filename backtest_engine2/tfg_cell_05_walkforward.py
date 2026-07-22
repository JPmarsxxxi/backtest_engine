# Cell 5 — Walk-forward: 10y train / 6m test, non-overlapping 6-month folds (spec).
# Each OOS day belongs to exactly one fold; refit every 6 months on trailing 10y.
# WalkForward splitter is single-split only — manual fold loop here (PROTOCOL §7).
import numpy as np
import pandas as pd
from backtest.engine import Engine
from backtest.metrics import compute_metrics

OOS_START = pd.Timestamp("2015-01-01")
OOS_END = pd.Timestamp("2025-10-31")
TRAIN_YEARS = 10
TEST_MONTHS = 6
CAPITAL = 1_000_000

fold_months = pd.date_range(OOS_START, OOS_END, freq="6MS")  # non-overlapping 6m folds
stitched_ret: dict = {}
fold_rows = []
total_costs = 0.0

for i, month in enumerate(fold_months):
    pos = panel.dates.searchsorted(month, side="left")
    if pos >= len(panel.dates):
        break
    ts = panel.dates[pos]
    if ts > OOS_END:
        break

    prior = panel.dates[panel.dates < ts]
    if len(prior) == 0:
        continue
    train_end = prior[-1]
    train_start = train_end - pd.DateOffset(years=TRAIN_YEARS)
    train_dates = panel.dates[(panel.dates >= train_start) & (panel.dates <= train_end)]
    test_end = min(ts + pd.DateOffset(months=TEST_MONTHS), OOS_END)
    test_dates = panel.dates[(panel.dates >= ts) & (panel.dates <= test_end)]
    if len(train_dates) < 252 * 5 or len(test_dates) < 10:
        continue

    strat_fold = ForecastToFillGoldStrategy()
    eng = Engine(panel, strat_fold, costs=costs, liquidity=liq, initial_capital=CAPITAL)
    res = eng.run(train_dates=train_dates, test_dates=test_dates)
    total_costs += float(res.costs.sum())
    r = res.returns.dropna()
    for t, v in r.items():
        stitched_ret[t] = v
    fold_rows.append({
        "test_start": ts.date(),
        "train_bars": len(train_dates),
        "test_bars": len(test_dates),
        "lam": strat_fold._lam,
        "net_ret": float((1 + r).prod() - 1),
    })
    if (i + 1) % 4 == 0:
        print(f"  {i + 1}/{len(fold_months)} folds through {ts.date()} ...")

wf_oos_ret = pd.Series(stitched_ret, name="return").sort_index()
wf_oos_ret = wf_oos_ret[(wf_oos_ret.index >= OOS_START) & (wf_oos_ret.index <= OOS_END)]
wf_oos_eq = (1.0 + wf_oos_ret).cumprod() * CAPITAL

bm = benchmark.reindex(wf_oos_ret.index).pct_change().fillna(0.0)
wf_metrics = compute_metrics(wf_oos_ret, wf_oos_eq, benchmark=bm)

fold_df = pd.DataFrame(fold_rows)

print(f"Walk-forward folds run: {len(fold_df)}")
print(f"Stitched OOS: {wf_oos_ret.index[0].date()} -> {wf_oos_ret.index[-1].date()} "
      f"({len(wf_oos_ret)} bars)")
print(f"Total costs (all folds): ${total_costs:,.0f}")
print(f"Sharpe: {wf_metrics.sharpe:+.2f} | Sortino: {wf_metrics.sortino:+.2f} | "
      f"Calmar: {wf_metrics.calmar:+.2f}")
print(f"Ann return: {wf_metrics.ann_return:.2%} | ann vol: {wf_metrics.ann_vol:.2%} | "
      f"maxDD: {wf_metrics.max_drawdown:.2%}")
print(f"Hit rate: {wf_metrics.hit_rate:.1%} | IR vs XAUUSD: {wf_metrics.information_ratio:+.2f} | "
      f"beta: {wf_metrics.beta:+.3f}")
print(f"Mean fold lam: {fold_df.lam.mean():.3f} (min {fold_df.lam.min():.2f}, max {fold_df.lam.max():.2f})")

wf_oos_eq.tail(3)

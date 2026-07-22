# Cell 6 — Walk-forward @ paper 15% vol budget (vol_only=True, W_max=2.0 caps).
# Skips fractional-Kelly attenuation; sizes via vol-target + confidence shaping only.
# Non-overlapping 6-month folds, refit on trailing 10y (same scheme as Cell 5).
import numpy as np
import pandas as pd
from backtest.engine import Engine
from backtest.metrics import compute_metrics

OOS_START = pd.Timestamp("2015-01-01")
OOS_END = pd.Timestamp("2025-10-31")
TRAIN_YEARS = 10
TEST_MONTHS = 6
CAPITAL = 1_000_000

def _make_strat():
    return ForecastToFillGoldStrategy(
        max_position=P["w_max"],
        max_gross=P["w_max"],
        max_net=P["w_max"],
        vol_only=True,
    )

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

    strat_fold = _make_strat()
    eng = Engine(panel, strat_fold, costs=costs, liquidity=liq, initial_capital=CAPITAL)
    res = eng.run(train_dates=train_dates, test_dates=test_dates)
    total_costs += float(res.costs.sum())
    r = res.returns.dropna()
    for t, v in r.items():
        stitched_ret[t] = v
    fold_rows.append({
        "test_start": ts.date(),
        "test_bars": len(test_dates),
        "net_ret": float((1 + r).prod() - 1),
        "max_w": float(res.weights[ASSET].abs().max()),
        "unfilled": float(res.unfilled.abs().sum().sum()),
    })
    if (i + 1) % 4 == 0:
        print(f"  {i + 1}/{len(fold_months)} folds through {ts.date()} ...")

wf15_oos_ret = pd.Series(stitched_ret, name="return").sort_index()
wf15_oos_ret = wf15_oos_ret[(wf15_oos_ret.index >= OOS_START) & (wf15_oos_ret.index <= OOS_END)]
wf15_oos_eq = (1.0 + wf15_oos_ret).cumprod() * CAPITAL
bm = benchmark.reindex(wf15_oos_ret.index).pct_change().fillna(0.0)
wf15_metrics = compute_metrics(wf15_oos_ret, wf15_oos_eq, benchmark=bm)

fold_df15 = pd.DataFrame(fold_rows)
print("Mode: vol_only (15% ann vol target + confidence; no Kelly shrinkage)")
print(f"Risk caps: max_position={P['w_max']} (paper W_max)")
print(f"Walk-forward folds: {len(fold_df15)} | OOS bars: {len(wf15_oos_ret)}")
print(f"Total costs: ${total_costs:,.0f}")
print(f"Sharpe: {wf15_metrics.sharpe:+.2f} | Sortino: {wf15_metrics.sortino:+.2f} | "
      f"Calmar: {wf15_metrics.calmar:+.2f}")
print(f"Ann return: {wf15_metrics.ann_return:.2%} | ann vol: {wf15_metrics.ann_vol:.2%} | "
      f"maxDD: {wf15_metrics.max_drawdown:.2%}")
print(f"Hit rate: {wf15_metrics.hit_rate:.1%} | IR: {wf15_metrics.information_ratio:+.2f} | "
      f"beta: {wf15_metrics.beta:+.3f}")
print(f"Max |weight| across folds: {fold_df15.max_w.max():.2%} | "
      f"mean |weight| max per fold: {fold_df15.max_w.mean():.2%}")
print(f"Unfilled notional (all folds): ${fold_df15.unfilled.sum():,.0f}")
print(f"Final equity: ${wf15_oos_eq.iloc[-1]:,.0f} (P&L {wf15_oos_eq.iloc[-1] / CAPITAL - 1:+.2%})")

# Compare to 1% / full-Kelly run (Cell 5) if present
if "wf_metrics" in dir():
    print(f"\nvs Cell 5 (1% cap + Kelly): Sharpe {wf_metrics.sharpe:+.2f} -> {wf15_metrics.sharpe:+.2f} | "
          f"ann vol {wf_metrics.ann_vol:.2%} -> {wf15_metrics.ann_vol:.2%}")

wf15_oos_eq.tail(3)

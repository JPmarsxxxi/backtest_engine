# Cell 13 — EXPLORATORY: leverage-to-budget variant (lev = 15% / 6.37% realized).
# Cell 6 showed the 15% vol budget is never realized (flat ~76% of days -> 6.37% ann vol).
# This scales the vol_only target by lev, re-capped at paper W_max=2.0, so in-trade
# sizing spends the budget. Cap binds when unlevered w_conf > W_max/lev = 0.85
# (Cell 6 peak was 1.37), so realized vol will land below 15%. Same 22-fold WF as
# Cell 6. New trial for the registry (exploratory) — do NOT treat as the headline run.
# Self-contained: redefines WF constants so it runs on a fresh kernel with only Cells 1-3.
from backtest.engine import Engine
from backtest.metrics import compute_metrics

OOS_START = pd.Timestamp("2015-01-01")
OOS_END = pd.Timestamp("2025-10-31")
TRAIN_YEARS = 10
TEST_MONTHS = 6
CAPITAL = 1_000_000
fold_months = pd.date_range(OOS_START, OOS_END, freq="6MS")
LEV = 0.15 / 0.0637  # ~2.35


class LeveredForecastToFillGold(ForecastToFillGoldStrategy):
    """vol_only sizing scaled by lev, re-capped at paper W_max. All else identical."""

    def __init__(self, lev: float, **kw):
        super().__init__(**kw)
        self._lev = lev

    def _raw_target(self, data, t: pd.Timestamp) -> float:
        base = super()._raw_target(data, t)  # parent cap never binds: w_conf <= w_vol <= W_max
        return float(min(P["w_max"], self._lev * base))


def _make_strat13():
    return LeveredForecastToFillGold(
        lev=LEV,
        max_position=P["w_max"],
        max_gross=P["w_max"],
        max_net=P["w_max"],
        vol_only=True,
    )


stitched13: dict = {}
fold_rows13 = []
costs13 = 0.0
cap_days = 0
active_days = 0

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

    eng = Engine(panel, _make_strat13(), costs=costs, liquidity=liq, initial_capital=CAPITAL)
    res = eng.run(train_dates=train_dates, test_dates=test_dates)
    costs13 += float(res.costs.sum())
    w = res.weights[ASSET].abs()
    cap_days += int((w > P["w_max"] - 1e-6).sum())
    active_days += int((w > 1e-4).sum())
    r = res.returns.dropna()
    for t, v in r.items():
        stitched13[t] = v
    fold_rows13.append({
        "test_start": ts.date(),
        "net_ret": float((1 + r).prod() - 1),
        "max_w": float(w.max()),
        "unfilled": float(res.unfilled.abs().sum().sum()),
    })
    if (i + 1) % 4 == 0:
        print(f"  {i + 1}/{len(fold_months)} folds through {ts.date()} ...")

wf13_ret = pd.Series(stitched13, name="return").sort_index()
wf13_ret = wf13_ret[(wf13_ret.index >= OOS_START) & (wf13_ret.index <= OOS_END)]
wf13_eq = (1.0 + wf13_ret).cumprod() * CAPITAL
bm13 = benchmark.reindex(wf13_ret.index).pct_change().fillna(0.0)
wf13_metrics = compute_metrics(wf13_ret, wf13_eq, benchmark=bm13)

fold_df13 = pd.DataFrame(fold_rows13)
print(f"Mode: levered vol_only  lev={LEV:.2f}  cap=W_max={P['w_max']}")
print(f"Folds: {len(fold_df13)} | OOS bars: {len(wf13_ret)} | total costs: ${costs13:,.0f}")
print(f"Sharpe: {wf13_metrics.sharpe:+.2f} | Sortino: {wf13_metrics.sortino:+.2f} | "
      f"Calmar: {wf13_metrics.calmar:+.2f}")
print(f"Ann return: {wf13_metrics.ann_return:.2%} | ann vol: {wf13_metrics.ann_vol:.2%} | "
      f"maxDD: {wf13_metrics.max_drawdown:.2%}")
print(f"Hit rate: {wf13_metrics.hit_rate:.1%} | IR vs XAUUSD: {wf13_metrics.information_ratio:+.2f} | "
      f"beta: {wf13_metrics.beta:+.3f}")
print(f"Cap-bound days: {cap_days}/{active_days} active ({cap_days / max(active_days, 1):.0%}) | "
      f"max |w|: {fold_df13.max_w.max():.2%}")
print(f"Unfilled notional: ${fold_df13.unfilled.sum():,.0f}")
print(f"Final equity: ${wf13_eq.iloc[-1]:,.0f} (P&L {wf13_eq.iloc[-1] / CAPITAL - 1:+.2%})")
# Cell 6 reference (stored run: Sharpe +0.55, ann ret 3.47%, vol 6.37%, maxDD 13.31%)
print(f"\nvs Cell 6 (unlevered): Sharpe +0.55 -> {wf13_metrics.sharpe:+.2f} | "
      f"ann ret 3.47% -> {wf13_metrics.ann_return:.2%} | "
      f"vol 6.37% -> {wf13_metrics.ann_vol:.2%} | "
      f"maxDD 13.31% -> {wf13_metrics.max_drawdown:.2%}")
wf13_eq.tail(3)

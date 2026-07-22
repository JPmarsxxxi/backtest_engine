# Cell 14 — EXPLORATORY: 2011-2015 gold bear-window diagnostic (not a selection trial).
# Question: does the trend overlay protect capital when the drift disappears?
# Same WF scheme as Cells 6/13 (6m non-overlapping folds, refit each fold), but:
#   MISMATCH 1: data starts 2005 -> early folds train on ~6y, not the spec's 10y.
#   MISMATCH 2: 2015 overlaps the headline OOS window -> diagnostic only, K unaffected.
# Runs both scales (unlevered vol_only + lev=2.35) and buy-and-hold on identical dates.
BEAR_START = pd.Timestamp("2011-01-01")
BEAR_END = pd.Timestamp("2015-12-31")
bear_months = pd.date_range(BEAR_START, BEAR_END, freq="6MS")


def _make_strat14():  # Cell 6's unlevered vol_only config (Cell 6 not loaded in this kernel)
    return ForecastToFillGoldStrategy(
        max_position=P["w_max"], max_gross=P["w_max"], max_net=P["w_max"], vol_only=True,
    )


runs14 = {"unlevered": _make_strat14, "levered": _make_strat13}
stitched14: dict = {name: {} for name in runs14}
costs14 = {name: 0.0 for name in runs14}
train_spans = []

for i, month in enumerate(bear_months):
    pos = panel.dates.searchsorted(month, side="left")
    if pos >= len(panel.dates):
        break
    ts = panel.dates[pos]
    if ts > BEAR_END:
        break
    prior = panel.dates[panel.dates < ts]
    if len(prior) == 0:
        continue
    train_end = prior[-1]
    train_start = train_end - pd.DateOffset(years=TRAIN_YEARS)
    train_dates = panel.dates[(panel.dates >= train_start) & (panel.dates <= train_end)]
    test_end = min(ts + pd.DateOffset(months=TEST_MONTHS), BEAR_END)
    test_dates = panel.dates[(panel.dates >= ts) & (panel.dates <= test_end)]
    if len(train_dates) < 252 * 5 or len(test_dates) < 10:
        continue
    train_spans.append(len(train_dates) / 252)

    for name, maker in runs14.items():
        eng = Engine(panel, maker(), costs=costs, liquidity=liq, initial_capital=CAPITAL)
        res = eng.run(train_dates=train_dates, test_dates=test_dates)
        costs14[name] += float(res.costs.sum())
        for t, v in res.returns.dropna().items():
            stitched14[name][t] = v
    print(f"  fold {i + 1}/{len(bear_months)} {ts.date()} (train {train_spans[-1]:.1f}y) done")

print(f"\nBear window 2011-01 -> 2015-12 | train spans {min(train_spans):.1f}y-{max(train_spans):.1f}y "
      f"(spec wants 10y; data starts 2005)")

rows14 = []
ret14 = {}
for name in runs14:
    r = pd.Series(stitched14[name], name="return").sort_index()
    r = r[(r.index >= BEAR_START) & (r.index <= BEAR_END)]
    ret14[name] = r
    eq = (1.0 + r).cumprod() * CAPITAL
    bmr = benchmark.reindex(r.index).pct_change().fillna(0.0)
    m = compute_metrics(r, eq, benchmark=bmr)
    active = float((r.abs() > 1e-6).mean())
    rows14.append({
        "run": name, "ann_ret": m.ann_return, "ann_vol": m.ann_vol, "sharpe": m.sharpe,
        "maxDD": m.max_drawdown, "total": eq.iloc[-1] / CAPITAL - 1,
        "costs": costs14[name], "active_days": active,
    })

# Buy-and-hold spot gold on the identical stitched dates
bh = benchmark.reindex(ret14["unlevered"].index).pct_change().fillna(0.0)
bh_eq = (1.0 + bh).cumprod() * CAPITAL
bh_m = compute_metrics(bh, bh_eq)
rows14.append({
    "run": "buy_hold_XAUUSD", "ann_ret": bh_m.ann_return, "ann_vol": bh_m.ann_vol,
    "sharpe": bh_m.sharpe, "maxDD": bh_m.max_drawdown, "total": bh_eq.iloc[-1] / CAPITAL - 1,
    "costs": 0.0, "active_days": 1.0,
})

df14 = pd.DataFrame(rows14).set_index("run")
with pd.option_context("display.float_format", "{:.4f}".format):
    print(df14.to_string(formatters={
        "ann_ret": "{:+.2%}".format, "ann_vol": "{:.2%}".format, "sharpe": "{:+.2f}".format,
        "maxDD": "{:.2%}".format, "total": "{:+.2%}".format, "costs": "${:,.0f}".format,
        "active_days": "{:.1%}".format,
    }))

print(f"\nGold over window: {benchmark.reindex(ret14['unlevered'].index).iloc[0]:,.0f} -> "
      f"{benchmark.reindex(ret14['unlevered'].index).dropna().iloc[-1]:,.0f}")

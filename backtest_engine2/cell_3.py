# Cell 4 — Rolling walk-forward (1yr OOS steps) — NOT single in-sample
# Each calendar year: fresh strategy, $1M capital, test window only.
# Indicators at t still use history <= t (no future prices). Stitched OOS metrics.
# Set RUN_SINGLE_IN_SAMPLE=True for one full-history smoke run only.

import time
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.engine import Engine
from backtest.metrics import compute_metrics

WARMUP_BARS = max(
    _OLS_WINDOW,
    _MAX_HOLD,
    globals().get("W_LIQ", 168),
    globals().get("T_TREND", 168),
)
ANN_FACTOR = HOURS_PER_YEAR
INITIAL_CAPITAL = 1_000_000.0
FIRST_OOS_YEAR = 2020
EMBARGO_BARS = 168  # gap after each OOS year before next (no overlap in stitched returns)

RUN_SINGLE_IN_SAMPLE = False

DECISION_LOG_PATH = Path("data/ou_trend_decision_log.parquet")
TRADE_LOG_PATH = Path("data/ou_trend_trade_log.parquet")
DECISION_LOG_PATH.parent.mkdir(exist_ok=True)


def make_strat():
    return OuTrendPullbackStrategy(fast_mode=FAST_MODE)


def _tag_logs(strat, oos_year: int):
    for row in strat.decision_log:
        row["oos_year"] = oos_year
    for row in strat.trade_log:
        row["oos_year"] = oos_year


print(f"FAST_MODE              : {FAST_MODE}")
print(f"OU paths per entry     : {make_strat().num_simulated_paths:,}")
print(f"Walk-forward mode      : {not RUN_SINGLE_IN_SAMPLE}")
print(f"OOS years              : {FIRST_OOS_YEAR}+ (1yr step, fresh capital each year)")
print(f"Embargo between years  : {EMBARGO_BARS} bars\n")

all_decisions = []
all_trades = []
wf_results = []

if RUN_SINGLE_IN_SAMPLE:
    print("SINGLE IN-SAMPLE RUN (debug only — not walk-forward)\n")
    strat = make_strat()
    engine = Engine(panel, strat, costs=costs, liquidity=liq, initial_capital=INITIAL_CAPITAL)
    t0 = time.perf_counter()
    result = engine.run(start=panel.dates[WARMUP_BARS])
    print(f"...done in {(time.perf_counter() - t0) / 60:.1f} min\n")
    _tag_logs(strat, oos_year=-1)
    all_decisions.extend(strat.decision_log)
    all_trades.extend(strat.trade_log)
    wf_results.append({"year": "full", "result": result, "strat": strat})
else:
    panel_end = panel.dates[-1]
    windows = []
    for year in range(FIRST_OOS_YEAR, panel_end.year + 1):
        test_start = pd.Timestamp(f"{year}-01-01")
        test_end = min(pd.Timestamp(f"{year}-12-31 23:00:00"), panel_end)
        test_dates = panel.dates[(panel.dates >= test_start) & (panel.dates <= test_end)]
        if len(test_dates) < max(WARMUP_BARS, 24):
            continue
        windows.append(
            dict(year=year, test_start=test_start, test_end=test_end, test_dates=test_dates)
        )

    print(f"{'Year':>6}  {'OOS start':>12}  {'OOS end':>12}  {'Test bars':>10}")
    for w in windows:
        print(
            f"{w['year']:>6}  {str(w['test_start'].date()):>12}  "
            f"{str(w['test_end'].date()):>12}  {len(w['test_dates']):>10,}"
        )
    print(f"\nRolling walk-forward: {len(windows)} OOS windows\n")

    t_all = time.perf_counter()
    for w in windows:
        print(f"[{w['year']}] OOS {w['test_start'].date()} -> {w['test_end'].date()} ...", flush=True)
        strat_w = make_strat()
        engine = Engine(
            panel, strat_w, costs=costs, liquidity=liq, initial_capital=INITIAL_CAPITAL
        )
        t0 = time.perf_counter()
        res = engine.run(start=w["test_start"], end=w["test_end"])
        dt = time.perf_counter() - t0
        tr = res.equity_curve.iloc[-1] / res.metadata["initial_capital"] - 1
        print(
            f"       {dt / 60:.1f} min | return {tr:+.2%} | trades {len(strat_w.trade_log):,} | "
            f"decisions {len(strat_w.decision_log):,} | costs ${res.costs.sum():,.0f}"
        )
        _tag_logs(strat_w, oos_year=w["year"])
        all_decisions.extend(strat_w.decision_log)
        all_trades.extend(strat_w.trade_log)
        wf_results.append({"year": w["year"], "result": res, "strat": strat_w})

    print(f"\nAll windows done in {(time.perf_counter() - t_all) / 60:.1f} min\n")

    oos_rets = []
    for item in wf_results:
        r = item["result"].returns.dropna()
        if EMBARGO_BARS > 0 and len(r) > EMBARGO_BARS:
            r = r.iloc[EMBARGO_BARS:]
        oos_rets.append(r)
    all_oos_returns = pd.concat(oos_rets).sort_index()
    stitched_equity = (1.0 + all_oos_returns).cumprod() * INITIAL_CAPITAL
    total_ret = stitched_equity.iloc[-1] / INITIAL_CAPITAL - 1

    result = wf_results[-1]["result"]
    strat = wf_results[-1]["strat"]

    print("Stitched OOS (non-overlapping yearly windows, embargo trimmed):")
    print(f"  OOS bars           : {len(all_oos_returns):,}")
    print(f"  Stitched return    : {total_ret:+.2%}")
    wf_metrics = compute_metrics(
        all_oos_returns,
        stitched_equity,
        costs=pd.concat([x["result"].costs for x in wf_results]).reindex(all_oos_returns.index).fillna(0),
        ann_factor=ANN_FACTOR,
    )
    print(f"  Stitched Sharpe    : {wf_metrics.sharpe:+.3f}")
    print(f"  Stitched max DD    : {wf_metrics.max_drawdown:.2%}")

    print(f"\nPer-year OOS returns:")
    per_year = {}
    for item in wf_results:
        y = item["year"]
        r = item["result"].equity_curve.iloc[-1] / item["result"].metadata["initial_capital"] - 1
        per_year[int(y)] = float(r)
        print(f"  {y}: {r:+.2%}  (trades {len(item['strat'].trade_log):,})")
    import json
    summary = {
        "oos_bars": int(len(all_oos_returns)),
        "stitched_return": float(total_ret),
        "stitched_sharpe": float(wf_metrics.sharpe),
        "stitched_max_dd": float(wf_metrics.max_drawdown),
        "per_year_return": per_year,
    }
    _summary_path = DECISION_LOG_PATH.parent / "ou_trend_wf_summary.json"
    _summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWF summary saved -> {_summary_path}")

if all_decisions:
    dec = pd.DataFrame(all_decisions)
    dec.to_parquet(DECISION_LOG_PATH, index=False)
    print(f"\nDecision log saved -> {DECISION_LOG_PATH}  ({len(dec):,} rows)")
    print(dec.groupby("oos_year")["decision"].value_counts().head(12).to_string())

if all_trades:
    trades = pd.DataFrame(all_trades)
    trades.to_parquet(TRADE_LOG_PATH, index=False)
    print(f"Trade log saved -> {TRADE_LOG_PATH}  ({len(trades):,} rows)")
    print(f"Overall win rate   : {trades['win'].mean():.1%}")
    print(trades.groupby("oos_year")["win"].mean().to_string())
else:
    print("\nNo closed trades across walk-forward windows.")
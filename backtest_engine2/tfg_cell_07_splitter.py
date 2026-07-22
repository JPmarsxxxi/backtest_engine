# Cell 7 — Stage 6: CPCV splitter construction (skills/06-splitters.md).
# Spec has no CPCV (walk-forward only, Cells 5-6); PROTOCOL-recommended robustness:
# 15 splits stitched into 5 full-timeline equity paths over the OOS window; feeds DSR.
# Accepted caveat: MultiPathEngine fits at as_of(train[-1]) (multi_path.py:212-213);
# our trailing-10y fit can overlap test groups — purge cannot mask fit inputs.
from backtest.splitters import CombinatorialPurgedCV

CPCV_START = pd.Timestamp("2015-01-01")
CPCV_END = pd.Timestamp("2025-10-31")
eval_dates = panel.dates[(panel.dates >= CPCV_START) & (panel.dates <= CPCV_END)]

splitter = CombinatorialPurgedCV(
    n_splits=6,
    n_test_groups=2,
    purge_bars=50,      # longest feature lookback (K=50 momentum)
    embargo_pct=0.01,
)

n = len(eval_dates)
embargo_bars = int(n * splitter.embargo_pct)
grp = n // splitter.n_groups
print(f"Eval window:        {eval_dates[0].date()} -> {eval_dates[-1].date()}  ({n} bars)")
print(f"Groups:             {splitter.n_groups} contiguous, ~{grp} bars (~{grp / 21:.0f} months) each")
print(f"Splits (trainings): {splitter.n_splits()}")
print(f"Paths (curves):     {splitter.n_paths()}")
print(f"Purge:              {splitter.purge_bars} bars each side of every test run")
print(f"Embargo:            {embargo_bars} bars after each test run "
      f"(trailing buffer = {splitter.purge_bars + embargo_bars})")

splits = list(splitter.split(eval_dates))
first_train, first_test = splits[0]
print(f"First split:        train={len(first_train)}, test={len(first_test)}, "
      f"disjoint={len(first_train.intersection(first_test)) == 0}")
print(f"All splits:         {len(splits)} | "
      f"train sizes {min(len(tr) for tr, _ in splits)}-{max(len(tr) for tr, _ in splits)} | "
      f"test sizes {min(len(te) for _, te in splits)}-{max(len(te) for _, te in splits)}")

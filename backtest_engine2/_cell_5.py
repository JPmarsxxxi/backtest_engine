# Cell 6 — CPCV splitter + MultiPathEngine
from backtest.splitters import CombinatorialPurgedCV
from backtest.engine import MultiPathEngine

# ── Splitter ──────────────────────────────────────────────────────────────────
splitter = CombinatorialPurgedCV(
    n_splits=6,
    n_test_groups=2,
    purge_bars=168,       # 7d × 24h — matches longest feature lookback
    embargo_pct=0.01,
)

n = len(panel.dates)
embargo_bars = int(n * splitter.embargo_pct)
print(f"Timeline            : {n:,} bars  ({panel.dates[0].date()} → {panel.dates[-1].date()})")
print(f"Groups              : {splitter.n_groups} of ~{n // splitter.n_groups:,} bars each (~{n // splitter.n_groups / 24:.0f} days)")
print(f"Splits (trainings)  : {splitter.n_splits()}")
print(f"Paths (curves)      : {splitter.n_paths()}")
print(f"Purge               : {splitter.purge_bars} bars (leading)")
print(f"Embargo             : {embargo_bars} bars (trailing-only; trailing total = {splitter.purge_bars + embargo_bars})")

first_train, first_test = next(iter(splitter.split(panel.dates)))
print(f"First split         : train={len(first_train):,}  test={len(first_test):,}  "
      f"disjoint={len(first_train.intersection(first_test)) == 0}")

# ── MultiPathEngine — factory so each fold starts with clean strategy state ───
strat_factory = lambda: CryptoRegimeOuTrendStrategy()

mp = MultiPathEngine(
    panel,
    strat_factory,
    splitter,
    costs=costs,
    liquidity=liq,
    initial_capital=1_000_000,
)

print(f"\nRunning {splitter.n_splits()} splits (n_jobs=-1) …")
res = mp.run(n_jobs=-1)

print(f"\nSplits run          : {res.n_splits}")
print(f"Paths assembled     : {res.n_paths}")
print(f"path_returns shape  : {res.path_returns.shape}")
print(f"path_equity shape   : {res.path_equity.shape}")
print(f"Date range          : {res.path_returns.index[0].date()} → {res.path_returns.index[-1].date()}")
print(f"Per-path NaN count  : min={res.path_returns.isna().sum().min()},  max={res.path_returns.isna().sum().max()}")
print()
print("Final equity per path:")
print(res.path_equity.iloc[-1].rename("final_equity").map(lambda v: f"${v:,.0f}"))

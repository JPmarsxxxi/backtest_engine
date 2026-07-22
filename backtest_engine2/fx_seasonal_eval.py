"""Evaluate the #016 live forward-test: reads data/fx_seasonal_live_log.parquet and prints live
net bp/day vs the +4.95 backtest target, per-pair/per-session breakdown, realized-vs-assumed spread,
and the PRE-COMMITTED confirm/kill verdict. Run anytime: python fx_seasonal_eval.py"""
import numpy as np
import pandas as pd

LOG = r"C:\Users\User\backtest_engine\backtest_engine2\data\fx_seasonal_live_log.parquet"
TARGET_DAY = 4.95                                   # backtest combined net bp/day
TARGET_PAIR = {"EURUSD": 1.29, "GBPUSD": 1.70, "USDJPY": 1.94}   # backtest net bp/day per pair
ASSUMED_HALF = {"EURUSD": {21: 0.087, 23: 0.086}, "GBPUSD": {21: 0.187, 23: 0.187},
                "USDJPY": {21: 0.156, 23: 0.125}}
EVAL_DAYS = 15

try:
    df = pd.read_parquet(LOG)
except FileNotFoundError:
    print("No trades logged yet. First session fires tonight at 21:00 London.")
    raise SystemExit

df = df[df.get("dry", False) == False].copy()       # live trades only
if df.empty:
    print("Log exists but no LIVE trades yet (only dry rows). First live session 21:00 London.")
    raise SystemExit

# Account-aware pooling: FTMO Free Trial demos expire ~14 days < the ~15 trading days we need, so the
# test necessarily spans MULTIPLE demo accounts. Each round-trip's net bp is independent and all on
# FTMO-Demo spreads, so we POOL the per-leg edge across accounts while reporting days per account.
if "account" not in df.columns:
    df["account"] = pd.NA
df["account"] = df["account"].fillna("1513677687")   # pre-tag rows = the original (now-expired) demo

df["date"] = pd.to_datetime(df["date"])
df["sess_hr"] = df["session"].map({"A": 21, "B": 23})
n_days = df["date"].nunique()
daily = df.groupby("date")["net_bp"].sum()          # combined net bp/day (sum of the day's legs)

print(f"=== #016 LIVE FORWARD-TEST — {len(df)} trades over {n_days} trading days "
      f"({df.date.min().date()} -> {df.date.max().date()}) ===\n")

if df["account"].nunique() > 1:
    print("pooled across demo accounts (Free-Trial expiry forces new accounts):")
    for acct, g in df.groupby("account"):
        print(f"   acct {acct}: {len(g)} trades, {g['date'].nunique()} days, "
              f"net {g.groupby('date')['net_bp'].sum().mean():+.2f} bp/day "
              f"({g.date.min().date()} -> {g.date.max().date()})")
    print()

# headline: combined net bp/day, live vs target
mu, sd = daily.mean(), daily.std()
t = mu / sd * np.sqrt(len(daily)) if len(daily) > 1 and sd > 0 else float("nan")
print(f"COMBINED net/day:  LIVE {mu:+.2f} bp   vs backtest {TARGET_DAY:+.2f} bp   "
      f"({100*mu/TARGET_DAY:.0f}% of target)  | day-hit {100*(daily>0).mean():.0f}% | t {t:+.1f}")
print(f"   gross/day {1e0*df.groupby('date')['gross_bp'].sum().mean():+.2f} bp -> spread cost "
      f"{df.groupby('date')['gross_bp'].sum().mean()-mu:+.2f} bp/day\n")

# per pair
print("per-pair net bp/day (live vs backtest):")
for p, g in df.groupby("symbol"):
    pd_ = g.groupby("date")["net_bp"].sum().mean()
    print(f"   {p}: {pd_:+.2f}  vs {TARGET_PAIR.get(p, float('nan')):+.2f}")

# per (pair,session): per-trade net + realized vs assumed round-trip spread
print("\nper-leg net bp/trade + spread (realized r/t vs assumed):")
for (p, h), g in df.groupby(["symbol", "sess_hr"]):
    rt_real = (g["entry_half_bp"] + g["exit_half_bp"]).mean()
    rt_assumed = 2 * ASSUMED_HALF[p][h]
    flag = "  <-WIDE" if rt_real > 1.5 * rt_assumed else ""
    print(f"   {p} {h:02d}:00  net {g['net_bp'].mean():+.2f} bp/trade (n={len(g)}) | "
          f"spread r/t live {rt_real:.3f} vs assumed {rt_assumed:.3f}{flag}")

# verdict (pre-committed)
print("\n" + "=" * 56)
if n_days < 10:
    print(f"VERDICT: WATCHING — {n_days}/{EVAL_DAYS} days. Need ~{EVAL_DAYS} for a call.")
    print(f"  (running live {mu:+.2f} bp/day; confirm>=3, kill<2)")
elif mu >= 3.0:
    print(f"VERDICT: CONFIRM ✓ — live {mu:+.2f} bp/day >= 3 over {n_days} days.")
    print("  -> proceed to sizing (leverage the Sharpe to ~10% vol) + pre-commit exit, then a real challenge.")
elif mu < 2.0:
    print(f"VERDICT: KILL ✗ — live {mu:+.2f} bp/day < 2 over {n_days} days.")
    print("  -> slippage ate the IC; the backtest was execution-optimistic. Do NOT fund a challenge.")
else:
    print(f"VERDICT: BORDERLINE — live {mu:+.2f} bp/day (2-3 band) over {n_days} days. Keep running.")
print("=" * 56)

# Cell 1 — #018 GEX -> forward index RETURN (Moreira-Muir vol-risk-premium channel).
# Question: does the dealer-gamma VOL forecast predict next-day / next-week index returns?
# Daily series = GEX file's own cash-S&P price (2011-2026, ~3800d; longer & perfectly GEX-aligned
# vs the 2019+ intraday panel). PIT: gex[T] (known after close T) -> forward return measured from T+1.
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ENG = r"C:\Users\User\backtest_engine\backtest_engine2"
g = pd.read_parquet(ENG + r"\data\sqz_dix_gex.parquet").set_index("date").sort_index()

daily = g[["price", "gex"]].copy()
daily["fwd1"] = daily["price"].shift(-1) / daily["price"] - 1          # T+1 return
daily["fwd5"] = daily["price"].shift(-5) / daily["price"] - 1          # T+1..T+5 (weekly)
d = daily.dropna(subset=["gex", "fwd1"]).copy()

print(f"daily rows {len(daily)} | {daily.index.min().date()} -> {daily.index.max().date()}")
print(f"usable (gex & fwd1): {len(d)} | NaN fwd5: {int(d['fwd5'].isna().sum())}")

for h in ["fwd1", "fwd5"]:
    sub = d.dropna(subset=[h])
    rho, p = spearmanr(sub["gex"], sub[h])
    print(f"\n[{h}] spearman(GEX, fwd) = {rho:+.3f} (p={p:.3f})   [>0 = MM/managed-vol; <0 = rebound]")
    sub = sub.assign(q=pd.qcut(sub["gex"], 5, labels=[1, 2, 3, 4, 5]))
    tab = sub.groupby("q", observed=True)[h].agg(["count", "mean"])
    print("  mean fwd by GEX quintile (Q1 low-gamma .. Q5 high-gamma):")
    for q, row in tab.iterrows():
        print(f"    Q{q}: n={int(row['count']):>4}  mean {1e4*row['mean']:+7.1f} bps")

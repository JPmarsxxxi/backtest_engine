# Cell 2 — #018 PIT gate: long the index after low-gamma days (rebound sign from Cell 1).
# PIT expanding-20th-pct low-gamma flag on gex[T] (known close T) -> hold T+1 (1d) / T+1..T+5 (5d).
# Pre-registered: survives iff positive in MAJORITY of years AND positive ex-{2020,2022} AND the
# 1-day hold stays net-positive after swap(long ~1.6bp/nt)+spread(~0.5bp). Cost gate ~2bp for 1d.
exp_p = daily["gex"].expanding(min_periods=120).apply(
    lambda x: float((x.iloc[:-1] < x.iloc[-1]).mean()) if len(x) > 1 else np.nan, raw=False)
sig = daily.assign(low=(exp_p < 0.20))
sig = sig[sig["low"].notna()]

def report(col, nights, label):
    r = sig.loc[sig["low"] == True, col].dropna()
    yrs = sorted(set(r.index.year))
    pos = sum(1 for y in yrs if r[r.index.year == y].mean() > 0)
    net = r.mean() - nights * 1.6e-4 - 0.5e-4
    print(f"\n--- {label}: {len(r)} signals | gross {1e4*r.mean():+.1f} bps | "
          f"NET(swap+spr) {1e4*net:+.1f} bps | hit {100*(r>0).mean():.1f}% | pos yrs {pos}/{len(yrs)}")
    print("    per yr bps:", {int(y): round(1e4 * r[r.index.year == y].mean(), 0) for y in yrs})
    ex = r[~r.index.year.isin([2020, 2022])]
    pos_ex = sum(1 for y in set(ex.index.year) if ex[ex.index.year == y].mean() > 0)
    print(f"    ex-2020&2022: {len(ex)} sig | gross {1e4*ex.mean():+.1f} bps | "
          f"hit {100*(ex>0).mean():.1f}% | pos yrs {pos_ex}/{ex.index.year.nunique()}")

report("fwd1", 1, "1-day hold (long low-gamma)")
report("fwd5", 5, "5-day hold (long low-gamma)")
print(f"\nPIT low-gamma signals total: {int(sig['low'].sum())}")
yr = sig[sig.low == True].index.year.value_counts().sort_index()
print("signals/yr:", {int(k): int(v) for k, v in yr.items()})

"""A/B comparison of the #016 forward test: v1 (fire at HH:00:05) vs v2 (spread-gated, flare-aware entry).
Loads both live logs and shows, per leg and overall: fill/skip rate, realized ENTRY spread, net bp/trade,
net bp/day — so we can read whether v2's entry fix actually recovered the edge. Run: python fx_seasonal_compare.py"""
import numpy as np
import pandas as pd

V1 = r"C:\Users\User\backtest_engine\backtest_engine2\data\fx_seasonal_live_log.parquet"
V2 = r"C:\Users\User\backtest_engine\backtest_engine2\data\fx_seasonal_live_log_v2.parquet"
TARGET_DAY = 4.95                                  # backtest combined net bp/day


def load(path):
    try:
        df = pd.read_parquet(path)
    except FileNotFoundError:
        return None
    df = df[df.get("dry", False) == False].copy()
    if "skipped" not in df.columns:
        df["skipped"] = False
    df["skipped"] = df["skipped"].fillna(False)
    df["sess_hr"] = df["session"].map({"A": 21, "B": 23})
    df["date"] = pd.to_datetime(df["date"])
    return df


def summarize(name, df):
    if df is None or df.empty:
        print(f"\n### {name}: no live trades yet."); return
    filled = df[~df["skipped"]].copy()
    n_skip = int(df["skipped"].sum())
    n_fill = len(filled)
    days = filled["date"].nunique()
    print(f"\n### {name} — {n_fill} fills + {n_skip} skips over {df['date'].nunique()} days "
          f"({df.date.min().date()} -> {df.date.max().date()})")
    if n_fill:
        daily = filled.groupby("date")["net_bp"].sum()
        mu = daily.mean()
        t = mu / daily.std() * np.sqrt(len(daily)) if len(daily) > 1 and daily.std() > 0 else float("nan")
        print(f"   net/day {mu:+.2f} bp ({100*mu/TARGET_DAY:.0f}% of +{TARGET_DAY} target) | "
              f"net/trade {filled['net_bp'].mean():+.2f} bp | gross/trade {filled['gross_bp'].mean():+.2f} bp | t {t:+.1f}")
        if n_skip:
            print(f"   skip rate {100*n_skip/(n_fill+n_skip):.0f}% (legs refused on wide spread)")
    print("   per leg: fills / skips | avg ENTRY half-spread (filled) | net bp/trade")
    for (p, h), g in df.groupby(["symbol", "sess_hr"]):
        gf = g[~g["skipped"]]
        sk = int(g["skipped"].sum())
        ent = gf["entry_half_bp"].mean() if len(gf) else float("nan")
        net = gf["net_bp"].mean() if len(gf) else float("nan")
        print(f"     {p} {h:02d}:00  {len(gf)}f/{sk}s | entry {ent:.3f}bp | net {net:+.2f}bp")


df1, df2 = load(V1), load(V2)
print("=" * 70)
print("#016 FORWARD TEST — A/B: v1 (instant entry)  vs  v2 (spread-gated entry)")
print("=" * 70)
summarize("v1  (fire at HH:00:05)", df1)
summarize("v2  (poll + spread gate)", df2)

if df2 is not None and not df2.empty and df1 is not None and not df1.empty:
    f1 = df1[~df1.get("skipped", False).fillna(False)] if "skipped" in df1 else df1
    f2 = df2[~df2["skipped"]]
    print("\n" + "-" * 70)
    print("READ: if v2 fills at much tighter ENTRY spreads than v1 at the SAME hour, the flare is")
    print("transient and the fix works. If v2 mostly SKIPS the 23:00 legs (spread never settles),")
    print("the hour is structurally expensive — edge is real but untradeable at retail spreads.")

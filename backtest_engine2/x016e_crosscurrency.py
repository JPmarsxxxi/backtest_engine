# #016 cross-currency confirmation: do 21:00/23:00 London show up in GBPUSD & USDJPY?
# Mechanism test: if USD-driven (NY 4pm fix / Asia-open), USD-quote pairs (EUR,GBP) share signs,
# USD-base (JPY) inverts. Expected: 21:00 EUR-/GBP-/JPY+ ; 23:00 EUR+/GBP+/JPY-.
# Signs selected on TRAIN 2019-21, validated OOS 2022-26. Real MT5-measured spreads. Combined 3-pair
# book = breadth play (6 positions/day -> higher IR via sqrt-breadth if pairs not fully correlated).
import numpy as np
import pandas as pd

ENG = r"C:\Users\User\backtest_engine\backtest_engine2"
HALF = {"EURUSD": {21: 0.087, 23: 0.086}, "GBPUSD": {21: 0.187, 23: 0.187},
        "USDJPY": {21: 0.156, 23: 0.125}}
EXPECT = {"EURUSD": {21: -1, 23: +1}, "GBPUSD": {21: -1, 23: +1}, "USDJPY": {21: +1, 23: -1}}  # USD-driven

def load(sym):
    s = pd.read_parquet(rf"{ENG}\data\{sym.lower()}_h1.parquet").set_index("open_time")["close"].sort_index()
    r = s.pct_change()
    r = r[(s.index.to_series().diff() == pd.Timedelta("1h")).values]
    ldn = r.index.tz_convert("Europe/London")
    d = pd.DataFrame({"ret": r.values, "hr": ldn.hour, "yr": ldn.year}, index=r.index)
    d["day"] = pd.to_datetime(pd.Series(ldn.date, index=d.index))
    return d

books = {}
print(f"{'pair':>7} {'hr':>3} {'train bp':>9} {'t':>6} {'sgn':>4} {'USDpred':>8} {'test bp':>9} {'t':>6} {'OOS?':>5}")
for sym in ["EURUSD", "GBPUSD", "USDJPY"]:
    try:
        d = load(sym)
    except FileNotFoundError:
        print(f"{sym}: not pulled yet"); continue
    tr, te = d[d.yr <= 2021], d[d.yr >= 2022]
    signs = {}
    for hh in (21, 23):
        xtr, xte = tr.loc[tr.hr == hh, "ret"], te.loc[te.hr == hh, "ret"]
        ttr = xtr.mean() / xtr.std() * np.sqrt(len(xtr))
        tte = xte.mean() / xte.std() * np.sqrt(len(xte))
        sg = int(np.sign(xtr.mean()))
        signs[hh] = sg
        ok = "OK" if np.sign(xte.mean()) == sg else "FLIP"
        print(f"{sym:>7} {hh:>3} {1e4*xtr.mean():>+8.2f} {ttr:>+6.1f} {sg:>+4d} {EXPECT[sym][hh]:>+8d} "
              f"{1e4*xte.mean():>+8.2f} {tte:>+6.1f} {ok:>5}")
    # per-pair OOS book on measured spread
    bk = te[te.hr.isin([21, 23])].copy()
    bk["sret"] = [signs[h] for h in bk.hr] * bk["ret"]
    bk["cost"] = [2 * HALF[sym][h] / 1e4 for h in bk.hr]
    dg = bk.groupby("day")["sret"].sum()
    dc = bk.groupby("day")["cost"].sum()
    books[sym] = (dg - dc).dropna()

def sh(x):
    return x.mean() / x.std() * np.sqrt(252)

print("\nper-pair OOS 2022-26 net (measured spreads):")
for sym, nb in books.items():
    print(f"  {sym}: net Sh {sh(nb):+.2f} | {1e4*nb.mean():+.2f} bp/day | hit {100*(nb>0).mean():.0f}% | ann {nb.mean()*252*100:+.1f}%")

if len(books) >= 2:
    comb = pd.concat(books.values(), axis=1).dropna().sum(axis=1)   # equal-weight combined
    print(f"\nCOMBINED {len(books)}-pair book (equal wt, {len(comb)} days): net Sh {sh(comb):+.2f} | "
          f"{1e4*comb.mean():+.2f} bp/day | hit {100*(comb>0).mean():.0f}% | ann {comb.mean()*252*100:+.1f}%")
    # pairwise corr of the per-pair net streams (breadth quality)
    cc = pd.concat(books.values(), axis=1, keys=books.keys()).dropna().corr()
    print("  cross-pair net-return corr:\n" + cc.round(2).to_string())
    for yr, g in comb.groupby(comb.index.year):
        print(f"    {yr}: net Sh {sh(g):+.2f} | {1e4*g.mean():+.2f} bp/day")

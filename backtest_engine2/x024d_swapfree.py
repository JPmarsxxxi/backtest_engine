"""
x024d — re-test the COT-extremes book with SWAP REMOVED (swap-free account question).
Charge ONLY spread (on position changes); set swap=0. Then re-run drop-one to see if
the oil-concentration problem survives swap removal (it's structural, not a cost issue).
"""
import numpy as np
import pandas as pd

cot = pd.read_parquet("data/cot_legacy.parquet")
px  = pd.read_parquet("data/ftmo_daily.parquet")[["date", "symbol", "close"]]
spec = pd.read_parquet("data/ftmo_specs.parquet").set_index("symbol")

INVERT = {"USDJPY", "USDCHF", "USDCAD"}
PXMAP  = {"USOIL.cash": "USOIL", "US500.cash": "US500", "US2000.cash": None}
ROLL = 156

def rt_bp(pxsym):
    if pxsym not in spec.index: return 4.0
    r = spec.loc[pxsym]; price = r["price"]
    if not (price and price > 0): return 2.0
    half = (r["spread_pts"] * r["point"] / price) * 1e4
    return max(2 * max(half, 0.0), 1.0)

results = {}
for sym, c in cot.groupby("symbol"):
    pxsym = PXMAP.get(sym, sym)
    if pxsym is None: continue
    c = c.sort_values("report_date").copy()
    oi = c["open_interest_all"].replace(0, np.nan)
    nsp = (c["noncomm_positions_long_all"] - c["noncomm_positions_short_all"]) / oi
    z = (nsp - nsp.rolling(ROLL, min_periods=104).mean()) / nsp.rolling(ROLL, min_periods=104).std()
    sign = -1.0 if sym in INVERT else 1.0
    c["z"] = z.values; c["sign"] = sign
    p = px[px.symbol == pxsym][["date","close"]].dropna().sort_values("date").reset_index(drop=True)
    if len(p) < 200: continue
    a = pd.merge_asof(c[["release_date","z","sign"]].sort_values("release_date"),
                      p.rename(columns={"date":"entry_date","close":"entry_px"}),
                      left_on="release_date", right_on="entry_date",
                      direction="forward", tolerance=pd.Timedelta(days=7)).dropna(subset=["entry_px","z"])
    a["exit_px"] = a["entry_px"].shift(-1)
    a["fwd_ret"] = a["exit_px"]/a["entry_px"] - 1.0
    a = a.dropna(subset=["fwd_ret"]); a["sym"] = sym; a["rt"] = rt_bp(pxsym)
    results[sym] = a

def sharpe(x):
    x = x.dropna()
    return np.nan if (len(x) < 20 or x.std() == 0) else x.mean()/x.std()*np.sqrt(52)

def book(thr, lo, swap_free=True, drop=None):
    parts = []
    for sym, a in results.items():
        if sym == drop: continue
        a = a.copy()
        a["pos"] = np.where(np.abs(a["z"]) > thr, -np.sign(a["z"])*a["sign"], 0.0)
        dpos = a["pos"].diff().abs().fillna(a["pos"].abs())
        cost = dpos * (a["rt"]/2)/1e4            # spread only; swap=0
        a["net"] = a["pos"]*a["fwd_ret"] - cost
        parts.append(a[["entry_date","net"]])
    allp = pd.concat(parts)
    if lo: allp = allp[allp.entry_date >= lo]
    return allp.groupby("entry_date")["net"].mean()

print("=== SWAP-FREE net Sharpe (spread only, no swap) ===")
print(f"{'thr':>5} | {'full':>6} {'2015+':>6} {'2020+':>6} {'2022+':>6}   mean_bp/wk(2020+)")
for thr in [1.0, 1.5, 2.0, 2.5]:
    row = [sharpe(book(thr, lo)) for lo in [None,"2015-01-01","2020-01-01","2022-01-01"]]
    mb = book(thr, "2020-01-01").mean()*1e4
    print(f"{thr:>5} | " + " ".join(f"{v:+6.2f}" for v in row) + f"      {mb:+6.1f}")

print("\n=== SWAP-FREE drop-one (thr=1.5, 2020+) — does oil still carry it? ===")
print(f"  baseline (all):  {sharpe(book(1.5,'2020-01-01')):+.2f}")
for d in results:
    print(f"  w/o {d:12s}: {sharpe(book(1.5,'2020-01-01',drop=d)):+.2f}")

# crude oil standalone, swap-free
oil = results.get("USOIL.cash")
a = oil.copy(); a["pos"] = np.where(np.abs(a["z"])>1.5, -np.sign(a["z"]), 0.0)
a = a[a.entry_date>="2020-01-01"]
print(f"\n  USOIL alone (swap-free, thr1.5, 2020+): Sharpe {sharpe(a['pos']*a['fwd_ret']):+.2f}, "
      f"trades={int((a['pos']!=0).sum())}")

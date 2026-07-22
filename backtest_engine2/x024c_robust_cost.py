"""
x024c — kill-tests for the COT-extremes lead:
  (1) threshold insensitivity: sweep ZTHR in {1.0,1.5,2.0,2.5}
  (2) COST GATE: charge round-trip spread on every position CHANGE + weekly swap drag.
  (3) drop-one-instrument (is the book carried by 1-2 names?)
All on the modern window where the signal lives. GROSS already shown +0.8 (2020+).
"""
import numpy as np
import pandas as pd

cot = pd.read_parquet("data/cot_legacy.parquet")
px  = pd.read_parquet("data/ftmo_daily.parquet")[["date", "symbol", "close"]]
spec = pd.read_parquet("data/ftmo_specs.parquet").set_index("symbol")

INVERT = {"USDJPY", "USDCHF", "USDCAD"}
PXMAP  = {"USOIL.cash": "USOIL", "US500.cash": "US500", "US2000.cash": None}
ROLL = 156

# round-trip spread cost (bp) per position change, and weekly swap drag (bp), per CFD.
# spread: from ftmo_specs spread_pts*point / price; doubled for round trip; floors for realism.
# swap : worst-side daily swap pts -> bp/day * 7. Use the side we'd most often hold (avg of |long|,|short|).
def costs_bp(pxsym):
    if pxsym not in spec.index:
        return 4.0, 3.0
    r = spec.loc[pxsym]
    price = r["price"] if r["price"] > 0 else np.nan
    pt = r["point"]; sp = r["spread_pts"]
    half = (sp * pt / price) * 1e4 if price == price and price > 0 else 1.0
    rt = max(2 * max(half, 0.0), 1.0)            # round-trip bp, floor 1bp
    # swap: pts/day -> price move; convert to bp of notional. swap pts are in quote ccy per lot.
    # approximate weekly swap as |avg daily swap side| * point / price *1e4 * 7
    sw_day = (abs(r["swap_long"]) + abs(r["swap_short"])) / 2.0
    sw_bp = (sw_day * pt / price) * 1e4 * 7 if price == price and price > 0 else 5.0
    return rt, sw_bp

results = {}
for sym, c in cot.groupby("symbol"):
    pxsym = PXMAP.get(sym, sym)
    if pxsym is None:
        continue
    c = c.sort_values("report_date").copy()
    oi = c["open_interest_all"].replace(0, np.nan)
    nsp = (c["noncomm_positions_long_all"] - c["noncomm_positions_short_all"]) / oi
    z = (nsp - nsp.rolling(ROLL, min_periods=104).mean()) / nsp.rolling(ROLL, min_periods=104).std()
    sign = -1.0 if sym in INVERT else 1.0
    c["z"] = z.values; c["sign"] = sign
    p = px[px.symbol == pxsym][["date", "close"]].dropna().sort_values("date").reset_index(drop=True)
    if len(p) < 200:
        continue
    a = pd.merge_asof(c[["release_date","z","sign"]].sort_values("release_date"),
                      p.rename(columns={"date":"entry_date","close":"entry_px"}),
                      left_on="release_date", right_on="entry_date",
                      direction="forward", tolerance=pd.Timedelta(days=7)).dropna(subset=["entry_px","z"])
    a["exit_px"] = a["entry_px"].shift(-1)
    a["fwd_ret"] = a["exit_px"]/a["entry_px"] - 1.0
    a = a.dropna(subset=["fwd_ret"])
    a["sym"] = sym; a["pxsym"] = pxsym
    rt, sw = costs_bp(pxsym)
    a["rt_bp"] = rt; a["sw_bp"] = sw
    results[sym] = a

def sharpe(x):
    x = x.dropna()
    return np.nan if (len(x) < 20 or x.std() == 0) else x.mean()/x.std()*np.sqrt(52)

def book_pnl(thr, lo, net=False):
    """equal-weight book pnl per week at threshold thr; net charges spread on pos change + swap."""
    parts = []
    for sym, a in results.items():
        a = a.copy()
        a["pos"] = np.where(np.abs(a["z"]) > thr, -np.sign(a["z"])*a["sign"], 0.0)
        a["gross"] = a["pos"] * a["fwd_ret"]
        dpos = a["pos"].diff().abs().fillna(a["pos"].abs())
        cost = dpos * (a["rt_bp"]/2)/1e4 + (a["pos"].abs() * a["sw_bp"]/1e4)  # half-rt per change side + swap when in
        a["net"] = a["gross"] - cost
        parts.append(a[["entry_date","gross","net","pos","sym"]])
    allp = pd.concat(parts)
    if lo: allp = allp[allp.entry_date >= lo]
    col = "net" if net else "gross"
    # equal-weight across active instruments
    port = allp.groupby("entry_date")[col].mean()
    return port

print("=== (1) THRESHOLD SWEEP — gross Sharpe ===")
print(f"{'thr':>5} | {'full':>6} {'2015+':>6} {'2020+':>6} {'2022+':>6}")
for thr in [1.0, 1.5, 2.0, 2.5]:
    row = [sharpe(book_pnl(thr, lo)) for lo in [None,"2015-01-01","2020-01-01","2022-01-01"]]
    print(f"{thr:>5} | " + " ".join(f"{v:+6.2f}" for v in row))

print("\n=== (2) COST GATE — NET Sharpe (spread on pos-change + weekly swap) ===")
print(f"{'thr':>5} | {'full':>6} {'2015+':>6} {'2020+':>6} {'2022+':>6}   mean_net_bp/wk(2020+)")
for thr in [1.0, 1.5, 2.0, 2.5]:
    row = [sharpe(book_pnl(thr, lo, net=True)) for lo in [None,"2015-01-01","2020-01-01","2022-01-01"]]
    mb = book_pnl(thr, "2020-01-01", net=True).mean()*1e4
    print(f"{thr:>5} | " + " ".join(f"{v:+6.2f}" for v in row) + f"      {mb:+6.1f}")

print("\nPer-instrument cost assumptions (round-trip spread bp, weekly swap bp):")
for sym, a in sorted(results.items()):
    print(f"  {a['pxsym'].iloc[0]:10s} rt={a['rt_bp'].iloc[0]:5.2f}bp  swap/wk={a['sw_bp'].iloc[0]:5.2f}bp")

print("\n=== (3) DROP-ONE-INSTRUMENT (thr=1.5, 2020+, NET) — book Sharpe without each ===")
base = sharpe(book_pnl(1.5, "2020-01-01", net=True))
print(f"  baseline (all): {base:+.2f}")
syms = list(results.keys())
for drop in syms:
    parts = []
    for sym, a in results.items():
        if sym == drop: continue
        a = a.copy()
        a["pos"] = np.where(np.abs(a["z"]) > 1.5, -np.sign(a["z"])*a["sign"], 0.0)
        dpos = a["pos"].diff().abs().fillna(a["pos"].abs())
        a["net"] = a["pos"]*a["fwd_ret"] - (dpos*(a["rt_bp"]/2)/1e4 + a["pos"].abs()*a["sw_bp"]/1e4)
        parts.append(a[["entry_date","net"]])
    allp = pd.concat(parts); allp = allp[allp.entry_date >= "2020-01-01"]
    s = sharpe(allp.groupby("entry_date")["net"].mean())
    print(f"  w/o {drop:12s}: {s:+.2f}")

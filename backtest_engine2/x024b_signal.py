"""
x024b — COT positioning signal, raw backtest + DECAY-FIRST gate.

Signal (per instrument, weekly):
  net_spec      = noncomm_long - noncomm_short
  net_spec_pct  = net_spec / open_interest
  z             = (net_spec_pct - rolling_mean_156w) / rolling_std_156w   # 3yr trailing, causal
  position      = -tanh(z) * sign_align                                   # CONTRARIAN: fade extremes

Entry at first CFD close >= release_date (Tue positions made public the Fri after = +3d).
Hold ~1 week to the next release entry. Forward weekly return from actual CFD closes.
Equal-weight across instruments = diversified positioning book.

Reports GROSS Sharpe full-sample AND recent sub-windows (decay-first). Costs come later
ONLY if the gross effect survives recent OOS.
"""
import numpy as np
import pandas as pd

cot = pd.read_parquet("data/cot_legacy.parquet")
px  = pd.read_parquet("data/ftmo_daily.parquet")[["date", "symbol", "close"]]

# CFD symbols whose futures positioning is INVERTED vs the FTMO quote (USD is the base ccy)
INVERT = {"USDJPY", "USDCHF", "USDCAD"}
# COT symbol -> ftmo_daily price symbol (fix .cash suffix); None = no CFD price, skip
PXMAP = {"USOIL.cash": "USOIL", "US500.cash": "US500", "US2000.cash": None}

ROLL = 156   # 3yr trailing window (weeks)
ZTHR = 1.5   # "extreme" positioning threshold for the extremes variant
results = {}   # symbol -> DataFrame[release, z, pos, fwd_ret]

for sym, c in cot.groupby("symbol"):
    pxsym = PXMAP.get(sym, sym)
    if pxsym is None:
        continue
    c = c.sort_values("report_date").copy()
    oi = c["open_interest_all"].replace(0, np.nan)
    net_spec_pct = (c["noncomm_positions_long_all"] - c["noncomm_positions_short_all"]) / oi
    m = net_spec_pct.rolling(ROLL, min_periods=104).mean()
    s = net_spec_pct.rolling(ROLL, min_periods=104).std()
    z = (net_spec_pct - m) / s
    sign = -1.0 if sym in INVERT else 1.0
    pos = -np.tanh(z) * sign           # contrarian, sign-aligned to the CFD
    c["z"] = z.values
    c["pos"] = pos.values
    # extremes variant: trade only crowded positioning, flat otherwise
    c["pos_ext"] = np.where(np.abs(z) > ZTHR, -np.sign(z) * sign, 0.0)

    # align to CFD price by release_date (forward to next available close)
    p = (px[px.symbol == pxsym][["date", "close"]]
         .dropna().sort_values("date").reset_index(drop=True))
    if len(p) < 200:
        continue
    a = pd.merge_asof(
        c[["release_date", "z", "pos", "pos_ext"]].sort_values("release_date"),
        p.rename(columns={"date": "entry_date", "close": "entry_px"}),
        left_on="release_date", right_on="entry_date", direction="forward",
        tolerance=pd.Timedelta(days=7),
    ).dropna(subset=["entry_px", "pos"])
    a["exit_px"] = a["entry_px"].shift(-1)         # exit = next week's entry
    a["fwd_ret"] = a["exit_px"] / a["entry_px"] - 1.0
    a = a.dropna(subset=["fwd_ret"])
    a["pnl"] = a["pos"] * a["fwd_ret"]
    a["pnl_ext"] = a["pos_ext"] * a["fwd_ret"]
    a["sym"] = sym
    results[sym] = a

allp = pd.concat(results.values(), ignore_index=True)

def sharpe(x):
    x = x.dropna()
    if x.std() == 0 or len(x) < 20:
        return np.nan
    return x.mean() / x.std() * np.sqrt(52)   # weekly -> annualized

def report(label, df):
    # equal-weight portfolio: average pnl across instruments per entry week
    port = df.groupby("entry_date")["pnl"].mean()
    n = df.groupby("entry_date").size().mean()
    print(f"{label:18s} wks={port.shape[0]:4d}  avg_instr={n:4.1f}  "
          f"gross_Sh={sharpe(port):+5.2f}  mean_bp/wk={port.mean()*1e4:+6.1f}  "
          f"hit={ (port>0).mean()*100:4.1f}%")
    return port

print("=== CONTRARIAN COT (fade specs), equal-weight book, GROSS, no costs ===")
windows = {
    "full":   (None, None),
    ">=2010": ("2010-01-01", None),
    ">=2015": ("2015-01-01", None),
    ">=2020": ("2020-01-01", None),
    "2022+":  ("2022-01-01", None),
}
for lab, (lo, hi) in windows.items():
    d = allp.copy()
    if lo: d = d[d.entry_date >= lo]
    report(lab, d)

print("\n=== Per-instrument GROSS Sharpe (full / 2015+) ===")
for sym, a in sorted(results.items()):
    full = sharpe(a["pnl"])
    rec  = sharpe(a[a.entry_date >= "2015-01-01"]["pnl"])
    print(f"  {sym:12s} full={full:+5.2f}   2015+={rec:+5.2f}   wks={len(a)}")

# also: continuation (book's literal 'long rising spec OI') = opposite sign, recent only
print("\n=== ALT sign — CONTINUATION (follow specs), 2015+ ===")
cont = allp.copy(); cont["pnl"] = -cont["pnl"]
report("continuation 2015+", cont[cont.entry_date >= "2015-01-01"])

# ---- EXTREMES variant: the actual hypothesis (trade only crowded |z|>1.5) ----
def report_ext(label, df):
    df = df[df.pos_ext != 0]
    if df.empty:
        print(f"{label:18s}  (no extreme weeks)"); return
    port = df.groupby("entry_date")["pnl_ext"].mean()
    print(f"{label:18s} extreme_wks={port.shape[0]:4d}  "
          f"gross_Sh={sharpe(port):+5.2f}  mean_bp/wk={port.mean()*1e4:+6.1f}  "
          f"hit={(port>0).mean()*100:4.1f}%")

print("\n=== EXTREMES contrarian (|z|>1.5 only), equal-weight, GROSS ===")
for lab, (lo, hi) in windows.items():
    d = allp.copy()
    if lo: d = d[d.entry_date >= lo]
    report_ext(lab, d)

print("\n=== EXTREMES per-instrument (full / 2020+) ===")
for sym, a in sorted(results.items()):
    e = a[a.pos_ext != 0]
    full = sharpe(e["pnl_ext"])
    rec  = sharpe(e[e.entry_date >= "2020-01-01"]["pnl_ext"])
    print(f"  {sym:12s} full={full:+5.2f} (n={len(e):4d})   2020+={rec:+5.2f} (n={len(e[e.entry_date>='2020-01-01']):3d})")

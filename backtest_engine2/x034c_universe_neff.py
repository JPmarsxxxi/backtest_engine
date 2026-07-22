"""#034c — the maximal swap-dodged CROSS-ASSET overnight universe on FTMO: one number, N_eff.

Route-1 diversification test (alpha log #034b): indices alone = N_eff 1.9. How many INDEPENDENT
overnight bets can this venue host if we open the 18:00-NY cross-section to every instrument group
(FX, exotics, metals, energy, ags, crypto, indices — everything except the dead Equities-I)?

Two stages, all measurement, no signal work:
  1. profile every symbol (x034 profiler): does it QUOTE at 18:00 NY (>=3 of last 12 days) and at
     what half-spread? plus the 09:30-NY-exit half.
  2. survivors: overnight returns 18:00 -> next 09:30 NY from M5 bars (x034b logic), correlation
     matrix -> N_eff = (sum lambda)^2 / sum(lambda^2), per-group medians, sigma_XS, per-name
     break-even IC = round-trip toll / sigma_XS.
Outputs data/x034c_universe.parquet (overnight return matrix) + printed verdict. BID-bar caveat as
in x034b (fine for dispersion/corr estimates; per-name drift claims need tick mids).
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

E = r"C:\Users\User\backtest_engine\backtest_engine2"
D = os.path.join(E, "data")
LOGFILE = os.path.join(D, "x034c.out")
if sys.stdout is None:
    sys.stdout = sys.stderr = open(LOGFILE, "a", encoding="utf-8")

import MetaTrader5 as mt5


def load_mod(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(E, f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


x034 = load_mod("x034_equity_spread_profile")
x034b = load_mod("x034b_index_dispersion")

MIN_18H_DAYS = 3
MAX_HALF_BP = 6.0            # legs pricier than this can't break even on any sane IC


def run():
    if not mt5.initialize():
        print("MT5 init failed")
        return 1
    off = x034.server_offset_h()
    spec = pd.read_parquet(os.path.join(D, "ftmo_specs.parquet"))
    spec["grp"] = spec.path.str.split(chr(92)).str[0]
    univ = spec[~spec.grp.eq("Equities I CFD")]          # stocks measured dead at 18:00 NY (x034)
    print(f"stage 1 — profiling {len(univ)} symbols for 18:00-NY quoting + spread...")
    rows = []
    for _, r in univ.iterrows():
        p = x034.profile(r.symbol, off)
        if p is None:
            continue
        g = p.set_index("ny_hour")
        d18 = int(g.loc[18, "days_seen"]) if 18 in g.index else 0
        h18 = float(g.loc[18, "median_half_bp"]) if 18 in g.index else np.nan
        h9 = float(g.loc[9, "median_half_bp"]) if 9 in g.index else np.nan
        rows.append({"symbol": r.symbol, "grp": r.grp, "h18_days": d18,
                     "h18_half_bp": h18, "h9_half_bp": h9})
    prof = pd.DataFrame(rows)
    alive = prof[(prof.h18_days >= MIN_18H_DAYS) & (prof.h18_half_bp <= MAX_HALF_BP)]
    print(f"   quoting at 18:00 NY on >={MIN_18H_DAYS}d AND half<= {MAX_HALF_BP}bp: "
          f"{len(alive)}/{len(prof)} profiled")
    print(alive.groupby("grp").size().to_string())

    print("\nstage 2 — overnight returns (18:00 -> 09:30 NY), correlation, N_eff...")
    R = {}
    for sym in alive.symbol:
        r = x034b.overnight_returns(sym, off)
        if r is not None and len(r) >= 60:
            R[sym] = r
    mt5.shutdown()
    M = pd.DataFrame(R).dropna()
    M.to_parquet(os.path.join(D, "x034c_universe.parquet"))
    print(f"   matrix: {M.shape[1]} instruments x {len(M)} common nights "
          f"({M.index.min().date()} -> {M.index.max().date()})")

    corr = M.corr()
    lam = np.linalg.eigvalsh(corr.values)
    n_eff = lam.sum() ** 2 / (lam ** 2).sum()
    sigma_xs = M.sub(M.mean(axis=1), axis=0).std(axis=1)
    grp = alive.set_index("symbol").grp
    print(f"\n=== FULL UNIVERSE: N_eff = {n_eff:.1f}  (of {M.shape[1]} instruments) ===")
    print(f"sigma_XS median {sigma_xs.median():.1f}bp (P25 {sigma_xs.quantile(0.25):.1f} / "
          f"P75 {sigma_xs.quantile(0.75):.1f})")
    # per-group internal N_eff + cross-group corr
    print(f"\n{'group':>16} {'n':>3} {'N_eff':>6} {'med half18':>10}")
    for gname, syms in grp.groupby(grp):
        cols = [s for s in syms.index if s in M.columns]
        if len(cols) < 1:
            continue
        if len(cols) == 1:
            ne = 1.0
        else:
            l = np.linalg.eigvalsh(M[cols].corr().values)
            ne = l.sum() ** 2 / (l ** 2).sum()
        print(f"{gname:>16} {len(cols):>3} {ne:>6.1f} "
              f"{alive.set_index('symbol').loc[cols, 'h18_half_bp'].median():>10.2f}")
    med = sigma_xs.median()
    a = alive.set_index("symbol")
    a["toll"] = a.h18_half_bp + a.h9_half_bp.fillna(a.h18_half_bp)
    a = a.loc[[s for s in M.columns]]
    a["breakeven_IC"] = a.toll / med
    ok = a[a.breakeven_IC <= 0.05]
    print(f"\nlegs with break-even IC <= 0.05: {len(ok)}/{len(a)}")
    print(ok.sort_values("breakeven_IC")[["grp", "h18_half_bp", "toll", "breakeven_IC"]]
          .round(3).to_string())
    print(f"\nread: N_eff is the breadth a signal gets multiplied by (sqrt). Indices-only was 1.9.")
    return 0


if __name__ == "__main__":
    sys.exit(run())

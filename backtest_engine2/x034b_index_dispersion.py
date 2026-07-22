"""#034b — is there anything for a cross-sectional INDEX book to fight over? (follow-up to the x034
equity kill; user's counter: FTMO has multiple indices and 7 of them DO quote at 18:00 NY.)

Measures, from MT5 M5 bars (~as far back as the terminal serves): per NY night, the overnight return
of each index 18:00 -> next 09:30 (the #023 swap-dodge window), then the CROSS-SECTIONAL dispersion
across the 7 names each night = the raw material a ranking signal could capture. Compares it to the
measured tolls (x034 profiler: entry half at 18:00 + exit half at cash open) via the capture math:
    gross/name/night ≈ IC × σ_XS   →   IC needed = toll / σ_XS
If the IC needed is fantasy (>0.1 for a daily signal), the index version dies at the gate too.
Also reports avg pairwise correlation and effective N (breadth quality: 7 quoted names ≠ 7 bets).

HYGIENE CAVEAT (stated, quantified): MT5 bars are BID-based. Contamination = spread CHANGE between
the two snapshots, bounded by the halves involved (0.25-2.4bp) — small vs overnight index moves
(tens of bp) and roughly symmetric across names; fine for a dispersion ESTIMATE, would NOT be fine
for measuring a per-name drift edge (that later step needs tick mids, x034-style).
"""
import sys
import time as _t
from datetime import datetime, timedelta, timezone

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

SYMS = ["US30.cash", "US100.cash", "US500.cash", "US2000.cash",
        "GER40.cash", "JP225.cash", "UK100.cash"]
# measured half-spreads bp (x034 profiler 2026-07-16): entry at 18:00 NY, exit ~cash session
HALF18 = {"US30.cash": 0.25, "US100.cash": 0.31, "US500.cash": 0.40, "US2000.cash": 2.10,
          "GER40.cash": 0.66, "JP225.cash": 1.20, "UK100.cash": 2.45}
HALFCASH = {"US30.cash": 0.20, "US100.cash": 0.25, "US500.cash": 0.40, "US2000.cash": 1.58,
            "GER40.cash": 0.27, "JP225.cash": 0.74, "UK100.cash": 0.69}


def server_offset_h():
    mt5.symbol_select("EURUSD", True)
    t = mt5.symbol_info_tick("EURUSD")
    return round((t.time - _t.time()) / 3600)


def overnight_returns(sym, off_h, days=500):
    mt5.symbol_select(sym, True)
    # copy_rates_range silently returns 0 for far-past starts; from_pos serves the full depth
    bars = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M5, 0, 99999)
    if bars is None or len(bars) == 0:
        return None
    df = pd.DataFrame(bars)
    utc = pd.to_datetime(df["time"] - off_h * 3600, unit="s", utc=True)
    ny = utc.dt.tz_convert("America/New_York")
    df["hm"] = ny.dt.hour * 100 + ny.dt.minute
    df["day"] = ny.dt.date
    e = df[df.hm == 1800].groupby("day")["open"].first()          # price at 18:00 NY
    x = df[df.hm == 930].groupby("day")["open"].first()           # price at 09:30 NY
    e.index = pd.to_datetime(e.index)
    x.index = pd.to_datetime(x.index)
    x_next = x.copy()
    x_next.index = x_next.index - pd.Timedelta(days=1)            # align next-morning exit to entry day
    # handle Fri->Mon: also try -3d for unmatched
    r = (x_next / e - 1).dropna()
    fri = e.index.difference(r.index)
    if len(fri):
        x3 = x.copy(); x3.index = x3.index - pd.Timedelta(days=3)
        r = pd.concat([r, (x3.reindex(fri) / e.reindex(fri) - 1).dropna()]).sort_index()
    return r * 1e4                                                # bp


def run():
    if not mt5.initialize():
        print("MT5 init failed"); return 1
    off = server_offset_h()
    R = {}
    for s in SYMS:
        r = overnight_returns(s, off)
        if r is None or len(r) < 60:
            print(f"{s}: insufficient bars ({0 if r is None else len(r)})")
            continue
        R[s] = r
    mt5.shutdown()
    M = pd.DataFrame(R).dropna()
    print(f"\n#034b index overnight (18:00->09:30 NY) cross-section — {len(M)} nights, "
          f"{M.index.min().date()} -> {M.index.max().date()}, {M.shape[1]} indices")
    print(f"\nper-index overnight return (bp/night): mean | sd")
    for s in M.columns:
        print(f"   {s:>12}: {M[s].mean():+6.2f} | {M[s].std():6.1f}")
    corr = M.corr()
    avg_corr = corr.values[np.triu_indices_from(corr, 1)].mean()
    lam = np.linalg.eigvalsh(corr.values)
    n_eff = lam.sum() ** 2 / (lam ** 2).sum()
    xs = M.sub(M.mean(axis=1), axis=0)                            # demeaned = what a $-neutral book trades
    sigma_xs = xs.std(axis=1)
    print(f"\navg pairwise corr: {avg_corr:+.2f}   ->  effective N: {n_eff:.1f} (of {M.shape[1]})")
    print(f"cross-sectional dispersion sigma_XS per night: median {sigma_xs.median():.1f}bp "
          f"(P25 {sigma_xs.quantile(0.25):.1f} / P75 {sigma_xs.quantile(0.75):.1f})")
    print(f"\ncapture math (gross/name/night ≈ IC × sigma_XS, vs round-trip toll):")
    med = sigma_xs.median()
    for s in M.columns:
        toll = HALF18[s] + HALFCASH[s]
        print(f"   {s:>12}: toll {toll:4.2f}bp  -> IC needed just to BREAK EVEN: {toll/med:.3f}")
    print(f"\nread: a very good daily XS signal has IC ~0.02-0.05. If break-even IC is above that,")
    print(f"the leg cannot pay its toll no matter the signal. (And N_eff caps the breadth boost.)")
    return 0


if __name__ == "__main__":
    sys.exit(run())

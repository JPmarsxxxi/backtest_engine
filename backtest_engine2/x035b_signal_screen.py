"""#035b — first signal screen on the swap-dodged cross-asset overnight panel (x035).

Candidates (all known at the 18:00-NY entry; all rank/XS; families per the standing rules —
fast trend on FX/indices is BANNED (tick-size rule), so momentum appears only as a REFERENCE row
expected to fail):
    rev1     fade the last 24h move  (XS reversal, 1d)
    rev5     fade the last 5d move   (XS reversal, slower)
    daynight fade TODAY's session move (day<->night reversal, the equity-lore signal)
    seas     trailing 60-night mean overnight return (habitual overnight drifters; #016's family)
    carry    static swap-implied carry rank from ftmo_specs (interest differential expressed
             overnight; snapshot-of-today caveat: mildly anachronistic for a 1yr screen — flag,
             don't trust alone)
    mom20    20d continuation — REFERENCE ONLY (expect dead/wrong-signed)

Metrics: nightly Spearman IC (mean, t, by half), and a toll-charged book: long top-1/3, short
bottom-1/3, weights 1/trailing-vol, cost = per-name round-trip toll (x034c halves) on each night's
traded legs. BID-bar + screen-grade caveats apply (data-hygiene.md): a survivor here earns a proper
two-sided validation, not deployment.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

E = r"C:\Users\User\backtest_engine\backtest_engine2"
D = os.path.join(E, "data")


def load():
    p = pd.read_parquet(os.path.join(D, "x035_nightly_panel.parquet"))
    on = p.pivot(index="day", columns="symbol", values="on_ret")
    p18 = p.pivot(index="day", columns="symbol", values="p18")
    idr = p.pivot(index="day", columns="symbol", values="id_ret")
    on = on[pd.to_datetime(on.index).dayofweek < 5]
    p18, idr = p18.reindex(on.index), idr.reindex(on.index)
    spec = pd.read_parquet(os.path.join(D, "ftmo_specs.parquet")).set_index("symbol")
    carry = {}
    for s in on.columns:
        r = spec.loc[s]
        carry[s] = (-r.swap_long * r.point / r.price * 1e4) if r.price > 0 else np.nan
    carry = pd.Series(carry)
    # tolls: entry half at 18:00 + exit half at 09:30 (x034c profile)
    prof = pd.read_parquet(os.path.join(D, "ftmo_equity_spread_profile.parquet"))  # equities only; redo generic
    return on, p18, idr, carry


def tolls():
    """per-name round-trip toll bp from the x034c stage-1 profile if present, else re-derive
    crude tolls from spec spread_pts (fallback)."""
    f = os.path.join(D, "x034c_tolls.parquet")
    if os.path.exists(f):
        return pd.read_parquet(f)["toll"]
    spec = pd.read_parquet(os.path.join(D, "ftmo_specs.parquet")).set_index("symbol")
    t = (spec.spread_pts * spec.point / spec.price * 1e4).replace([np.inf, -np.inf], np.nan)
    return t


def ic_row(sig, on, name):
    ics = []
    for d in on.index:
        s, r = sig.loc[d], on.loc[d]
        ok = s.notna() & r.notna()
        if ok.sum() >= 20:
            ics.append(spearmanr(s[ok], r[ok])[0])
    ics = pd.Series(ics)
    h1, h2 = ics[: len(ics) // 2], ics[len(ics) // 2:]
    return dict(signal=name, n_nights=len(ics), IC=ics.mean(),
                t=ics.mean() / ics.std() * np.sqrt(len(ics)),
                IC_h1=h1.mean(), IC_h2=h2.mean())


def book_row(sig, on, toll, name):
    vol = on.rolling(20, min_periods=10).std().shift(1)
    grosses, nets = [], []
    for d in on.index:
        s, r, v = sig.loc[d], on.loc[d], vol.loc[d]
        ok = s.notna() & r.notna() & v.notna() & (v > 0)
        if ok.sum() < 20:
            continue
        s, r, v = s[ok], r[ok], v[ok]
        rk = s.rank(pct=True)
        w = pd.Series(0.0, index=s.index)
        w[rk >= 2 / 3] = 1.0
        w[rk <= 1 / 3] = -1.0
        w = w / v
        w[w > 0] /= w[w > 0].sum()
        w[w < 0] /= -w[w < 0].sum()          # $1 long / $1 short
        c = (w.abs() * toll.reindex(s.index).fillna(2.0)).sum() / 2   # toll on each unit traded
        grosses.append((w * r).sum() / 2)     # per $1 gross book
        nets.append((w * r).sum() / 2 - c)
    g, n = pd.Series(grosses), pd.Series(nets)
    return dict(signal=name, gross_bp=g.mean(), net_bp=n.mean(),
                net_Sh=n.mean() / n.std() * np.sqrt(252) if n.std() > 0 else np.nan)


def main():
    on, p18, idr, carry = load()
    toll = tolls()
    r1 = np.log(p18 / p18.shift(1)) * 1e4
    r5 = np.log(p18 / p18.shift(5)) * 1e4
    r20 = np.log(p18 / p18.shift(20)) * 1e4
    vol20 = r1.rolling(20, min_periods=10).std()
    sigs = {
        "rev1": -(r1 / vol20),
        "rev5": -(r5 / (vol20 * np.sqrt(5))),
        "daynight": -(idr / vol20),
        "seas": on.shift(1).rolling(60, min_periods=30).mean(),
        "carry": pd.DataFrame({c: carry for c in [0]}).T.reindex(on.index, method=None)
                   .ffill().bfill()[on.columns.intersection(carry.index)]
                   if False else pd.DataFrame([carry[on.columns]] * len(on), index=on.index),
        "mom20 (ref)": r20 / (vol20 * np.sqrt(20)),
    }
    print(f"panel: {on.shape[1]} names, {len(on)} weekday nights "
          f"({on.index.min().date()} -> {on.index.max().date()})\n")
    rows_ic, rows_bk = [], []
    for name, sig in sigs.items():
        rows_ic.append(ic_row(sig, on, name))
        rows_bk.append(book_row(sig, on, toll, name))
    ic = pd.DataFrame(rows_ic).set_index("signal").round(4)
    bk = pd.DataFrame(rows_bk).set_index("signal").round(3)
    out = ic.join(bk)
    print(out.to_string())
    print("\nread: IC = nightly Spearman(signal, overnight ret); needs same sign both halves + |t|>3")
    print("to graduate. net book charges measured round-trip tolls; screen-grade (bid bars, no")
    print("engine) — survivors go to two-sided validation, not deployment.")
    out.to_csv(os.path.join(D, "x035b_screen.csv"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

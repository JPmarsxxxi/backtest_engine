"""#035 — build the nightly cross-asset panel for the swap-dodged overnight XS book.

For each of the 63 instruments that passed x034c's gate: pull the terminal's full M5 depth
(~17 months), snapshot prices at the three NY moments that define the trade, and save one long
panel keyed (day, symbol):
    p18   price at 18:00 NY (the swap-dodged entry)
    p0930 price at 09:30 NY next morning (the exit)
    p16   price at 16:00 NY (cash close, for the day-night signal)
    on_ret  overnight 18:00->next 09:30, bp
    id_ret  intraday 09:30->18:00 same day, bp (the day the entry decision is made)
Output: data/x035_nightly_panel.parquet. HYGIENE: MT5 bars are BID-based — stated, acceptable for
IC screening on 60-160bp/night moves with tolls charged separately (data-hygiene.md); any surviving
signal's final validation must move to two-sided data (FX block has Dukascopy mids 2019+).
"""
import os
import sys
import time as _t

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

E = r"C:\Users\User\backtest_engine\backtest_engine2"
D = os.path.join(E, "data")
OUT = os.path.join(D, "x035_nightly_panel.parquet")


def server_offset_h():
    mt5.symbol_select("EURUSD", True)
    t = mt5.symbol_info_tick("EURUSD")
    return round((t.time - _t.time()) / 3600)


def snap(df, hm0, hm1):
    """first bar OPEN in [hm0, hm1] per day — indices/metals/oil reopen at 18:05 NY (halt from
    ~16:50), so an exact-18:00 match silently drops entire asset classes."""
    w = df[(df.hm >= hm0) & (df.hm <= hm1)].sort_values("hm")
    return w.groupby("day")["open"].first()


def build(sym, off_h):
    mt5.symbol_select(sym, True)
    bars = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M5, 0, 99999)
    if bars is None or len(bars) == 0:
        return None
    df = pd.DataFrame(bars)
    # FTMO server clock = NY + 7h YEAR-ROUND (verified: zero Friday FX bars after 17:00 NY in any
    # month — the broker shifts UTC+2/+3 to pin the 17:00-NY rollover). A fixed UTC offset measured
    # today mislabels winter bars by 1h (the bug that emptied indices/metals from the first panel).
    ny = pd.to_datetime(df["time"], unit="s") - pd.Timedelta(hours=7)
    df["hm"] = ny.dt.hour * 100 + ny.dt.minute
    df["day"] = ny.dt.normalize()
    p18, p0930, p16 = snap(df, 1800, 1830), snap(df, 930, 1000), snap(df, 1600, 1630)
    days = p18.index
    nxt = p0930.reindex(days + pd.Timedelta(days=1))
    nxt.index = days
    mon = nxt.isna()
    if mon.any():                                        # Fri (and holiday) -> next quoted morning
        n3 = p0930.reindex(days + pd.Timedelta(days=3)); n3.index = days
        nxt = nxt.fillna(n3)
    out = pd.DataFrame({"p18": p18, "p0930_next": nxt, "p16": p16.reindex(days)})
    out["on_ret"] = (out.p0930_next / out.p18 - 1) * 1e4
    out["id_ret"] = (out.p18 / out.p0930_next.shift(1) - 1) * 1e4   # prev morning -> today 18:00
    out["symbol"] = sym
    return out.reset_index().rename(columns={"index": "day"})


def main():
    if not mt5.initialize():
        print("MT5 init failed")
        return 1
    off = server_offset_h()
    syms = pd.read_parquet(os.path.join(D, "x034c_universe.parquet")).columns.tolist()
    frames = []
    for s in syms:
        r = build(s, off)
        if r is None or r.on_ret.notna().sum() < 100:
            print(f"   {s}: thin, skipped")
            continue
        frames.append(r)
    mt5.shutdown()
    panel = pd.concat(frames, ignore_index=True)
    panel.to_parquet(OUT, index=False)
    nights = panel.groupby("day").symbol.nunique()
    print(f"panel: {panel.symbol.nunique()} symbols, {len(nights)} days "
          f"({nights.index.min().date()} -> {nights.index.max().date()}), "
          f"median names/night {int(nights.median())}")
    # firm N_eff on the long panel
    M = panel.pivot(index="day", columns="symbol", values="on_ret")
    M50 = M.dropna(thresh=int(M.shape[1] * 0.8))         # nights with >=80% of names
    C = M50.corr(min_periods=100)
    lam = np.linalg.eigvalsh(C.fillna(0).values)
    print(f"N_eff (long panel, {len(M50)} nights >=80% coverage): "
          f"{lam.sum()**2 / (lam**2).sum():.1f} of {M.shape[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

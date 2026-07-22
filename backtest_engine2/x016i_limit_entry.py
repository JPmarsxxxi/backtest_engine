"""#016i — LIMIT-ENTRY test for the live #016 book (the "earn the toll instead of paying it" upgrade).

Replays 2023->2026 on 1-min bid/ask OHLC (Dukascopy, banked by #030) for the v3 book:
  session A: SELL at 21:00 London, buy back at 21:50
  session B: BUY  at 23:00 London, sell     at 23:58

Entry styles compared on the SAME nights:
  MKT@00      market order on the hour's first bar (v1-style: pay whatever the spread is)
  MKT gated   first minute within 10 min whose half-spread <= cap (v2/v3-style; skip if never) —
              cap = 2.2 x this dataset's median entry-hour half, per pair (same rule as live recal)
  LIM k=0     resting order AT MID posted on the first bar;  k=1: at the far touch (bid for a buy,
              ask for a sell). Fill = the tradeable price actually crossed the level within the
              window (buy fills when ask_low <= L; sell when bid_high >= L) — touch=fill, so still
              optimistic on queue position, honest on adverse selection.
              Unfilled after W minutes -> two fallbacks reported: CROSS (market then) or SKIP.
Exit is always a market order at session end (the exit is time-forced ahead of the rollover halt,
so only the ENTRY side is up for optimization). GROSS mid->mid shown as the no-toll ceiling.

Caveat for reading absolute levels: Dukascopy spreads are 2-7x wider than FTMO's (#029f), so the
bp/day GAIN measured here overstates what FTMO will hand over; the RANKING of entry styles and the
fill rates are the transferable result. Verdict standard: LIM beats MKT gated at high fill rate,
stable per year -> greenlight a live v4 leg experiment.
"""
import os
import sys

import numpy as np
import pandas as pd

D = r"C:\Users\User\backtest_engine\backtest_engine2\data"
PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY"]
SESSIONS = {"A": {"hr": 21, "last_min": 49, "exit_min": 50, "side": -1},
            "B": {"hr": 23, "last_min": 57, "exit_min": 58, "side": +1}}
GATE_WAIT_MIN = 10          # v2-style gate: give the flare up to 10 1-min bars
LIM_WINDOWS = (5, 10, 30)   # minutes a resting order may wait
KS = (0.0, 1.0)             # 0 = at mid, 1 = at the far touch


def load_pair(sym):
    fb = os.path.join(D, f"{sym.lower()}_bid_1m.parquet")
    fa = os.path.join(D, f"{sym.lower()}_ask_1m.parquet")
    if not (os.path.exists(fb) and os.path.exists(fa)):
        return None
    b = pd.read_parquet(fb).set_index("open_time").sort_index()
    a = pd.read_parquet(fa).set_index("open_time").sort_index()
    idx = b.index.intersection(a.index)
    df = pd.DataFrame({"bid_o": b.loc[idx, "open"], "bid_h": b.loc[idx, "high"],
                       "bid_l": b.loc[idx, "low"], "bid_c": b.loc[idx, "close"],
                       "ask_o": a.loc[idx, "open"], "ask_h": a.loc[idx, "high"],
                       "ask_l": a.loc[idx, "low"], "ask_c": a.loc[idx, "close"]}, index=idx)
    ldn = df.index.tz_convert("Europe/London")
    df["hr"], df["mi"], df["day"], df["dow"] = ldn.hour, ldn.minute, ldn.date, ldn.dayofweek
    df["yr"] = ldn.year
    return df[df.dow < 5]


def session_nights(df, ses):
    """One row per (day): entry-window bars + exit price fields."""
    s = SESSIONS[ses]
    h = df[df.hr == s["hr"]]
    out = []
    for day, g in h.groupby("day"):
        g = g.sort_values("mi")
        ent = g[g.mi <= s["last_min"]]
        if ent.empty or ent.mi.iloc[0] > 2:      # need the top of the hour actually quoted
            continue
        ex = g[g.mi >= s["exit_min"]]
        if ex.empty:                              # no exit bar -> use last entry-window close
            exit_bid, exit_ask = ent.bid_c.iloc[-1], ent.ask_c.iloc[-1]
        else:
            exit_bid, exit_ask = ex.bid_o.iloc[0], ex.ask_o.iloc[0]
        out.append((day, ent, exit_bid, exit_ask, g.yr.iloc[0]))
    return out


def run_pair(sym, df):
    rows = []
    for ses, s in SESSIONS.items():
        side = s["side"]                          # +1 buy the hour, -1 sell it
        nights = session_nights(df, ses)
        # per-pair Dukascopy gate cap, same rule as the live recal (2.2 x median entry-hour half)
        halves = pd.concat([n[1] for n in nights])
        half_med = ((halves.ask_o - halves.bid_o) / 2 / ((halves.ask_o + halves.bid_o) / 2) * 1e4).median()
        cap = 2.2 * half_med
        for day, ent, exit_bid, exit_ask, yr in nights:
            mid0 = (ent.bid_o.iloc[0] + ent.ask_o.iloc[0]) / 2
            half0 = (ent.ask_o.iloc[0] - ent.bid_o.iloc[0]) / 2
            scale = 1e4 / mid0
            exit_px = exit_bid if side > 0 else exit_ask          # close long -> sell bid; short -> buy ask
            exit_mid = (exit_bid + exit_ask) / 2
            rec = {"pair": sym, "ses": ses, "day": day, "yr": yr,
                   "gross": side * (exit_mid - mid0) * scale,
                   "half0_bp": half0 * scale}
            # MKT@00 (v1-style)
            e = ent.ask_o.iloc[0] if side > 0 else ent.bid_o.iloc[0]
            rec["mkt00"] = side * (exit_px - e) * scale
            # MKT gated (v2-style): first minute with half <= cap, up to GATE_WAIT_MIN
            w = ent[ent.mi <= ent.mi.iloc[0] + GATE_WAIT_MIN]
            hbp = (w.ask_o - w.bid_o) / 2 / ((w.ask_o + w.bid_o) / 2) * 1e4
            okm = w[hbp <= cap]
            if len(okm):
                e = okm.ask_o.iloc[0] if side > 0 else okm.bid_o.iloc[0]
                rec["gated"] = side * (exit_px - e) * scale
            else:
                rec["gated"] = np.nan                              # skipped night
            # LIMIT variants
            for k in KS:
                L = mid0 - k * half0 if side > 0 else mid0 + k * half0
                for W in LIM_WINDOWS:
                    w = ent[ent.mi <= ent.mi.iloc[0] + W]
                    touched = (w.ask_l <= L) if side > 0 else (w.bid_h >= L)
                    tag = f"k{k:.0f}_w{W}"
                    if touched.any():
                        rec[f"lim_{tag}_cross"] = side * (exit_px - L) * scale
                        rec[f"lim_{tag}_skip"] = rec[f"lim_{tag}_cross"]
                        rec[f"fill_{tag}"] = 1.0
                    else:                                          # unfilled after W
                        j = w.index[-1]
                        e = df.loc[j, "ask_o"] if side > 0 else df.loc[j, "bid_o"]
                        rec[f"lim_{tag}_cross"] = side * (exit_px - e) * scale
                        rec[f"lim_{tag}_skip"] = np.nan
                        rec[f"fill_{tag}"] = 0.0
            rows.append(rec)
    return pd.DataFrame(rows), cap


def sharpe_daily(x):
    d = x.groupby(level=0).sum()
    return d.mean() / d.std() * np.sqrt(252) if d.std() > 0 else np.nan


def main():
    have = {s: load_pair(s) for s in PAIRS}
    have = {s: df for s, df in have.items() if df is not None}
    print(f"pairs with 1-min bid+ask on disk: {list(have)}")
    allrows = []
    for sym, df in have.items():
        r, cap = run_pair(sym, df)
        print(f"   {sym}: {len(r)} pair-nights, duka gate cap {cap:.2f}bp")
        allrows.append(r)
    R = pd.concat(allrows, ignore_index=True)
    span = f"{R.day.min()} -> {R.day.max()}"
    ndays = R.day.nunique()
    print(f"\n=== #016i LIMIT-ENTRY vs MARKET — {len(R):,} leg-nights, {ndays} days, {span} ===")
    print("(Dukascopy spreads = 2-7x FTMO -> absolute gains overstated; compare RANKINGS)\n")

    def line(name, col, fill_col=None):
        x = R[col]
        traded = x.notna()
        Rt = R[traded]
        bp_trade = Rt[col].mean()
        daily = R.set_index("day")[col].fillna(0.0)
        bp_day = daily.groupby(level=0).sum().mean()
        sh = sharpe_daily(daily)
        fr = R[fill_col].mean() * 100 if fill_col else 100.0 * traded.mean()
        yrs = R[traded].groupby("yr")[col].mean()
        yr_pos = f"{(yrs > 0).sum()}/{len(yrs)}"
        print(f"{name:>22}: {bp_trade:+6.3f} bp/leg | {bp_day:+7.2f} bp/day | Sh {sh:+5.2f} | "
              f"traded {fr:5.1f}% | yrs+ {yr_pos}")

    line("GROSS mid-to-mid", "gross")
    line("MKT @ HH:00 (v1)", "mkt00")
    line("MKT gated (v2/v3)", "gated")
    for k in KS:
        for W in LIM_WINDOWS:
            tag = f"k{k:.0f}_w{W}"
            line(f"LIM k={k:.0f} w={W}m cross", f"lim_{tag}_cross", f"fill_{tag}")
    for k in KS:
        for W in LIM_WINDOWS:
            tag = f"k{k:.0f}_w{W}"
            line(f"LIM k={k:.0f} w={W}m skip", f"lim_{tag}_skip", f"fill_{tag}")

    print("\nper-leg detail, best limit vs gated (bp/leg where both traded):")
    for (sym, ses), g in R.groupby(["pair", "ses"]):
        both = g[g["gated"].notna()]
        print(f"   {sym} {ses}: gated {both['gated'].mean():+6.3f} | "
              f"lim k=1 w=10 cross {g['lim_k1_w10_cross'].mean():+6.3f} | "
              f"fill {g['fill_k1_w10'].mean()*100:4.1f}% | entry half med {g['half0_bp'].median():.3f}bp")
    R.to_parquet(os.path.join(D, "x016i_results.parquet"), index=False)
    print("\nsaved night-level results -> data/x016i_results.parquet")


if __name__ == "__main__":
    main()

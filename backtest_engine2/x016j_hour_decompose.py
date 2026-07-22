"""#016j — DECOMPOSE the 21:00/23:00 hour edge from 1-min BID+ASK (2023-2026).

x016i found: session A (short 21:00->21:50) has NEGATIVE mid-to-mid gross on every pair, although the
hourly analysis (BID closes) says hour 21 falls. Two hypotheses this script separates:
  H1 timing:   the fall is concentrated in 21:50->22:00 (the minutes live CANNOT hold — broker halt)
  H2 artifact: the "fall" is bid-quote mechanics — half-spread widens into the 22:00 rollover, so
               bid (= mid − half) drops even if mid never moves; mirror-image inflation for hour 23
               (spread narrows back after rollover -> bid rises artificially).
For each pair: median half-spread at the boundary minutes; mean MID returns over the live windows,
the forfeited tails, and the full h1-style hour; mean BID returns over the same full hours (should
reproduce the h1 numbers); artifact size = half(start) − half(end) in bp.
"""
import os

import numpy as np
import pandas as pd

D = r"C:\Users\User\backtest_engine\backtest_engine2\data"
PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY"]


def load(sym):
    fb, fa = (os.path.join(D, f"{sym.lower()}_{f}_1m.parquet") for f in ("bid", "ask"))
    if not (os.path.exists(fb) and os.path.exists(fa)):
        return None
    b = pd.read_parquet(fb).set_index("open_time").sort_index()
    a = pd.read_parquet(fa).set_index("open_time").sort_index()
    idx = b.index.intersection(a.index)
    df = pd.DataFrame({"bid": b.loc[idx, "open"], "ask": a.loc[idx, "open"]}, index=idx)
    df["mid"] = (df.bid + df.ask) / 2
    df["half_bp"] = (df.ask - df.bid) / df.mid / 2 * 1e4
    ldn = df.index.tz_convert("Europe/London")
    df["hm"] = ldn.hour * 100 + ldn.minute
    df["day"] = ldn.date
    df["dow"] = ldn.dayofweek
    return df[df.dow < 5]


def snap(df, hm):
    """one row per day: the quote at minute hm (exact minute only — no fill-forward lies)."""
    g = df[df.hm == hm].groupby("day").first()
    return g


def ret_bp(a, b):
    """mean + t of log-ish return b vs a in bp, days aligned."""
    j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
    r = (j.iloc[:, j.columns.get_loc("mid_b")] / j.iloc[:, j.columns.get_loc("mid_a")] - 1) * 1e4
    return r.mean(), r.mean() / r.std() * np.sqrt(len(r)), len(r)


def retcol_bp(a, b, col):
    j = a[[col]].join(b[[col]], lsuffix="_a", rsuffix="_b", how="inner")
    r = (j[f"{col}_b"] / j[f"{col}_a"] - 1) * 1e4
    return r.mean(), r.mean() / r.std() * np.sqrt(len(r)), len(r)


def main():
    print(f"{'pair':>7} | {'half 2059':>9} {'half 2150':>9} {'half 2159':>9} {'half 2259':>9} "
          f"{'half 2359':>9} | spread-artifact bp (h21 / h23)")
    packs = {}
    for sym in PAIRS:
        df = load(sym)
        if df is None:
            print(f"{sym:>7} | no 1-min data yet")
            continue
        s = {hm: snap(df, hm) for hm in (2059, 2100, 2150, 2159, 2259, 2300, 2358, 2359)}
        packs[sym] = s
        h = {hm: s[hm]["half_bp"].median() for hm in s}
        # bid-return artifact ≈ half(start) − half(end): bid = mid − half (in bp of mid)
        art21 = h[2059] - h[2159]
        art23 = h[2259] - h[2359]
        print(f"{sym:>7} | {h[2059]:>9.3f} {h[2150]:>9.3f} {h[2159]:>9.3f} {h[2259]:>9.3f} "
              f"{h[2359]:>9.3f} | {art21:>+7.3f} / {art23:>+7.3f}")

    print(f"\n{'pair':>7} {'window':>16} | {'MID mean bp':>11} {'t':>6} {'n':>5} | {'BID mean bp':>11} {'t':>6}")
    for sym, s in packs.items():
        for name, a, b in (("21:00->21:50", 2100, 2150), ("21:50->21:59", 2150, 2159),
                           ("h21 full 59->59", 2059, 2159),
                           ("23:00->23:58", 2300, 2358), ("22:59->23:00", 2259, 2300),
                           ("h23 full 59->59", 2259, 2359)):
            m, t, n = retcol_bp(s[a], s[b], "mid")
            mb, tb, _ = retcol_bp(s[a], s[b], "bid")
            print(f"{sym:>7} {name:>16} | {m:>+11.3f} {t:>+6.1f} {n:>5} | {mb:>+11.3f} {tb:>+6.1f}")
        print()


if __name__ == "__main__":
    main()

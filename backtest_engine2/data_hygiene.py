"""Data hygiene helpers — the code twin of finding-alphas/data-hygiene.md (born from the #016
bid-only postmortem, alpha log 2026-07-15).

The identity: observed price = true mid + half_spread * side  (p = m + c*b). Signals computed on a
one-sided quote series inherit fake alpha whenever c or b moves systematically — random b => fake
mean-reversion (Roll 1984); scheduled c (rollover/open/close/fix) => fake time-of-day drift (#016).

Rules these helpers enforce/support:
  * signal research on MID, spread charged separately  -> load_mid_panel()
  * every backtest carries a zero-cost mid-to-mid row  -> gross_tripwire()
  * unknown/one-sided series get a Roll check           -> roll_effective_spread()
  * know where YOUR data's spread moves on a clock      -> spread_by_hour()
"""
import os

import numpy as np
import pandas as pd

DATA = r"C:\Users\User\backtest_engine\backtest_engine2\data"


def load_mid_panel(sym, freq="1m", data_dir=DATA):
    """Load {sym}_{bid,ask}_{freq}.parquet -> DataFrame with mid/half columns (open & close).

    Refuses to exist on one feed alone: raises if either side is missing, so a script cannot
    silently regress to single-sided research. Columns: bid_o/h/l/c, ask_o/h/l/c, mid_o, mid_c,
    half_o_bp, half_c_bp (half-spread in bp of mid).
    """
    fb = os.path.join(data_dir, f"{sym.lower()}_bid_{freq}.parquet")
    fa = os.path.join(data_dir, f"{sym.lower()}_ask_{freq}.parquet")
    missing = [f for f in (fb, fa) if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(
            f"{sym}: missing {missing} — data-hygiene rule 1: BOTH quote sides on disk before "
            f"any signal work (pull with x030_pull_1min_ohlc.py, DUKA_FEED=BID and =ASK)")
    b = pd.read_parquet(fb).set_index("open_time").sort_index()
    a = pd.read_parquet(fa).set_index("open_time").sort_index()
    idx = b.index.intersection(a.index)
    df = pd.DataFrame(index=idx)
    for c in ("open", "high", "low", "close"):
        df[f"bid_{c[0]}"] = b.loc[idx, c]
        df[f"ask_{c[0]}"] = a.loc[idx, c]
    df["mid_o"] = (df.bid_o + df.ask_o) / 2
    df["mid_c"] = (df.bid_c + df.ask_c) / 2
    df["half_o_bp"] = (df.ask_o - df.bid_o) / df.mid_o / 2 * 1e4
    df["half_c_bp"] = (df.ask_c - df.bid_c) / df.mid_c / 2 * 1e4
    return df


def roll_effective_spread(price, in_bp=True):
    """Roll (1984): estimate the effective HALF-spread embedded in a single price series from the
    negative serial covariance of its changes:  c = sqrt(-cov(dp_t, dp_{t-1})).

    Use on any series of uncertain construction BEFORE trusting fine-scale patterns on it. Returns
    half-spread (bp of price if in_bp) or np.nan if the serial covariance is positive (no detectable
    bounce — either genuinely clean, or drift/trend dominates at this sampling).
    """
    p = pd.Series(price).dropna().astype(float)
    dp = p.diff().dropna()
    cov = dp.autocorr(1)
    cov = cov * dp.var() if cov is not None and not np.isnan(cov) else np.nan
    if np.isnan(cov) or cov >= 0:
        return np.nan
    c = np.sqrt(-cov)
    return float(c / p.mean() * 1e4) if in_bp else float(c)


def gross_tripwire(mid_entry, mid_exit, side, label="GROSS mid-to-mid (zero-cost)"):
    """The mandatory sanity row: per-trade gross bp at TRUE prices with ZERO costs.

    If the strategy's net looks good while this row reads ~0, the 'edge' is quote mechanics /
    execution accounting, not the market (this row is what caught #016). Returns (mean_bp, t, n)
    and prints the row.
    """
    r = np.asarray(side, float) * (np.asarray(mid_exit, float) / np.asarray(mid_entry, float) - 1) * 1e4
    r = r[~np.isnan(r)]
    m = r.mean()
    t = m / r.std() * np.sqrt(len(r)) if len(r) > 1 and r.std() > 0 else np.nan
    print(f"{label}: {m:+.3f} bp/trade (t {t:+.1f}, n {len(r):,})"
          + ("   <-- ~0: edge is quote mechanics, STOP" if abs(m) < 0.05 else ""))
    return m, t, len(r)


def spread_by_hour(sym, freq="1m", tz="Europe/London", data_dir=DATA):
    """Median half-spread (bp) by hour of day — run ONCE per new instrument so you know where the
    data's spread moves on a clock (that's where clock-anchored signals get manufactured)."""
    df = load_mid_panel(sym, freq, data_dir)
    hr = df.index.tz_convert(tz).hour
    return df.groupby(hr)["half_o_bp"].median().rename(f"{sym} median half bp by {tz} hour")


if __name__ == "__main__":
    # smoke test on whatever majors are on disk
    for sym in ("EURUSD", "NZDUSD"):
        try:
            df = load_mid_panel(sym)
            emb = roll_effective_spread(df["bid_c"].resample("15min").last().dropna())
            print(f"{sym}: {len(df):,} bars | Roll half-spread embedded in BID closes: {emb:.3f}bp "
                  f"| true median half: {df.half_c_bp.median():.3f}bp")
        except FileNotFoundError as e:
            print(e)

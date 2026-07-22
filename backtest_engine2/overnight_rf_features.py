"""overnight_rf — SHARED feature builder: the single source of truth used by BOTH the research
notebook (overnight_rf.ipynb Cell 12 asserts parity with its in-notebook features) and the live
daemon (overnight_live_v2.py). Any feature change happens HERE, is re-validated in the notebook
(new walk-forward = new trial, counted), and only then reaches the live side.

Provenance/hygiene (data-hygiene.md): SPY daily OHLC = real ETF prints via Yahoo chart API (chosen
over ^GSPC after the 2026-07-17 fake-open discovery; residual open==prevClose nights dropped);
VIX = CBOE close (published ~16:15 ET, knowable at the 18:05 live entry). Row T's features use
nothing after T's close stamps. Target (research only): green = open(T+1) > close(T).
"""
import httpx
import numpy as np
import pandas as pd


def yahoo_ohlc(sym, period1="-631152000"):
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
           f"?period1={period1}&period2=9999999999&interval=1d")
    r = httpx.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30).json()["chart"]["result"][0]
    q = r["indicators"]["quote"][0]
    df = pd.DataFrame({k: q[k] for k in ("open", "high", "low", "close", "volume")},
                      index=pd.to_datetime(r["timestamp"], unit="s"))
    df.index = df.index.normalize()
    return df.dropna(subset=["open", "close"])


def build_features(px, vix):
    """px = SPY OHLCV daily (real prints); vix = VIX close series. Returns (F, on_bp, valid).
    Must stay byte-identical in logic to overnight_rf.ipynb Cell 2 (Cell 12 asserts it)."""
    F = pd.DataFrame(index=px.index)
    F["ret_day"] = (px.close / px.open - 1) * 1e4
    F["ret_1d"] = px.close.pct_change() * 1e4
    F["ret_5d"] = px.close.pct_change(5) * 1e4
    F["ret_20d"] = px.close.pct_change(20) * 1e4
    on_hist = (px.open / px.close.shift(1) - 1) * 1e4
    F["on_prev"] = on_hist
    F["on_prev2"] = on_hist.shift(1)
    sgn = np.sign(on_hist)
    F["on_streak"] = sgn.groupby((sgn != sgn.shift()).cumsum()).cumcount().add(1).mul(sgn)
    F["on_mean20"] = on_hist.rolling(20).mean()
    F["on_vol20"] = on_hist.rolling(20).std()
    r1 = px.close.pct_change()
    F["rvol5"] = r1.rolling(5).std() * 1e4
    F["rvol20"] = r1.rolling(20).std() * 1e4
    F["rvol_ratio"] = F.rvol5 / F.rvol20
    tr = np.maximum(px.high - px.low,
                    np.maximum((px.high - px.close.shift(1)).abs(),
                               (px.low - px.close.shift(1)).abs()))
    F["atr14"] = (tr / px.close).rolling(14).mean() * 1e4
    F["range_day"] = ((px.high - px.low) / px.close) * 1e4
    F["close_pos_range"] = ((px.close - px.low) / (px.high - px.low)).clip(0, 1)
    sma50, sma200 = px.close.rolling(50).mean(), px.close.rolling(200).mean()
    F["dist_sma200"] = (px.close / sma200 - 1) * 1e4
    F["dist_sma50"] = (px.close / sma50 - 1) * 1e4
    F["in_mkt"] = (px.close > sma200).astype(int)
    F["sma50_slope"] = sma50.pct_change(20) * 1e4
    F["dd_from_high"] = (px.close / px.close.rolling(252).max() - 1) * 1e4
    v = vix.reindex(px.index).ffill()
    F["vix"] = v
    F["vix_chg1"] = v.diff()
    F["vix_chg5"] = v.diff(5)
    F["vix_pct252"] = v.rolling(252).rank(pct=True)
    F["dow"] = px.index.dayofweek
    F["month"] = px.index.month
    gap_fwd = pd.Series(px.index, index=px.index).shift(-1).sub(
        pd.Series(px.index, index=px.index)).dt.days
    F["long_gap"] = (gap_fwd > 1).astype(int)
    mday = pd.Series(px.index, index=px.index).dt.day
    F["turn_of_month"] = ((mday <= 3) | (mday >= 27)).astype(int)
    F["nfp_friday"] = ((px.index.dayofweek == 3) & (mday <= 7)).astype(int)
    F["vol_ratio20"] = (px.volume / px.volume.rolling(20).mean()).replace(
        [np.inf, -np.inf], np.nan)
    on_bp = (px.open.shift(-1) / px.close - 1) * 1e4
    valid = on_bp.notna() & (on_bp != 0)
    return F, on_bp, valid


def latest_feature_row(px, vix):
    """The LIVE entry point: features for TODAY's row (the night about to be traded). long_gap for
    the last row can't come from the next index entry (doesn't exist yet) — set it from the calendar
    (Thu->Fri is not long; Fri handled by the Mon-Thu trading rule; holidays approximated as 0,
    consistent with how rare they are in training)."""
    F, _, _ = build_features(px, vix)
    row = F.iloc[[-1]].copy()
    row["long_gap"] = 0
    return row

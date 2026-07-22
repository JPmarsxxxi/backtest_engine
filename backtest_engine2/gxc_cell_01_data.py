# Cell 1 — #017b low-gamma intraday CONTINUATION: data + PIT features (no engine yet).
# Same us500 30m panel + lagged GEX as #017. The #017 decay-check used IN-SAMPLE quintiles
# (threshold lookahead); #017b must define "low gamma" PIT, so the flag is an EXPANDING
# percentile of prior lagged-GEX only. Long-only continuation (revert leg was absent in #017).
import numpy as np
import pandas as pd
from backtest.data import DataPanel

ENG = r"C:\Users\User\backtest_engine\backtest_engine2"

RAW = pd.read_parquet(ENG + r"\data\us500_1m.parquet")
s = RAW.set_index("open_time")["close"].sort_index()
s.index = s.index.tz_localize(None)
px30 = s.resample("30min", closed="left", label="right").last().dropna()
prices = px30.to_frame("USA500")
panel = DataPanel(prices, check_outliers=False)

ET = "America/New_York"
et_idx = prices.index.tz_localize("UTC").tz_convert(ET)
px_ = prices["USA500"]
tmin = np.asarray(et_idx.hour) * 60 + np.asarray(et_idx.minute)
et_date = pd.DatetimeIndex(et_idx.normalize().date)

g = pd.read_parquet(ENG + r"\data\sqz_dix_gex.parquet").set_index("date").sort_index()
gex_lag = g["gex"].shift(1)                                   # PIT: prior-session GEX
gex_lag.index = pd.DatetimeIndex(gex_lag.index.normalize())

us_days = pd.DatetimeIndex(sorted(pd.Series(et_date).unique()))
overlap = us_days.intersection(gex_lag.dropna().index)

# --- price stamps -> morning sign (known noon) + afternoon return (earned) ---
def _at(t):
    m = tmin == t
    return pd.Series(px_.values[m], index=pd.DatetimeIndex(et_date[m]))

p_open = _at(570).combine_first(_at(600))                    # 9:30, fallback 10:00
p_noon, p_close = _at(720), _at(960)                         # 12:00, 16:00
morn_sign = np.sign(p_noon / p_open - 1).rename("morn_sign")
aft_ret = (p_close / p_noon - 1).rename("aft_ret")

# --- PIT expanding-percentile low-gamma flag (no in-sample quintile) ---
gj = gex_lag.reindex(overlap).dropna()
exp_pct = gj.expanding(min_periods=60).apply(
    lambda x: float((x.iloc[:-1] < x.iloc[-1]).mean()) if len(x) > 1 else np.nan, raw=False
).rename("exp_pct")
low_g = (exp_pct < 0.20).rename("low_g")

feat = pd.concat([gj.rename("gex"), exp_pct, low_g, morn_sign, aft_ret], axis=1).loc[overlap]

print(f"overlap days {len(feat)} | warmup-valid (exp_pct notna) {int(feat.exp_pct.notna().sum())}")
print(f"morn_sign / aft_ret coverage: {int(feat.morn_sign.notna().sum())} / {int(feat.aft_ret.notna().sum())}")
print(f"low-gamma days (PIT, bottom 20% expanding): {int(feat.low_g.sum())}")
yr = feat[feat.low_g == True].index.year.value_counts().sort_index()
print("low-gamma days/yr:", {int(k): int(v) for k, v in yr.items()})
print(feat.dropna().tail(4).to_string())

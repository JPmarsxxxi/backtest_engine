# Cell 1 â€” Setup + data load
# Binance hourly (2021-04-01+) + Deribit BTC DVOL + PIT universe + entry signals + DataPanel

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

for _eng in [
    Path(r"C:\Users\User\backtest_engine"),
    Path("/content/drive/MyDrive/backtest_engine"),
    Path("/content/backtest_engine"),
    Path("..").resolve() / "backtest_engine",
]:
    if (_eng / "backtest").is_dir():
        sys.path.insert(0, str(_eng))
        break
else:
    raise FileNotFoundError("backtest_engine not found")

_nb_dir = str(Path.cwd())
if _nb_dir not in sys.path:
    sys.path.insert(0, _nb_dir)

from backtest.data import DataPanel, FieldSpec

# â”€â”€ Parameters â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
FAST_MODE        = True
CRYPTO_TOP_N     = 20 if FAST_MODE else 50
W_LIQ            = 168   # trailing qvol lookback for PIT universe ranking (hours)
T_TREND          = 168   # uptrend cumulative log-return window (hours)
R_MIN            = 0.06  # minimum cumulative log return for trend_up
L_MA             = 24    # SMA window for pullback signal (hours)
RECENT_DOWN_BARS = 1     # require negative log return on last N bar(s)
MIN_HISTORY_FRAC = 0.9

START_UTC = pd.Timestamp("2021-04-01", tz="UTC")   # first date Deribit DVOL is available
END_UTC   = pd.Timestamp.now(tz="UTC").floor("h")

STABLE_EXCLUDE = frozenset({
    "USDTUSDT", "USDCUSDT", "USDSUSDT", "USDEUSDT", "USD1USDT", "BFUSDUSDT",
    "UUSDT", "RLUSDUSDT", "PAXGUSDT", "XAUTUSDT", "EURUSDT", "FDUSDUSDT",
    "TUSDUSDT", "USDPUSDT",
})

CACHE_DIR     = Path("data")
BINANCE_CACHE = CACHE_DIR / "binance_hourly_top50_pool.parquet"
DVOL_CACHE    = CACHE_DIR / "btc_dvol_hourly.parquet"
MATIC_TO_POL  = pd.Timestamp("2024-09-04", tz="UTC")

ANCHOR_BINANCE = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "SHIBUSDT", "AVAXUSDT", "DOTUSDT",
    "LINKUSDT", "TRXUSDT", "BCHUSDT", "NEARUSDT", "MATICUSDT",
    "UNIUSDT", "LTCUSDT", "ICPUSDT", "ETCUSDT", "HBARUSDT",
    "ATOMUSDT", "FILUSDT", "APTUSDT", "ARBUSDT", "OPUSDT",
    "INJUSDT", "SUIUSDT", "SEIUSDT", "TIAUSDT", "LDOUSDT",
]

# â”€â”€ Deribit DVOL fetch â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_DVOL_URL = "https://www.deribit.com/api/v2/public/get_volatility_index_data"

def _fetch_dvol(start_dt, end_dt, currency="BTC", resolution="3600", pause=0.15):
    start_ms, end_ms = int(start_dt.timestamp() * 1000), int(end_dt.timestamp() * 1000)
    candles, cur_end = [], end_ms
    while True:
        r = requests.get(
            _DVOL_URL,
            params={"currency": currency, "start_timestamp": start_ms,
                    "end_timestamp": cur_end, "resolution": resolution},
            timeout=15,
        )
        r.raise_for_status()
        res = r.json()["result"]
        candles.extend(res["data"])
        if res["continuation"] is None:
            break
        cur_end = res["continuation"]
        time.sleep(pause)
    candles.sort(key=lambda x: x[0])
    df = pd.DataFrame(candles, columns=["ts_ms", "open", "high", "low", "close"])
    df.index = pd.to_datetime(df.pop("ts_ms"), unit="ms", utc=True)
    return df

if DVOL_CACHE.exists():
    dvol_raw = pd.read_parquet(DVOL_CACHE)
    _last = dvol_raw.index.max()
    _last_utc = _last if _last.tzinfo else _last.tz_localize("UTC")
    if END_UTC - _last_utc > pd.Timedelta(hours=2):
        _new = _fetch_dvol(_last_utc - pd.Timedelta(hours=1), END_UTC)
        dvol_raw = pd.concat([dvol_raw, _new]).sort_index()
        dvol_raw = dvol_raw[~dvol_raw.index.duplicated(keep="last")]
        dvol_raw.to_parquet(DVOL_CACHE)
        print(f"DVOL cache updated â†’ {dvol_raw.index.max()}")
    else:
        print(f"DVOL cache up-to-date through {_last}")
else:
    print("Fetching Deribit BTC DVOL (2021-04-01 â†’ now) â€¦")
    dvol_raw = _fetch_dvol(START_UTC, END_UTC)
    dvol_raw.to_parquet(DVOL_CACHE)
    print(f"DVOL cached: {len(dvol_raw):,} bars  {dvol_raw.index.min()} â†’ {dvol_raw.index.max()}")

# â”€â”€ Load Binance hourly parquet â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if not BINANCE_CACHE.exists():
    raise FileNotFoundError(f"Missing {BINANCE_CACHE}. Run ou_optimal_trend_pullback.ipynb Cell 0 first.")

raw  = pd.read_parquet(BINANCE_CACHE)
raw["open_time"] = pd.to_datetime(raw["open_time"], utc=True)

matic = raw[raw["symbol"] == "MATICUSDT"]
pol   = raw[raw["symbol"] == "POLUSDT"]
if not matic.empty and not pol.empty:
    stitched = pd.concat(
        [matic[matic["open_time"] < MATIC_TO_POL],
         pol[pol["open_time"] >= MATIC_TO_POL]],
        ignore_index=True,
    ).assign(symbol="MATICUSDT")
    raw = pd.concat([raw[~raw["symbol"].isin(["MATICUSDT", "POLUSDT"])], stitched], ignore_index=True)

trade_cols = sorted({s for s in raw["symbol"].unique() if s != "POLUSDT"})
close = raw.pivot(index="open_time", columns="symbol", values="close").reindex(columns=trade_cols)
qvol  = raw.pivot(index="open_time", columns="symbol", values="quote_volume").reindex(columns=trade_cols)

full_idx = pd.date_range(close.index.min(), close.index.max(), freq="1h", tz="UTC")
close = close.reindex(full_idx)
qvol  = qvol.reindex(full_idx)
# strip timezone â€” DataPanel requires a plain DatetimeIndex (no tz)
close.index = close.index.tz_convert("UTC").tz_localize(None)
qvol.index  = qvol.index.tz_convert("UTC").tz_localize(None)

# â”€â”€ Filter to DVOL start date â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_start_naive = START_UTC.tz_localize(None)
close = close[close.index >= _start_naive].copy()
qvol  = qvol[qvol.index  >= _start_naive].copy()

# â”€â”€ Process DVOL â†’ tz-naive series aligned to close.index â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_dvol_s = dvol_raw["close"].copy()
if _dvol_s.index.tzinfo is not None:
    _dvol_s.index = _dvol_s.index.tz_convert("UTC").tz_localize(None)
dvol_aligned = _dvol_s.reindex(close.index, method="ffill").rename("DVOL")

# â”€â”€ Monthly PIT universe (top-N by trailing qvol, decided at prior month-end) â”€
MIN_HISTORY_BARS = max(int(W_LIQ * MIN_HISTORY_FRAC), 24)
_rank_cols   = [c for c in close.columns if c not in STABLE_EXCLUDE]
pit_universe = pd.DataFrame(False, index=close.index, columns=close.columns)

for _per in close.index.to_period("M").unique():
    _mask = close.index.to_period("M") == _per
    _idx  = np.flatnonzero(_mask)
    if _idx[0] == 0:
        continue
    _t_dec  = close.index[_idx[0] - 1]
    _px_lb  = close.loc[:_t_dec, _rank_cols].iloc[-W_LIQ:]
    _vol_lb = qvol.loc[:_t_dec, _rank_cols].iloc[-W_LIQ:]
    _n_ok   = _px_lb.notna().sum()
    _last   = _px_lb.iloc[-1]
    _elig   = (_n_ok >= MIN_HISTORY_BARS) & _last.notna() & _vol_lb.notna().any()
    _liq    = _vol_lb.median()
    _liq[~_elig] = np.nan
    _ranks  = _liq.rank(ascending=False, method="min")
    _picked = _ranks.index[_ranks <= CRYPTO_TOP_N]
    pit_universe.loc[_mask, _picked] = True

# â”€â”€ Causal entry features â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_min_trend = max(T_TREND // 2, 24)
_min_ma    = max(L_MA // 2, 12)

log_ret    = np.log(close / close.shift(1))
roll_trend = log_ret.rolling(T_TREND, min_periods=_min_trend).sum()
trend_up   = roll_trend > R_MIN
sma        = close.rolling(L_MA, min_periods=_min_ma).mean()
pullback   = close < sma
recent_down = log_ret < 0 if RECENT_DOWN_BARS == 1 else \
              log_ret.rolling(RECENT_DOWN_BARS, min_periods=RECENT_DOWN_BARS).sum() < 0
entry_signal = pit_universe & trend_up & pullback & recent_down

# â”€â”€ Broadcast DVOL scalar to all asset columns â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
dvol_broadcast = pd.DataFrame(
    np.repeat(dvol_aligned.values[:, None], len(close.columns), axis=1),
    index=close.index,
    columns=close.columns,
    dtype=np.float64,
)

# â”€â”€ DataPanel â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
features = {
    "pit_universe": pit_universe.astype(np.float64),
    "trend_up":     trend_up.astype(np.float64),
    "pullback":     pullback.astype(np.float64),
    "recent_down":  recent_down.astype(np.float64),
    "entry_signal": entry_signal.astype(np.float64),
    "roll_trend":   roll_trend.astype(np.float64),
    "sma_pullback": sma.astype(np.float64),
    "log_returns":  log_ret.astype(np.float64),
    "DVOL":         dvol_broadcast,
}

_feat_spec = FieldSpec(lag=0, missing="drop")
specs = {
    "prices":   FieldSpec(lag=0),
    "volume":   FieldSpec(lag=0),
    "universe": FieldSpec(lag=0),
    **{k: _feat_spec for k in features},
}

panel = DataPanel(
    prices=close,
    volume=qvol,
    features=features,
    universe=pit_universe.astype(bool),
    specs=specs,
    check_outliers=True,
)

# â”€â”€ Validation output â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_n_entry = int(entry_signal.sum().sum())
_n_pit   = int(pit_universe.sum().sum())
print(f"-- DataPanel (crypto_regime_ou_trend) --")
print(f"FAST_MODE          : {FAST_MODE}  (CRYPTO_TOP_N={CRYPTO_TOP_N})")
print(f"Bars               : {len(panel.dates):,}")
print(f"Assets in matrix   : {len(panel.assets_all)}")
print(f"Range              : {panel.dates[0]} â†’ {panel.dates[-1]}")
print(f"NaN fraction prices: {close.isna().mean().mean():.2%}")
print(f"DVOL bars          : {dvol_aligned.notna().sum():,}  (NaN after ffill: {dvol_aligned.isna().sum():,})")
print(f"DVOL range         : {dvol_aligned.min():.1f} â†’ {dvol_aligned.max():.1f}")
print(f"PIT universe cells : {_n_pit:,}")
print(f"Entry signal cells : {_n_entry:,}")
print(f"Fields             : {panel.available_fields()}")
print(f"\nEntry signals per year (sum over assets):")
print(entry_signal.sum(axis=1).resample("YE").sum().to_string())
print()
close[[c for c in ["BTCUSDT", "ETHUSDT", "SOLUSDT"] if c in close.columns]].tail(3)

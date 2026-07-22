# Cell 1 — Setup + hourly Binance + PIT universe + entry signals (Stage 1)
# PIT: each month, top-N liquid chosen using ONLY data through last bar of prior month.
# Signals (trend/pullback/down) are causal rolls; entry only where monthly pit_universe is True.
# FAST_MODE: TOP_N=20 for smoke; full run sets FAST_MODE=False.
# Data: loads data/binance_hourly_top50_pool.parquet (REFRESH_BINANCE_CACHE=False = no API download).
# Colab: mount Drive — data at MyDrive/backtest_engine2/data/*.parquet; library at MyDrive/backtest_engine.

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

for _eng in [
    Path("/content/drive/MyDrive/backtest_engine"),
    Path("/content/backtest_engine"),
    Path(r"C:\Users\User\backtest_engine"),
    Path("..").resolve() / "backtest_engine",
]:
    if (_eng / "backtest").is_dir():
        sys.path.insert(0, str(_eng))
        break
else:
    raise FileNotFoundError("backtest_engine not found — mount Drive or set path in Cell 1")
from backtest.data import DataPanel, FieldSpec

# ── FAST_MODE / crypto params (Cell 2 will mirror these) ─────────────────────
FAST_MODE = True

CRYPTO_TOP_N = 20 if FAST_MODE else 50
W_LIQ = 168          # median quote-volume lookback (hours)
T_TREND = 168        # uptrend window (hours)
R_MIN = 0.06         # min cumulative log return for uptrend (2a)
L_MA = 24            # pullback SMA (hours)
RECENT_DOWN_BARS = 1 # 3B: require negative log return on last N bar(s)
MIN_HISTORY_FRAC = 0.9

STABLE_EXCLUDE = frozenset({
    "USDTUSDT", "USDCUSDT", "USDSUSDT", "USDEUSDT", "USD1USDT", "BFUSDUSDT",
    "UUSDT", "RLUSDUSDT", "PAXGUSDT", "XAUTUSDT", "EURUSDT", "FDUSDUSDT",
    "TUSDUSDT", "USDPUSDT",
})

START_UTC = pd.Timestamp("2019-01-01", tz="UTC")
END_UTC = pd.Timestamp.now(tz="UTC").floor("h")
MATIC_TO_POL_DATE = pd.Timestamp("2024-09-04", tz="UTC")

CACHE_DIR = Path("data")
CACHE_DIR.mkdir(exist_ok=True)
BINANCE_CACHE = CACHE_DIR / "binance_hourly_top50_pool.parquet"
# False = use existing parquet only (Colab/local). True = incremental API fetch for missing symbols.
REFRESH_BINANCE_CACHE = False

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
BINANCE_INFO = "https://api.binance.com/api/v3/exchangeInfo"

KLINES_COLS = [
    "open_time", "open", "high", "low", "close", "base_volume", "close_time",
    "quote_volume", "n_trades", "taker_buy_base", "taker_buy_quote", "ignored",
]

ANCHOR_BINANCE = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "SHIBUSDT", "AVAXUSDT", "DOTUSDT",
    "LINKUSDT", "TRXUSDT", "BCHUSDT", "NEARUSDT", "MATICUSDT",
    "UNIUSDT", "LTCUSDT", "ICPUSDT", "ETCUSDT", "HBARUSDT",
    "ATOMUSDT", "FILUSDT", "APTUSDT", "ARBUSDT", "OPUSDT",
    "INJUSDT", "SUIUSDT", "SEIUSDT", "TIAUSDT", "LDOUSDT",
]


def fetch_binance_usdt_set(session=None):
    sess = session or requests.Session()
    r = sess.get(BINANCE_INFO, timeout=30)
    r.raise_for_status()
    return {
        s["symbol"]
        for s in r.json()["symbols"]
        if s.get("quoteAsset") == "USDT"
        and s.get("status") == "TRADING"
        and s.get("isSpotTradingAllowed") is not False
    }


def fetch_klines(symbol, start_ms, end_ms, session=None):
    sess = session or requests.Session()
    out, cur = [], start_ms
    while cur <= end_ms:
        params = dict(symbol=symbol, interval="1h", startTime=cur, endTime=end_ms, limit=1000)
        for attempt in range(5):
            r = sess.get(BINANCE_KLINES, params=params, timeout=30)
            if r.status_code == 200:
                break
            if r.status_code in (429, 418):
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 400:
                return pd.DataFrame(columns=KLINES_COLS).set_index("open_time")
            r.raise_for_status()
        else:
            raise RuntimeError(f"{symbol}: too many retries at {cur}")
        rows = r.json()
        if not rows:
            break
        out.extend(rows)
        last_open = rows[-1][0]
        if len(rows) < 1000:
            break
        cur = last_open + 3_600_000
        time.sleep(0.12)
    if not out:
        return pd.DataFrame(columns=KLINES_COLS).set_index("open_time")
    df = pd.DataFrame(out, columns=KLINES_COLS)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "base_volume", "quote_volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.set_index("open_time").sort_index()[~df.index.duplicated()]


# ── 1) Load / extend Binance hourly cache ─────────────────────────────────────
if not BINANCE_CACHE.exists():
    raise FileNotFoundError(
        f"Missing {BINANCE_CACHE}. Run hurst_optimal_exit_pairs.ipynb Cell 1 once, "
        "or we can add a full fetch path in a follow-up cell."
    )

raw = pd.read_parquet(BINANCE_CACHE)
raw["open_time"] = pd.to_datetime(raw["open_time"], utc=True)
have = set(raw["symbol"].unique())
target_symbols = sorted(s for s in ANCHOR_BINANCE if s in have)
# Full pool from cache; FAST_MODE only lowers CRYPTO_TOP_N (20 vs 50).
print(f"Loaded {BINANCE_CACHE.name}: {len(have)} symbols, {len(raw):,} rows")
print(f"FAST_MODE={FAST_MODE}  CRYPTO_TOP_N={CRYPTO_TOP_N}  REFRESH_BINANCE_CACHE={REFRESH_BINANCE_CACHE}")

start_ms = int(START_UTC.timestamp() * 1000)
end_ms = int(END_UTC.timestamp() * 1000)
need = [s for s in target_symbols if s not in have and s != "POLUSDT"]
if REFRESH_BINANCE_CACHE and need and not FAST_MODE:
    session = requests.Session()
    binance_usdt = fetch_binance_usdt_set(session)
    need = [s for s in target_symbols if s not in have and s not in binance_usdt and s != "POLUSDT"]
    print(f"Incremental fetch for {len(need)} symbols ...")
    pieces = []
    for i, sym in enumerate(need, 1):
        t0 = time.time()
        df = fetch_klines(sym, start_ms, end_ms, session=session)
        print(f"  [{i:3d}/{len(need)}] {sym:12s} bars={len(df):7,d}  {time.time()-t0:5.1f}s")
        if df.empty:
            continue
        pieces.append(
            df[["close", "quote_volume"]].reset_index().assign(symbol=sym)
        )
    if pieces:
        raw = pd.concat([raw, pd.concat(pieces, ignore_index=True)], ignore_index=True)
        raw.to_parquet(BINANCE_CACHE)
        print(f"Updated cache -> {BINANCE_CACHE}")

matic = raw[raw["symbol"] == "MATICUSDT"]
pol = raw[raw["symbol"] == "POLUSDT"]
if not matic.empty and not pol.empty:
    matic_keep = matic[matic["open_time"] < MATIC_TO_POL_DATE]
    pol_keep = pol[pol["open_time"] >= MATIC_TO_POL_DATE]
    stitched = pd.concat([matic_keep, pol_keep], ignore_index=True).assign(symbol="MATICUSDT")
    raw_x = pd.concat(
        [raw[~raw["symbol"].isin(["MATICUSDT", "POLUSDT"])], stitched],
        ignore_index=True,
    )
else:
    raw_x = raw.copy()

trade_cols = sorted({s for s in raw_x["symbol"].unique() if s != "POLUSDT"})
close = raw_x.pivot(index="open_time", columns="symbol", values="close").reindex(columns=trade_cols)
qvol = raw_x.pivot(index="open_time", columns="symbol", values="quote_volume").reindex(columns=trade_cols)

full_idx = pd.date_range(close.index.min(), close.index.max(), freq="1h", tz="UTC")
close = close.reindex(full_idx)
qvol = qvol.reindex(full_idx)
close.index = close.index.tz_convert("UTC").tz_localize(None)
qvol.index = qvol.index.tz_convert("UTC").tz_localize(None)

# ── 2) Monthly PIT universe (top-N liquid as of prior month-end) ───────────────
MIN_HISTORY_BARS = max(int(W_LIQ * MIN_HISTORY_FRAC), 24)
_rank_cols = [c for c in close.columns if c not in STABLE_EXCLUDE]

pit_universe = pd.DataFrame(False, index=close.index, columns=close.columns)

for per in close.index.to_period("M").unique():
    month_mask = close.index.to_period("M") == per
    idx = np.flatnonzero(month_mask)
    if idx[0] == 0:
        continue
    t_dec = close.index[idx[0] - 1]

    px_lb = close.loc[:t_dec, _rank_cols].iloc[-W_LIQ:]
    vol_lb = qvol.loc[:t_dec, _rank_cols].iloc[-W_LIQ:]

    n_ok = px_lb.notna().sum()
    last_px = px_lb.iloc[-1]
    eligible = (n_ok >= MIN_HISTORY_BARS) & last_px.notna() & vol_lb.notna().any()

    liq = vol_lb.median()
    liq[~eligible] = np.nan
    ranks = liq.rank(ascending=False, method="min")
    picked = ranks.index[ranks <= CRYPTO_TOP_N]

    pit_universe.loc[month_mask, picked] = True

# ── 3) Causal entry features, masked by PIT universe ─────────────────────────
min_trend = max(T_TREND // 2, 24)
min_ma = max(L_MA // 2, 12)

log_ret = np.log(close / close.shift(1))

roll_trend = log_ret.rolling(T_TREND, min_periods=min_trend).sum()
trend_up = roll_trend > R_MIN

sma = close.rolling(L_MA, min_periods=min_ma).mean()
pullback = close < sma

if RECENT_DOWN_BARS == 1:
    recent_down = log_ret < 0
else:
    recent_down = log_ret.rolling(RECENT_DOWN_BARS, min_periods=RECENT_DOWN_BARS).sum() < 0

entry_signal = pit_universe & trend_up & pullback & recent_down

features = {
    "pit_universe": pit_universe.astype(np.float64),
    "trend_up": trend_up.astype(np.float64),
    "pullback": pullback.astype(np.float64),
    "recent_down": recent_down.astype(np.float64),
    "entry_signal": entry_signal.astype(np.float64),
    "roll_trend": roll_trend.astype(np.float64),
    "sma_pullback": sma.astype(np.float64),
    "log_returns": log_ret.astype(np.float64),
}

feat_spec = FieldSpec(lag=0, missing="drop")
specs = {
    "prices": FieldSpec(lag=0),
    "volume": FieldSpec(lag=0),
    "universe": FieldSpec(lag=0),
    **{k: feat_spec for k in features},
}

panel = DataPanel(
    prices=close,
    volume=qvol,
    features=features,
    universe=pit_universe.astype(bool),
    specs=specs,
    check_outliers=True,
)

# ── 4) Validation output ─────────────────────────────────────────────────────
n_entry = int(entry_signal.sum().sum())
n_pit = int(pit_universe.sum().sum())
by_year = entry_signal.sum(axis=1).resample("YE").sum()
_pit_per_m = pit_universe.sum(axis=1).resample("ME").mean()

print(f"\n-- DataPanel (ou-optimal-trend-pullback Cell 1, PIT) --")
print(f"FAST_MODE          : {FAST_MODE}")
print(f"PIT universe       : monthly top-{CRYPTO_TOP_N} by trailing {W_LIQ}h qvol")
print(f"  decision time    : last bar of prior month (no same-month volume peek)")
print(f"  min history      : {MIN_HISTORY_BARS} valid price bars in lookback")
print(f"Bars               : {len(panel.dates):,}")
print(f"Assets in matrix   : {len(panel.assets_all)}  (broad cache; tradable = universe)")
print(f"Range              : {panel.dates[0]} -> {panel.dates[-1]}")
print(f"NaN prices         : {close.isna().mean().mean():.2%}")
print(f"NaN log_returns    : {log_ret.isna().mean().mean():.2%}")
print(f"Fields             : {panel.available_fields()}")
print(f"\nEntry params: TOP_N={CRYPTO_TOP_N} W_LIQ={W_LIQ} T_TREND={T_TREND} "
      f"R_MIN={R_MIN} L_MA={L_MA} recent_down_bars={RECENT_DOWN_BARS}")
print(f"Bar×asset True: pit_universe={n_pit:,}  entry_signal={n_entry:,}")
print(f"Avg names/month in universe (~{CRYPTO_TOP_N} cap): "
      f"{pit_universe.sum(axis=1).resample('ME').mean().mean():.1f}")
print(f"\nEntry signals per calendar year (sum over assets):")
print(by_year.to_string())

print("\nentry_signal.tail(3) [BTC, ETH, SOL if present]:")
_show = [c for c in ["BTCUSDT", "ETHUSDT", "SOLUSDT"] if c in close.columns]
print(entry_signal[_show].tail(3))

print("\nclose.tail(2):")
print(close[_show].tail(2))
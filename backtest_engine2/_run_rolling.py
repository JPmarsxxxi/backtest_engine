import sys, warnings, time, subprocess, requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
sys.path.insert(0, r'C:\Users\User\backtest_engine')
warnings.filterwarnings("ignore")

for pkg in ["coinmetrics-api-client", "pytrends", "pandas-datareader"]:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

# ── CELL 1: Data Load ─────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_datareader.data as pdr
from coinmetrics.api_client import CoinMetricsClient
from pytrends.request import TrendReq
from backtest.data import DataPanel, FieldSpec

START         = "2014-01-01"
END           = "2026-05-20"
MIN_MCAP_USD  = 10_000_000
LAG_BARS      = 1
UNIVERSE_SIZE = 150      # top N non-stable coins by market cap — no hardcoding
CG_SLEEP      = 7.0      # seconds between CoinGecko calls — public API ~8 req/min safe
CM_MAX_ASSETS = 50       # CM community only covers major coins; skip the long tail
CM_TIMEOUT    = 6        # seconds per CM API call before giving up (prevents hangs)
CACHE_DIR     = Path("data_cache")
CACHE_DIR.mkdir(exist_ok=True)

STABLECOINS = frozenset({
    "tether", "usd-coin", "binance-usd", "dai", "true-usd", "frax", "usdd",
    "tusd", "usdp", "gusd", "stasis-eurs", "neutrino", "paxos-standard",
    "liquity-usd", "fei-usd", "usdn", "celo-dollar", "nusd", "origin-dollar",
    "paypal-usd", "euro-coin", "angle-protocol", "terra-luna",
    "terrausd", "wrapped-steth", "staked-ether",   # wrapped/staked tokens
    "wrapped-bitcoin", "wrapped-ether",
})

# ── 3a. Dynamic universe from CoinGecko ───────────────────────────────────────
def _cg_top_coins(n, stables=STABLECOINS):
    """Return [(cg_id, SYMBOL)] for top n non-stable coins by market cap."""
    url, result, page = "https://api.coingecko.com/api/v3/coins/markets", [], 1
    while len(result) < n + 80:   # fetch extra to absorb stable/wrapped exclusions
        try:
            resp = requests.get(url, params={
                "vs_currency": "usd", "order": "market_cap_desc",
                "per_page": 250, "page": page, "sparkline": False,
            }, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            print(f"  CG markets page {page} failed: {e}"); break
        batch = resp.json()
        if not batch: break
        result.extend(batch)
        if len(batch) < 250: break
        page += 1
        time.sleep(1.0)
    # filter stables, then deduplicate symbols (keep first = higher market cap)
    seen, out = set(), []
    for c in result:
        sym = c["symbol"].upper()
        if c["id"] not in stables and sym not in seen:
            seen.add(sym)
            out.append((c["id"], sym))
    return out[:n]

# ── 3b. Per-coin historical fetch with parquet cache ──────────────────────────
def _cg_fetch(cg_id, sym, start, end, cache_dir, sleep_sec):
    """Fetch daily close/volume/mcap from CoinGecko. Returns DataFrame indexed by date."""
    cache_path = cache_dir / f"cg_{sym}.parquet"
    end_ts = pd.Timestamp(end)

    if cache_path.exists():
        try:
            cached = pd.read_parquet(cache_path)
            if not cached.empty and cached.index[-1] >= end_ts - pd.Timedelta(days=5):
                return cached
        except Exception:
            pass

    url    = f"https://api.coingecko.com/api/v3/coins/{cg_id}/market_chart/range"
    params = {"vs_currency": "usd",
              "from": int(pd.Timestamp(start).timestamp()),
              "to":   int(end_ts.timestamp())}
    data = None
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=60)
            if resp.status_code == 429:
                print(f"    [{sym}] 429 — sleeping 90s"); time.sleep(90); continue
            resp.raise_for_status()
            data = resp.json(); break
        except Exception:
            time.sleep(15 * (attempt + 1))
    if data is None or not data.get("prices"):
        return pd.DataFrame(columns=["close", "volume", "mcap"])

    p = pd.DataFrame(data["prices"],        columns=["ts", "close"])
    v = pd.DataFrame(data["total_volumes"], columns=["ts", "volume"])
    m = pd.DataFrame(data["market_caps"],   columns=["ts", "mcap"])
    df = pd.DataFrame({"close":  p["close"].values,
                        "volume": v["volume"].values,
                        "mcap":   m["mcap"].values},
                       index=pd.to_datetime(p["ts"], unit="ms").normalize())
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index = pd.DatetimeIndex(df.index)
    df.to_parquet(cache_path)
    time.sleep(sleep_sec)
    return df

print(f"Fetching top {UNIVERSE_SIZE} coins from CoinGecko...")
universe_list = _cg_top_coins(UNIVERSE_SIZE)
print(f"  {len(universe_list)} coins: {[s for _, s in universe_list[:12]]}...")

# ── 3c. Assemble price / volume / mcap DataFrames ─────────────────────────────
# Full calendar date range — crypto trades 24/7 (no weekend gaps unlike equities)
date_idx = pd.date_range(START, END, freq="D")
prices_d, volume_d, mcap_d = {}, {}, {}
for i, (cg_id, sym) in enumerate(universe_list):
    df = _cg_fetch(cg_id, sym, START, END, CACHE_DIR, CG_SLEEP)
    if df.empty or "close" not in df.columns:
        continue
    prices_d[sym] = df["close"].reindex(date_idx)
    volume_d[sym] = df["volume"].reindex(date_idx)
    mcap_d[sym]   = df["mcap"].reindex(date_idx)
    if (i + 1) % 25 == 0 or (i + 1) == len(universe_list):
        print(f"  {i+1}/{len(universe_list)} fetched...")

prices = pd.DataFrame(prices_d, index=date_idx)
volume = pd.DataFrame(volume_d, index=date_idx)
mcap   = pd.DataFrame(mcap_d,   index=date_idx)
SYMS   = list(prices.columns)
print(f"\nPrices: {prices.shape}  NaN: {prices.isna().mean().mean():.1%}")
print(f"Universe ({len(SYMS)} coins): {SYMS[:12]}...")

# ── 4. COIN METRICS — per-asset on-chain (AA, NT, MRR, NF, TF) ───────────────
# CM community covers ~20-30 major coins; rest NaN → imputed by pipeline.
# TV and MCAP now come from CoinGecko for ALL coins — no longer needed from CM.
CM_METRICS = {
    "AdrActCnt":    "AA",    # active addresses
    "TxCnt":        "NT",    # transaction count
    "CapMVRVCur":   "MRR",   # market value / realised value
    "FlowInExNtv":  "FLIN",  # exchange inflow (for NF/TF)
    "FlowOutExNtv": "FLOUT", # exchange outflow (for NF/TF)
}
cm = CoinMetricsClient(host="community-api.coinmetrics.io")

def _fetch_cm_metric(cm_client, syms, metric, start, end, date_idx,
                     sleep=0.15, call_timeout=CM_TIMEOUT, max_assets=CM_MAX_ASSETS):
    frames = []
    with ThreadPoolExecutor(max_workers=1) as exc:
        for sym in syms[:max_assets]:
            def _call(s=sym):   # default-arg captures loop variable by value
                df = cm_client.get_asset_metrics(
                    assets=[s.lower()], metrics=[metric],
                    start_time=start, end_time=end, frequency="1d",
                ).to_dataframe()
                df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
                return df.set_index("time")[[metric]].rename(columns={metric: s})
            try:
                df = exc.submit(_call).result(timeout=call_timeout)
                frames.append(df)
                time.sleep(sleep)
            except Exception:
                pass
    if not frames:
        return pd.DataFrame(index=date_idx, columns=prices.columns)
    return pd.concat(frames, axis=1).reindex(index=date_idx, columns=prices.columns)

print("\nFetching Coin Metrics (on-chain, major coins only)...")
cm_data = {}
for metric, label in CM_METRICS.items():
    df = _fetch_cm_metric(cm, SYMS, metric, START, END, date_idx)
    cov = df.notna().mean().mean()
    n   = df.notna().any().sum()
    print(f"  {label:6s}: {cov:.1%} coverage  {n} assets")
    cm_data[label] = df

flin  = cm_data.pop("FLIN",  None)
flout = cm_data.pop("FLOUT", None)

# ── 4b. DERIVED FEATURES ──────────────────────────────────────────────────────
# TV: CoinGecko USD volume for ALL coins (not limited to CM's ~7)
cm_data["TV"] = volume.copy()
print(f"  TV    : {volume.notna().mean().mean():.1%} coverage (CoinGecko vol — all {len(SYMS)} coins)")

# NTR = market cap / TV  (all coins — real cross-sectional signal for expanded universe)
tv_nz = volume.replace(0, np.nan)
ntr   = mcap.div(tv_nz).replace([np.inf, -np.inf], np.nan)
cm_data["NTR"] = ntr
print(f"  NTR   : {ntr.notna().mean().mean():.1%} coverage (MCAP/TV — all coins)")

# NF / TF from CM exchange flows (major coins only; NaN elsewhere → imputed)
if flin is not None and flout is not None:
    cm_data["NF"] = flin.sub(flout)
    cm_data["TF"] = flin.add(flout)
    print(f"  NF    : {cm_data['NF'].notna().mean().mean():.1%} coverage (CM FlowIn-FlowOut)")
    print(f"  TF    : {cm_data['TF'].notna().mean().mean():.1%} coverage (CM FlowIn+FlowOut)")

# ── 5. MACRO FACTORS ──────────────────────────────────────────────────────────
print("\nFetching macro factors...")
MACRO_YF = {"^GSPC": "SPX", "^VIX": "VIX", "DX-Y.NYB": "USDX", "GLD": "GLD", "CL=F": "WTI"}
macro = {}
for ticker, label in MACRO_YF.items():
    s = yf.download(ticker, start=START, end=END, progress=False,
                    auto_adjust=True)["Close"].squeeze()
    s.index = pd.to_datetime(s.index)
    macro[label] = s.reindex(date_idx, method="ffill")
    print(f"  {label}: {macro[label].notna().mean():.0%}")

try:
    ff  = pdr.DataReader("F-F_Research_Data_Factors_daily", "famafrench", START, END)[0] / 100
    mom = pdr.DataReader("F-F_Momentum_Factor_daily",       "famafrench", START, END)[0] / 100
    ff.index  = pd.to_datetime(ff.index)
    mom.index = pd.to_datetime(mom.index)
    macro["RMRF"] = ff["Mkt-RF"].reindex(date_idx, method="ffill")
    macro["SMB"]  = ff["SMB"].reindex(date_idx, method="ffill")
    macro["UMD"]  = mom.iloc[:, 0].reindex(date_idx, method="ffill")
    print("  FF RMRF/SMB/UMD: OK")
except Exception as e:
    print(f"  FF FAIL: {e}")

print("\nFetching Google Trends...")
def _fetch_trends(keywords, label):
    try:
        pt = TrendReq(hl='en-US', tz=360, timeout=(10, 25))
        pt.build_payload(keywords, timeframe=f"{START} {END}")
        df = pt.interest_over_time()
        if df.empty: raise ValueError("empty")
        if 'isPartial' in df.columns: df = df.drop(columns=['isPartial'])
        s = df.mean(axis=1); s.index = pd.to_datetime(s.index)
        result = s.reindex(date_idx, method="ffill")
        print(f"  {label}: OK ({df.shape[0]} pts)")
        return result
    except Exception as e:
        print(f"  {label}: FAIL ({type(e).__name__}: {str(e)[:50]}) -- NaN")
        return pd.Series(np.nan, index=date_idx)

macro["GT"]   = _fetch_trends(["Bitcoin"], "GT")
time.sleep(30)
macro["GTCX"] = _fetch_trends(["Bitcoin", "BTC", "cryptocurrency", "blockchain", "crypto"], "GTCX")

print("\nFetching EPU + TEU from FRED...")
for label, sid in [("EPU", "USEPUINDXD"), ("TEU", "EPUITU")]:
    try:
        s = pdr.DataReader(sid, "fred", START, END).squeeze()
        s.index = pd.to_datetime(s.index)
        macro[label] = s.reindex(date_idx, method="ffill")
        print(f"  {label}: OK ({macro[label].notna().mean():.0%})")
    except Exception as e:
        print(f"  {label}: FAIL ({type(e).__name__}) -- NaN")
        macro[label] = pd.Series(np.nan, index=date_idx)

# ── 6. BUILD FEATURE DICT ─────────────────────────────────────────────────────
def _broadcast(s, date_idx, cols):
    arr = np.tile(s.reindex(date_idx, method="ffill").values[:, None], (1, len(cols)))
    return pd.DataFrame(arr, index=date_idx, columns=cols)

features = {}
for label, df in cm_data.items():
    features[label] = df
for label, s in macro.items():
    features[label] = _broadcast(s, date_idx, prices.columns)
features["MCAP"] = mcap   # from CoinGecko (all coins, used for universe filter + VW)

volume = volume.reindex(index=date_idx, columns=prices.columns)

# ── 7. UNIVERSE MASK (dynamic market cap filter) ──────────────────────────────
universe = (mcap >= MIN_MCAP_USD).fillna(False)

# ── 8. FIELD SPECS ────────────────────────────────────────────────────────────
specs = {
    **{k: FieldSpec(lag=LAG_BARS, missing="ffill", max_staleness=7)  for k in cm_data},
    **{k: FieldSpec(lag=LAG_BARS, missing="ffill", max_staleness=3)
       for k in ["SPX", "VIX", "USDX", "GLD", "WTI", "RMRF", "SMB", "UMD"]},
    **{k: FieldSpec(lag=LAG_BARS, missing="ffill", max_staleness=35)
       for k in ["GT", "GTCX", "EPU", "TEU"]},
    "MCAP":   FieldSpec(lag=0, missing="ffill", max_staleness=7),
    "volume": FieldSpec(lag=0, missing="ffill", max_staleness=3),
}

# ── 9. BUILD DataPanel ────────────────────────────────────────────────────────
panel = DataPanel(prices, volume=volume, features=features, universe=universe,
                  specs=specs, check_outliers=False)
print(f"\n-- DataPanel --")
print(f"  Bars    : {len(panel.dates)}")
print(f"  Assets  : {len(panel.assets_all)}")
print(f"  Range   : {panel.dates[0].date()} to {panel.dates[-1].date()}")
avail = [f for f in panel.available_fields() if f not in ('prices', 'volume', 'MCAP')]
print(f"  Factors : {len(avail)} -- {avail}")

# ── CELL 2: Strategy ──────────────────────────────────────────────────────────
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from backtest.strategy import Strategy
from backtest.data import DataView

FACTOR_COLS = [
    "AA", "NT", "MRR",           # on-chain core (CM community)
    "TV", "NTR",                  # volume + NVT ratio (CoinGecko — all coins)
    "NF", "TF",                   # exchange flows (CM — major coins; NaN elsewhere)
    "RMRF", "SMB", "UMD",         # Fama-French
    "SPX", "VIX", "USDX", "GLD", "WTI",   # macro
    "GT", "GTCX", "EPU", "TEU",   # sentiment + uncertainty (NaN fallback if fetch fails)
]
LEVEL_DIFF_FACTORS = frozenset({"NF", "TF"})   # can be negative → diff(), not pct_change()
RETURN_FACTORS     = frozenset({"RMRF", "SMB", "UMD"})  # daily returns → weekly sum
OLS3_COLS          = ["MRR", "AA", "NT"]
NN_LAYERS          = {
    "nn1": (32,), "nn2": (32, 16), "nn3": (32, 16, 8),
    "nn4": (32, 16, 8, 4), "nn5": (32, 16, 8, 4, 2),
}
ALL_MODEL_TYPES = [
    "ols", "ols3", "pls", "lasso", "ridge", "enet",
    "rf", "gbrt", "knn", "nn1", "nn2", "nn3", "nn4", "nn5",
]


def _make_model(model_type):
    mt = model_type.lower()
    if mt in ("ols", "ols3"): return LinearRegression()
    if mt == "pls":           return PLSRegression(n_components=3)
    if mt == "lasso":         return Lasso(alpha=1e-3, max_iter=5000)
    if mt == "ridge":         return Ridge(alpha=1.0)
    if mt == "enet":          return ElasticNet(alpha=1e-3, l1_ratio=0.5, max_iter=5000)
    if mt == "rf":
        return RandomForestRegressor(n_estimators=200, max_depth=6,
                                     min_samples_leaf=5, n_jobs=-1, random_state=42)
    if mt == "gbrt":
        return GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                         learning_rate=0.05, subsample=0.8, random_state=42)
    if mt == "knn":           return KNeighborsRegressor(n_neighbors=5, weights="distance", n_jobs=-1)
    if mt in NN_LAYERS:
        return MLPRegressor(hidden_layer_sizes=NN_LAYERS[mt], max_iter=500,
                            early_stopping=True, validation_fraction=0.1,
                            random_state=42, learning_rate_init=1e-3)
    raise ValueError(f"Unknown: {model_type!r}")


class CryptoMLFactorStrategy(Strategy):
    rebalance_frequency = "W-SUN"

    def __init__(self, model_type="rf", min_train_samples=100, n_deciles=10,
                 gross_exposure=0.5, train_start=None):
        if model_type.lower() not in ALL_MODEL_TYPES:
            raise ValueError(f"model_type must be one of {ALL_MODEL_TYPES}")
        self.model_type        = model_type.lower()
        self.min_train_samples = min_train_samples
        self.n_deciles         = n_deciles
        self.gross_exposure    = gross_exposure
        self.train_start       = pd.Timestamp(train_start) if train_start is not None else None
        self.feature_cols_     = OLS3_COLS if self.model_type == "ols3" else FACTOR_COLS
        self._pipeline         = None
        self._fitted_cols_     = []

    def required_data(self):
        return ["volume", "MCAP"]

    def _to_weekly(self, daily_map):
        out = {}
        for col, df in daily_map.items():
            if col in RETURN_FACTORS:
                out[col] = df.resample("W-SUN").sum()
            elif col in LEVEL_DIFF_FACTORS:
                # NF/TF can be negative → pct_change crosses zero; use level diff instead
                weekly = df.resample("W-SUN").last()
                out[col] = weekly.diff()
            else:
                w = df.resample("W-SUN").last().pct_change()
                out[col] = w.replace([np.inf, -np.inf], np.nan)
        return out

    def _build_Xy(self, weekly, prices, feature_cols):
        fwdret  = prices.resample("W-SUN").last().pct_change().shift(-1)
        ref_idx = weekly[feature_cols[0]].index
        weeks   = sorted(set(fwdret.index[:-1]) & set(ref_idx))
        Xs, ys  = [], []
        for dt in weeks:
            y_row = fwdret.loc[dt].dropna()
            if len(y_row) < 3: continue
            assets = y_row.index
            X_cs = pd.DataFrame(
                {col: weekly[col].loc[dt] if dt in weekly[col].index
                       else pd.Series(np.nan, index=prices.columns)
                 for col in feature_cols}
            ).loc[assets]
            valid = assets[X_cs.notna().any(axis=1) & y_row.notna()]
            if len(valid) < 3: continue
            Xs.append(X_cs.loc[valid].to_numpy(dtype=float, na_value=np.nan))
            ys.append(y_row.loc[valid].to_numpy(dtype=float, na_value=np.nan))
        if not Xs:
            return np.empty((0, len(feature_cols))), np.empty(0)
        return np.vstack(Xs), np.concatenate(ys)

    def fit(self, data):
        avail = [col for col in self.feature_cols_ if data.has_feature(col)]
        if not avail:
            self._pipeline = None; return
        daily  = {col: data.feature(col) for col in avail}
        prices = data.prices
        if self.train_start is not None:
            daily  = {col: df.loc[self.train_start:] for col, df in daily.items()}
            prices = prices.loc[self.train_start:]
        weekly  = self._to_weekly(daily)
        n_weeks = weekly[avail[0]].dropna(how="all").shape[0]
        if n_weeks < self.min_train_samples:
            self._pipeline = None; return
        X, y = self._build_Xy(weekly, prices, avail)
        if len(X) == 0:
            self._pipeline = None; return
        nan_pct = np.isnan(X).mean(axis=0)
        keep    = np.where(nan_pct < 0.95)[0]
        if len(keep) == 0:
            self._pipeline = None; return
        X = X[:, keep]
        self._fitted_cols_ = [avail[i] for i in keep]
        base = _make_model(self.model_type)
        if isinstance(base, PLSRegression):
            nc = min(3, len(self._fitted_cols_), X.shape[1])
            base = PLSRegression(n_components=nc)
        self._pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("model",   base),
        ])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._pipeline.fit(X, y)

    def generate_weights(self, data: DataView, t: pd.Timestamp) -> pd.Series:
        if self._pipeline is None or not self._fitted_cols_:
            return pd.Series(dtype=float)
        daily  = {col: data.feature(col) for col in self._fitted_cols_}
        weekly = self._to_weekly(daily)
        snap = {
            col: weekly[col].iloc[-1] if not weekly[col].empty
                 else pd.Series(np.nan, index=data.prices.columns)
            for col in self._fitted_cols_
        }
        X_now  = pd.DataFrame(snap)
        valid  = X_now.index[X_now.notna().any(axis=1)]
        if len(valid) < self.n_deciles:
            return pd.Series(dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds = self._pipeline.predict(
                X_now.loc[valid].to_numpy(dtype=float, na_value=np.nan)
            ).ravel()
        pred_s = pd.Series(preds, index=valid)
        ranks  = pred_s.rank(pct=True)
        lo     = 1.0 / self.n_deciles
        hi     = (self.n_deciles - 1) / self.n_deciles
        long_  = ranks[ranks >= hi].index
        short_ = ranks[ranks <= lo].index
        mcap_s = data.feature("MCAP").iloc[-1]
        def _vw(idx):
            cap = mcap_s.loc[idx].clip(lower=0).fillna(0)
            tot = cap.sum()
            return cap / tot if tot > 0 else pd.Series(1.0 / len(idx), index=idx)
        w = pd.Series(0.0, index=valid)
        if len(long_):  w[long_]  =  _vw(long_)  * self.gross_exposure
        if len(short_): w[short_] = -_vw(short_) * self.gross_exposure
        return w

    def apply_risk(self, proposed, state, data):
        return proposed


print(f"Strategy OK — {len(FACTOR_COLS)} factors declared  "
      f"(LEVEL_DIFF: {sorted(LEVEL_DIFF_FACTORS)})")

# ── CELL 3: Costs ─────────────────────────────────────────────────────────────
from backtest.costs import (
    Commission, Spread, MarketImpact, ShortBorrow,
    CompositeCostModel, LiquidityCap,
)

costs = CompositeCostModel([
    Commission(bps=10),
    Spread(half_bps=15),
    MarketImpact(k=10, kind="sqrt", adv_lookback=20),
    ShortBorrow(annual_bps=300, trading_days=252),
])
liq = LiquidityCap(cap_pct=0.05, adv_lookback=20)
print("Costs OK")

# ── CELL 4: Rolling Walk-Forward ──────────────────────────────────────────────
from backtest.engine import Engine

TRAIN_YRS  = 7   # 4yr paper train + 3yr paper validation
FIRST_TEST = 2021
LAST_TEST  = 2026

windows = []
panel_end = panel.dates[-1]
for year in range(FIRST_TEST, LAST_TEST + 1):
    test_start  = pd.Timestamp(f"{year}-01-01")
    test_end    = min(pd.Timestamp(f"{year}-12-31"), panel_end)
    fit_start   = test_start - pd.DateOffset(years=TRAIN_YRS)
    fit_end     = test_start - pd.Timedelta(days=1)
    train_dates = panel.dates[(panel.dates >= fit_start) & (panel.dates <= fit_end)]
    test_dates  = panel.dates[(panel.dates >= test_start) & (panel.dates <= test_end)]
    if len(train_dates) == 0 or len(test_dates) == 0:
        continue
    windows.append(dict(year=year, fit_start=fit_start, fit_end=fit_end,
                        test_start=test_start, test_end=test_end,
                        train_dates=train_dates, test_dates=test_dates))

print(f"\nRolling walk-forward: {TRAIN_YRS}yr fit, 1yr OOS, {len(windows)} windows")
print(f"{'Year':>5}  {'Fit window':^25}  {'OOS window':^25}  {'Train':>6}  {'Test':>5}")
for w in windows:
    print(f"  {w['year']}  {str(w['fit_start'].date()):>12} -> {str(w['fit_end'].date()):<12}  "
          f"{str(w['test_start'].date()):>12} -> {str(w['test_end'].date()):<12}  "
          f"{len(w['train_dates']):>5}  {len(w['test_dates']):>5}")

results = []
for w in windows:
    print(f"\n[{w['year']}] Fit {w['fit_start'].date()} to {w['fit_end'].date()} "
          f"| OOS {w['test_start'].date()} to {w['test_end'].date()}")
    strat  = CryptoMLFactorStrategy(model_type="rf", train_start=w["fit_start"])
    engine = Engine(panel, strat, costs=costs, liquidity=liq, initial_capital=1_000_000)
    result = engine.run(train_dates=w["train_dates"], test_dates=w["test_dates"])
    tr     = result.equity_curve.iloc[-1] / result.metadata["initial_capital"] - 1
    fitted = strat._fitted_cols_
    print(f"  Return: {tr:>+8.2%} | Rebals: {result.metadata['n_rebalances']} | "
          f"Costs: ${result.costs.sum():>9,.0f} | Factors: {len(fitted)}")
    print(f"  Fitted: {fitted}")
    results.append((w["year"], result))

all_rets        = pd.concat([r.returns.dropna() for _, r in results]).sort_index()
stitched_equity = (1 + all_rets).cumprod() * 1_000_000
total_ret       = stitched_equity.iloc[-1] / 1_000_000 - 1
span_days       = (all_rets.index[-1] - all_rets.index[0]).days
ann_ret         = (1 + total_ret) ** (365 / span_days) - 1
ann_vol         = all_rets.std() * np.sqrt(252)
sharpe          = ann_ret / ann_vol if ann_vol > 0 else float("nan")
peak            = np.maximum.accumulate(stitched_equity.values)
max_dd          = ((stitched_equity.values - peak) / np.where(peak > 0, peak, 1)).min()
total_costs     = sum(r.costs.sum() for _, r in results)

print(f"\n{'='*60}")
print(f"Rolling Walk-Forward  (RF · {TRAIN_YRS}yr fit · 1yr step · {len(windows)} windows)")
print(f"Universe  : {len(SYMS)} coins  (CoinGecko top-{UNIVERSE_SIZE}, dynamic, no hardcoding)")
print(f"OOS period: {all_rets.index[0].date()} to {all_rets.index[-1].date()}")
print(f"{'='*60}")
print(f"Total return     : {total_ret:>+10.2%}")
print(f"Ann. return      : {ann_ret:>+10.2%}")
print(f"Ann. volatility  : {ann_vol:>+10.2%}")
print(f"Sharpe ratio     : {sharpe:>+10.2f}")
print(f"Max drawdown     : {max_dd:>+10.2%}")
print(f"Final equity     : ${stitched_equity.iloc[-1]:>12,.0f}")
print(f"Total costs      : ${total_costs:>12,.0f}")
print(f"\n{'Year':>5}  {'Return':>8}  {'Vol':>7}  {'SR':>6}  {'MaxDD':>8}  {'Costs':>10}  {'Rebals':>7}")
for year, res in results:
    r      = res.returns.dropna()
    yr_ret = res.equity_curve.iloc[-1] / res.metadata["initial_capital"] - 1
    yr_vol = r.std() * np.sqrt(252)
    yr_sr  = (r.mean() * 252) / yr_vol if yr_vol > 0 else float("nan")
    pk     = np.maximum.accumulate(res.equity_curve.values)
    yr_dd  = ((res.equity_curve.values - pk) / np.where(pk > 0, pk, 1)).min()
    print(f"  {year}  {yr_ret:>+8.2%}  {yr_vol:>7.2%}  {yr_sr:>6.2f}  "
          f"{yr_dd:>+8.2%}  ${res.costs.sum():>9,.0f}  {res.metadata['n_rebalances']:>6}")

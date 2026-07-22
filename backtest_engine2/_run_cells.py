# Cell 1 — Setup + hourly Binance + precomputed entry signals (Stage 1)
# Strategy: ou-optimal-trend-pullback (crypto). Stocks = later cell.
# Entry (precomputed): top-N liquid | uptrend (sum log ret > r_min) | close < SMA | last bar down (3B)
# FAST_MODE: smaller universe for smoke tests (Cell 4); full run sets FAST_MODE=False.

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, r"C:\Users\User\backtest_engine")
from backtest.data import DataPanel, FieldSpec

# ── FAST_MODE / crypto params (Cell 2 will mirror these) ─────────────────────
FAST_MODE = True

CRYPTO_TOP_N = 20 if FAST_MODE else 50
W_LIQ = 168          # median quote-volume lookback (hours)
T_TREND = 168        # uptrend window (hours)
R_MIN = 0.02         # min cumulative log return for uptrend (2a)
L_MA = 24            # pullback SMA (hours)
RECENT_DOWN_BARS = 1 # 3B: require negative log return on last N bar(s)

START_UTC = pd.Timestamp("2019-01-01", tz="UTC")
END_UTC = pd.Timestamp.now(tz="UTC").floor("h")
MATIC_TO_POL_DATE = pd.Timestamp("2024-09-04", tz="UTC")

CACHE_DIR = Path("data")
CACHE_DIR.mkdir(exist_ok=True)
BINANCE_CACHE = CACHE_DIR / "binance_hourly_top50_pool.parquet"

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

session = requests.Session()
binance_usdt = fetch_binance_usdt_set(session)

raw = pd.read_parquet(BINANCE_CACHE)
raw["open_time"] = pd.to_datetime(raw["open_time"], utc=True)
have = set(raw["symbol"].unique())
target_symbols = sorted(s for s in ANCHOR_BINANCE if s in binance_usdt or s in have)
# Full pool from cache; FAST_MODE only lowers CRYPTO_TOP_N (20 vs 50).
print(f"Loaded {BINANCE_CACHE.name}: {len(have)} symbols, {len(raw):,} rows")
print(f"FAST_MODE={FAST_MODE}  CRYPTO_TOP_N={CRYPTO_TOP_N}")

start_ms = int(START_UTC.timestamp() * 1000)
end_ms = int(END_UTC.timestamp() * 1000)
need = [s for s in target_symbols if s not in have and s != "POLUSDT"]
if need and not FAST_MODE:
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

# ── 2) Vectorized entry features (no lookahead: all use data <= t) ─────────────
min_liq = max(W_LIQ // 2, 24)
min_trend = max(T_TREND // 2, 24)
min_ma = max(L_MA // 2, 12)

log_ret = np.log(close / close.shift(1))

liq_score = qvol.rolling(W_LIQ, min_periods=min_liq).median()
liq_rank = liq_score.rank(axis=1, ascending=False, method="min")
liquid_top_n = liq_rank <= CRYPTO_TOP_N

roll_trend = log_ret.rolling(T_TREND, min_periods=min_trend).sum()
trend_up = roll_trend > R_MIN

sma = close.rolling(L_MA, min_periods=min_ma).mean()
pullback = close < sma

if RECENT_DOWN_BARS == 1:
    recent_down = log_ret < 0
else:
    recent_down = log_ret.rolling(RECENT_DOWN_BARS, min_periods=RECENT_DOWN_BARS).sum() < 0

entry_signal = liquid_top_n & trend_up & pullback & recent_down

features = {
    "liquid_top_n": liquid_top_n.astype(np.float64),
    "trend_up": trend_up.astype(np.float64),
    "pullback": pullback.astype(np.float64),
    "recent_down": recent_down.astype(np.float64),
    "entry_signal": entry_signal.astype(np.float64),
    "roll_trend": roll_trend.astype(np.float64),
    "sma_pullback": sma.astype(np.float64),
}

feat_spec = FieldSpec(lag=0, missing="drop")
specs = {
    "prices": FieldSpec(lag=0),
    "volume": FieldSpec(lag=0),
    **{k: feat_spec for k in features},
}

panel = DataPanel(
    prices=close,
    volume=qvol,
    features=features,
    specs=specs,
    check_outliers=True,
)

# ── 3) Validation output ─────────────────────────────────────────────────────
n_entry = int(entry_signal.sum().sum())
n_liquid = int(liquid_top_n.sum().sum())
by_year = entry_signal.sum(axis=1).resample("YE").sum()

print(f"\n-- DataPanel (ou-optimal-trend-pullback Cell 1) --")
print(f"FAST_MODE          : {FAST_MODE}")
print(f"Bars               : {len(panel.dates):,}")
print(f"Assets             : {len(panel.assets_all)}")
print(f"Range              : {panel.dates[0]} -> {panel.dates[-1]}")
print(f"NaN prices         : {close.isna().mean().mean():.2%}")
print(f"NaN volume         : {qvol.isna().mean().mean():.2%}")
print(f"Fields             : {panel.available_fields()}")
print(f"\nEntry params: TOP_N={CRYPTO_TOP_N} W_LIQ={W_LIQ} T_TREND={T_TREND} "
      f"R_MIN={R_MIN} L_MA={L_MA} recent_down_bars={RECENT_DOWN_BARS}")
print(f"Bar×asset True: liquid={n_liquid:,}  entry_signal={n_entry:,}")
print(f"\nEntry signals per calendar year (sum over assets):")
print(by_year.to_string())

print("\nentry_signal.tail(3) [BTC, ETH, SOL if present]:")
_show = [c for c in ["BTCUSDT", "ETHUSDT", "SOLUSDT"] if c in close.columns]
print(entry_signal[_show].tail(3))

print("\nclose.tail(2):")
print(close[_show].tail(2))

# Cell 2 — OuTrendPullbackStrategy + fused Numba OU calibration (Stage 2)
# Entry: precomputed entry_signal (Cell 1). Exit: spec pi+/pi- + max_holding_period.

import os
import subprocess
import sys
import time

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numba"])
    from numba import njit, prange
    HAS_NUMBA = True

N_CORES = os.cpu_count() or 8
os.environ["NUMBA_NUM_THREADS"] = str(N_CORES)
os.environ.setdefault("OMP_NUM_THREADS", str(N_CORES))

import numpy as np
import pandas as pd

from backtest.strategy import Strategy

# ── Params (keep aligned with Cell 1) ─────────────────────────────────────────
_FAST = globals().get("FAST_MODE", True)
_NUM_PATHS = 10_000 if _FAST else 100_000
_MAX_HOLD = 100
_OLS_WINDOW = 168
_MAX_POSITIONS = 10
_WEIGHT_PER_POSITION = 0.05
_MAX_NEW_ENTRIES_PER_BAR = 3

_PI_PLUS_GRID = np.arange(0.5, 10.0 + 0.5 * 0.5, 0.5, dtype=np.float64)
_PI_MINUS_GRID = np.arange(-10.0, -0.5 + 0.5 * 0.5, 0.5, dtype=np.float64)


@njit(cache=True)
def _fit_ou_numba(s: np.ndarray, m: float):
    n = len(s)
    if n < 10:
        return np.nan, np.nan
    y = np.empty(n - 1)
    x = np.empty(n - 1)
    for i in range(n - 1):
        y[i] = s[i + 1] - s[i]
        x[i] = s[i] - m
    mx = 0.0
    my = 0.0
    for i in range(n - 1):
        mx += x[i]
        my += y[i]
    mx /= n - 1
    my /= n - 1
    var_x = 0.0
    cov_xy = 0.0
    for i in range(n - 1):
        dx = x[i] - mx
        dy = y[i] - my
        var_x += dx * dx
        cov_xy += dx * dy
    if n - 1 > 1:
        var_x /= n - 2
        cov_xy /= n - 2
    if var_x < 1e-14:
        return np.nan, np.nan
    phi = 1.0 + cov_xy / var_x
    ss = 0.0
    for i in range(n - 1):
        resid = y[i] - (phi - 1.0) * x[i]
        ss += resid * resid
    sigma = np.sqrt(ss / (n - 2)) if n > 2 else np.nan
    if not np.isfinite(phi) or not np.isfinite(sigma) or sigma <= 0.0:
        return np.nan, np.nan
    if phi <= -1.0 or phi >= 1.0:
        return np.nan, np.nan
    return phi, sigma


@njit(parallel=True, cache=True)
def _calibrate_ou_fused(
    phi: float,
    sigma: float,
    s0: float,
    m: float,
    n_paths: int,
    max_h: int,
    seed: int,
    pi_plus_grid: np.ndarray,
    pi_minus_grid: np.ndarray,
):
    """Simulate n_paths once, score full Cartesian grid in compiled code."""
    np.random.seed(seed & 0xFFFFFFFF)
    n_steps = max_h + 1
    pi_paths = np.empty((n_paths, n_steps))
    for i in prange(n_paths):
        si = s0
        pi_paths[i, 0] = 0.0
        for step in range(1, n_steps):
            eps = np.random.randn()
            si = si + (1.0 - phi) * (m - si) + sigma * eps
            pi_paths[i, step] = si - s0

    n_plus = pi_plus_grid.shape[0]
    n_minus = pi_minus_grid.shape[0]
    best_sharpe = -1.0e300
    best_plus = pi_plus_grid[0]
    best_minus = pi_minus_grid[0]

    for ip in range(n_plus):
        pi_p = pi_plus_grid[ip]
        for im in range(n_minus):
            pi_m = pi_minus_grid[im]
            sum_p = 0.0
            sum_sq = 0.0
            for i in range(n_paths):
                exit_idx = max_h
                for j in range(1, n_steps):
                    pi = pi_paths[i, j]
                    if pi >= pi_p or pi <= pi_m:
                        exit_idx = j
                        break
                pnl = pi_paths[i, exit_idx]
                sum_p += pnl
                sum_sq += pnl * pnl
            mean = sum_p / n_paths
            var = (sum_sq - n_paths * mean * mean) / (n_paths - 1) if n_paths > 1 else 0.0
            if var < 1e-24:
                continue
            sharpe = mean / np.sqrt(var)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_plus = pi_p
                best_minus = pi_m

    if best_sharpe <= -1.0e299:
        return np.nan, np.nan, np.nan
    return best_plus, best_minus, best_sharpe


class OuTrendPullbackStrategy(Strategy):
    """Trend + pullback entry (Cell 1 features) + O-U optimal exit thresholds (spec)."""

    rebalance_frequency = "D"

    def __init__(
        self,
        fast_mode: bool = _FAST,
        num_simulated_paths: int | None = None,
        max_holding_period: int = _MAX_HOLD,
        ou_ols_window: int = _OLS_WINDOW,
        max_positions: int = _MAX_POSITIONS,
        weight_per_position: float = _WEIGHT_PER_POSITION,
        max_new_entries_per_bar: int = _MAX_NEW_ENTRIES_PER_BAR,
        pi_plus_grid: np.ndarray | None = None,
        pi_minus_grid: np.ndarray | None = None,
    ):
        self.fast_mode = fast_mode
        self.num_simulated_paths = (
            num_simulated_paths if num_simulated_paths is not None
            else (10_000 if fast_mode else 100_000)
        )
        self.max_holding_period = max_holding_period
        self.ou_ols_window = ou_ols_window
        self.max_positions = max_positions
        self.weight_per_position = weight_per_position
        self.max_new_entries_per_bar = max_new_entries_per_bar
        self._pi_plus_grid = (
            pi_plus_grid if pi_plus_grid is not None else _PI_PLUS_GRID.copy()
        )
        self._pi_minus_grid = (
            pi_minus_grid if pi_minus_grid is not None else _PI_MINUS_GRID.copy()
        )
        self.open_trades: dict = {}
        self.trade_log: list = []       # one row per closed trade (round-trip)
        self.decision_log: list = []    # every entry/exit/reject decision
        self._event_this_bar = False
        self._n_entries = 0
        self._n_exits = 0
        self._n_ou_skipped = 0

    def _log_decision(self, decision: str, t, asset: str, **fields):
        row = {"decision": decision, "t": t, "asset": asset}
        row.update(fields)
        self.decision_log.append(row)

    @staticmethod
    def _half_life(phi: float):
        if 0.0 < phi < 1.0:
            return float(-np.log(2.0) / np.log(phi))
        return np.nan

    def _entry_context(self, data, asset: str) -> dict:
        ctx = {}
        for name in ("entry_signal", "trend_up", "pullback", "recent_down", "roll_trend"):
            try:
                f = data.feature(name)
                if asset in f.index:
                    v = f[asset]
                    if np.isfinite(v):
                        ctx[name] = float(v)
            except (KeyError, AttributeError, TypeError):
                pass
        return ctx

    def __repr__(self):
        return (
            f"OuTrendPullbackStrategy(fast={self.fast_mode},paths={self.num_simulated_paths},"
            f"hold={self.max_holding_period},ols={self.ou_ols_window},"
            f"maxpos={self.max_positions},w={self.weight_per_position},"
            f"max_new={self.max_new_entries_per_bar})"
        )

    def required_data(self):
        return {"prices": None, "volume": None, "entry_signal": None}

    @staticmethod
    def _fit_ou(prices: np.ndarray, m: float):
        if HAS_NUMBA:
            return _fit_ou_numba(prices.astype(np.float64), float(m))
        s = np.asarray(prices, dtype=float)
        if len(s) < 10 or np.isnan(s).any():
            return None
        y = np.diff(s)
        x = s[:-1] - m
        var_x = np.var(x, ddof=1)
        if var_x < 1e-14:
            return None
        phi = 1.0 + np.cov(y, x, ddof=1)[0, 1] / var_x
        resid = y - (phi - 1.0) * x
        sigma = float(np.std(resid, ddof=1))
        if not np.isfinite(phi) or not np.isfinite(sigma) or sigma <= 0:
            return None
        if not (-1.0 < phi < 1.0):
            return None
        return float(phi), sigma

    def _calibrate_thresholds(self, phi, sigma, p0, m, seed: int):
        if not HAS_NUMBA:
            raise RuntimeError("Numba required for OU calibration speed path")
        return _calibrate_ou_fused(
            float(phi),
            float(sigma),
            float(p0),
            float(m),
            int(self.num_simulated_paths),
            int(self.max_holding_period),
            int(seed),
            self._pi_plus_grid,
            self._pi_minus_grid,
        )

    def _try_enter(self, asset: str, prices: pd.Series, t, ctx: dict | None = None) -> bool:
        ctx = dict(ctx or {})
        hist = prices[asset].dropna()
        if len(hist) < self.ou_ols_window:
            self._n_ou_skipped += 1
            self._log_decision(
                "entry_reject", t, asset, reason="short_history",
                hist_len=int(len(hist)), need=self.ou_ols_window, **ctx,
            )
            return False
        window = hist.iloc[-self.ou_ols_window :]
        p0 = float(window.iloc[-1])
        m = float(window.mean())
        ou = self._fit_ou(window.values, m)
        if ou is None:
            self._n_ou_skipped += 1
            self._log_decision(
                "entry_reject", t, asset, reason="ou_fit_fail",
                p0=p0, m0=m, **ctx,
            )
            return False
        phi, sigma = ou
        seed = hash((asset, str(t))) % (2**32)
        cal = self._calibrate_thresholds(phi, sigma, p0, m, seed)
        pi_plus, pi_minus, ou_sharpe = cal
        if not (np.isfinite(pi_plus) and np.isfinite(pi_minus) and np.isfinite(ou_sharpe)):
            self._n_ou_skipped += 1
            self._log_decision(
                "entry_reject", t, asset, reason="ou_calibrate_fail",
                p0=p0, m0=m, phi=float(phi), sigma=float(sigma), **ctx,
            )
            return False
        dip_pct = (p0 - m) / p0 if p0 else np.nan
        trade = {
            "entry_t": t,
            "p0": p0,
            "m0": m,
            "phi": float(phi),
            "sigma": float(sigma),
            "half_life": self._half_life(float(phi)),
            "pi_plus": float(pi_plus),
            "pi_minus": float(pi_minus),
            "ou_sharpe": float(ou_sharpe),
            "bars_held": 0,
            "entry_ctx": ctx,
        }
        self.open_trades[asset] = trade
        self._n_entries += 1
        self._event_this_bar = True
        self._log_decision(
            "entry_open", t, asset, reason="accepted",
            p0=p0, m0=m, dip_vs_mean_pct=float(dip_pct),
            phi=float(phi), sigma=float(sigma),
            half_life=trade["half_life"],
            pi_plus=float(pi_plus), pi_minus=float(pi_minus),
            ou_sharpe=float(ou_sharpe), weight=self.weight_per_position,
            **ctx,
        )
        return True

    def generate_weights(self, data, t):
        self._event_this_bar = False
        prices = data.prices
        entry_sig = data.feature("entry_signal")
        eligible = set(data.assets)
        weights = pd.Series(0.0, index=prices.index)

        for asset in list(self.open_trades.keys()):
            if asset not in eligible:
                weights[asset] = self.weight_per_position
                continue
            p = prices.get(asset)
            if p is None or not np.isfinite(p):
                weights[asset] = self.weight_per_position
                continue

            trade = self.open_trades[asset]
            trade["bars_held"] += 1
            pi_t = float(p) - trade["p0"]
            exit_now = (
                pi_t >= trade["pi_plus"]
                or pi_t <= trade["pi_minus"]
                or trade["bars_held"] >= self.max_holding_period
            )
            if exit_now:
                reason = (
                    "pi_plus" if pi_t >= trade["pi_plus"]
                    else "pi_minus" if pi_t <= trade["pi_minus"]
                    else "timeout"
                )
                p_exit = float(p)
                ret_pct = pi_t / trade["p0"] if trade["p0"] else np.nan
                closed = {
                    "asset": asset,
                    "entry_t": trade["entry_t"],
                    "exit_t": t,
                    "bars_held": trade["bars_held"],
                    "p0": trade["p0"],
                    "p_exit": p_exit,
                    "m0": trade["m0"],
                    "pi_exit": pi_t,
                    "ret_pct": float(ret_pct),
                    "win": bool(pi_t > 0),
                    "exit_reason": reason,
                    "phi": trade["phi"],
                    "sigma": trade["sigma"],
                    "half_life": trade.get("half_life"),
                    "pi_plus": trade["pi_plus"],
                    "pi_minus": trade["pi_minus"],
                    "ou_sharpe": trade["ou_sharpe"],
                    "entry_ctx": trade.get("entry_ctx"),
                }
                self.trade_log.append(closed)
                self._log_decision(
                    "exit", t, asset, reason=reason,
                    p_exit=p_exit, pi_exit=pi_t, ret_pct=float(ret_pct),
                    win=bool(pi_t > 0), bars_held=trade["bars_held"],
                    pi_plus=trade["pi_plus"], pi_minus=trade["pi_minus"],
                    hit_pi_plus=bool(pi_t >= trade["pi_plus"]),
                    hit_pi_minus=bool(pi_t <= trade["pi_minus"]),
                    **(trade.get("entry_ctx") or {}),
                )
                del self.open_trades[asset]
                self._n_exits += 1
                self._event_this_bar = True
            else:
                weights[asset] = self.weight_per_position

        new_entries = 0
        for asset in eligible:
            if asset in self.open_trades:
                continue
            sig_val = entry_sig.get(asset, 0.0) if asset in entry_sig.index else 0.0
            if not (np.isfinite(sig_val) and sig_val > 0.5):
                continue
            ctx = self._entry_context(data, asset)
            if len(self.open_trades) >= self.max_positions:
                self._log_decision(
                    "entry_reject", t, asset, reason="max_positions",
                    n_open=len(self.open_trades), cap=self.max_positions, **ctx,
                )
                continue
            if new_entries >= self.max_new_entries_per_bar:
                self._log_decision(
                    "entry_reject", t, asset, reason="max_new_per_bar",
                    cap=self.max_new_entries_per_bar, **ctx,
                )
                continue
            if self._try_enter(asset, prices, t, ctx):
                weights[asset] = self.weight_per_position
                new_entries += 1

        return weights

    def apply_risk(self, proposed, state, data):
        equity = state["equity"]
        if not self._event_this_bar and equity > 0:
            proposed = state["positions"] / equity
        return super().apply_risk(proposed, state, data)


strat = OuTrendPullbackStrategy(fast_mode=_FAST)
print(strat)
try:
    from numba import config as _numba_config
    _nth = _numba_config.NUMBA_NUM_THREADS
except Exception:
    _nth = "?"
print(f"Accel: cores={N_CORES}, numba={HAS_NUMBA}, numba_threads={_nth}")
print(
    f"O-U grid: {len(strat._pi_plus_grid)} x {len(strat._pi_minus_grid)} = "
    f"{len(strat._pi_plus_grid) * len(strat._pi_minus_grid)} nodes | "
    f"paths={strat.num_simulated_paths:,}"
)

rb = strat.rebalance_dates(panel.dates)
print(f"\nRebalance bars: {len(rb):,} (expect {len(panel.dates):,})")
print(f"required_data(): {strat.required_data()}")
print(f"Warmup hint (Cell 4): >= {max(_OLS_WINDOW, _MAX_HOLD, globals().get('T_TREND', 168), globals().get('W_LIQ', 168))} bars")
print(
    "Trade logging: strat.decision_log (every entry_open / entry_reject / exit), "
    "strat.trade_log (closed round-trips). Save parquet after Cell 4."
)

# ── JIT warmup + single calibration timing ────────────────────────────────────
_t0 = time.perf_counter()
_ = _calibrate_ou_fused(0.95, 0.01, 100.0, 100.0, 100, 10, 42, _PI_PLUS_GRID[:3], _PI_MINUS_GRID[:3])
print(f"Numba warmup (tiny grid): {(time.perf_counter() - _t0) * 1000:.0f} ms")

_test_asset = next((a for a in ["BTCUSDT", "ETHUSDT", "SOLUSDT"] if a in panel.assets_all), None)
if _test_asset is None:
    _test_asset = panel.assets_all[0]
_hist_full = close[_test_asset].dropna()
_t_probe = _hist_full.index[int(len(_hist_full) * 0.6)]
if len(_hist_full) >= _OLS_WINDOW:
    _w = _hist_full.loc[:_t_probe].iloc[-_OLS_WINDOW:]
    _p0 = float(_w.iloc[-1])
    _m = float(_w.mean())
    _ou = strat._fit_ou(_w.values, _m)
    print(f"\nProbe asset {_test_asset} @ {_t_probe}:")
    print(f"  O-U fit: {_ou}")
    if _ou:
        _t1 = time.perf_counter()
        _cal = strat._calibrate_thresholds(_ou[0], _ou[1], _p0, _m, 42)
        _dt = time.perf_counter() - _t1
        print(f"  Calibrate ({strat.num_simulated_paths:,} paths, full grid): {_dt:.2f}s -> {_cal}")
        _n_sig = int(entry_signal.sum().sum())
        if _n_sig > 0:
            print(
                f"  Worst-case if OU on every signal (not actual entries): "
                f"{_dt * _n_sig / 3600:.1f} hours"
            )
else:
    print(f"Probe: insufficient {_test_asset} history in close panel")

# Cell 3 — Costs + liquidity (Stage 3, mandatory per PROTOCOL)
# Long-only crypto hourly: Commission + Spread + sqrt impact; ShortBorrow kept for engine parity.

import pandas as pd

from backtest.costs import (
    Commission,
    Spread,
    MarketImpact,
    ShortBorrow,
    CompositeCostModel,
    LiquidityCap,
)

HOURS_PER_YEAR = 365 * 24
TAKER_BPS = 10
HALF_SPREAD_BPS = 5
IMPACT_K = 10
ADV_LOOKBACK = 168
LIQ_CAP_PCT = 0.10
BORROW_ANNUAL_BPS = 1500  # only applies if strategy shorts; long-only -> ~0 holding

costs = CompositeCostModel([
    Commission(bps=TAKER_BPS),
    Spread(half_bps=HALF_SPREAD_BPS),
    MarketImpact(k=IMPACT_K, kind="sqrt", adv_lookback=ADV_LOOKBACK),
    ShortBorrow(annual_bps=BORROW_ANNUAL_BPS, trading_days=HOURS_PER_YEAR),
])

liq = LiquidityCap(cap_pct=LIQ_CAP_PCT, adv_lookback=ADV_LOOKBACK)

print("Cost components:", [type(m).__name__ for m in costs.models])
for m in costs.models:
    if isinstance(m, Commission):
        print(f"  Commission        bps         = {m.bps}")
    elif isinstance(m, Spread):
        print(f"  Spread            half_bps    = {m.half_bps}")
    elif isinstance(m, MarketImpact):
        print(f"  MarketImpact      k={m.k}, kind={m.kind!r}, adv_lookback={m.adv_lookback}")
    elif isinstance(m, ShortBorrow):
        implied_annual_pct = m.daily_rate * HOURS_PER_YEAR * 100
        print(
            f"  ShortBorrow       per-bar rate={m.daily_rate:.3e}  "
            f"(annualized = {implied_annual_pct:.2f}%)"
        )

print(f"\nLiquidityCap      cap_pct={liq.cap_pct}, adv_lookback={liq.adv_lookback}")
print(f"Panel has volume  : {panel.has_field('volume')}  (required True)")

# Long-only smoke: $10k BTC buy on last bar with valid BTC price
_btc_idx = close["BTCUSDT"].dropna().index
if len(_btc_idx) == 0:
    _probe_t = panel.dates[-1]
    _probe_sym = panel.assets_all[0]
else:
    _probe_t = _btc_idx[-1]
    _probe_sym = "BTCUSDT"

sample_view = panel.as_of(_probe_t)
sample_trades = pd.Series({_probe_sym: 10_000.0}).reindex(panel.assets_all).fillna(0.0)
tc = costs.trade_cost(sample_trades, sample_view)
hc = costs.holding_cost(sample_trades, sample_view)
print(f"\nSmoke test ({_probe_t}, +$10k {_probe_sym} long):")
print(f"  trade_cost   = ${tc:,.2f}   ({tc / 10_000 * 1e4:.1f} bps of $10k notional)")
print(f"  holding_cost = ${hc:,.4f}   (long-only -> expect ~0)")
# Cell 2 — Hurst pairs Strategy class
import sys, subprocess
try:
    from statsmodels.tsa.stattools import coint
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "statsmodels"])
    from statsmodels.tsa.stattools import coint

from itertools import combinations
import numpy as np
import pandas as pd

from backtest.strategy import Strategy


class HurstPairsStrategy(Strategy):
    """Hurst-Crypto-Pairs trading (mathematics-12-02911).

    Entry: H_t<0.5 and spread in [m±1σ, m±2σ] band (windows ending t-1).
    Exit:  mean reversion / 2σ stop / 72h expiration.
    Pair selection: monthly top-K by lowest ADF p-value on OLS residual of log-price regression.

    Implementation notes:
      • b_t (hedge ratio) is recomputed at every bar from the rolling 30-day vol ratio
        of log returns ending at t-1 (paper-faithful).
      • Spreads in the rolling tw_hours window each use their own contemporaneous b.
      • All windows for m, std, H end at t-1 (no look-ahead).
      • State (selected pairs, open trades) lives on self → deep-copied per CPCV fold.
    """

    rebalance_frequency = "D"  # "D"/"daily"/"B" = every bar (here: every hour)

    def __init__(
        self,
        tw_hours: int = 168,
        hedge_lookback_hours: int = 720,
        hurst_q: int = 1,
        hurst_threshold: float = 0.5,
        entry_lower_std: float = 1.0,
        entry_upper_std: float = 2.0,
        stop_std: float = 2.0,
        max_hold_hours: int = 72,
        top_k_pairs: int = 20,
        coint_lookback_hours: int = 4320,   # 180 days
        pair_gross_alloc: float = 0.05,
    ):
        self.tw_hours = tw_hours
        self.hedge_lookback_hours = hedge_lookback_hours
        self.hurst_q = hurst_q
        self.hurst_threshold = hurst_threshold
        self.entry_lower_std = entry_lower_std
        self.entry_upper_std = entry_upper_std
        self.stop_std = stop_std
        self.max_hold_hours = max_hold_hours
        self.top_k_pairs = top_k_pairs
        self.coint_lookback_hours = coint_lookback_hours
        self.pair_gross_alloc = pair_gross_alloc

        # tau = 2^n for n = 0..floor(log2(TW))-2  (Eq. 4; TW=168 -> tau in {1,2,4,8,16,32})
        n_max = max(0, int(np.floor(np.log2(tw_hours))) - 2)
        self._taus = np.array([2 ** n for n in range(n_max + 1)], dtype=int)

        # State (deep-copied per CPCV path, so no leakage across folds)
        self._current_month = None
        self._selected_pairs = []         # list of (A, B) tuples for the current month
        self.open_trades = {}             # (A, B) -> {"entry_t", "side", "m0", "std0"}
        self._event_this_bar = False      # True iff open_trades mutated during the most recent
                                          # generate_weights call; used by apply_risk to suppress
                                          # mark-to-market micro-rebalancing trades on quiet bars.

    def __repr__(self):
        return (
            f"HurstPairsStrategy(tw={self.tw_hours},hl={self.hedge_lookback_hours},"
            f"q={self.hurst_q},Hth={self.hurst_threshold},"
            f"band=({self.entry_lower_std},{self.entry_upper_std}),"
            f"stop={self.stop_std},hold={self.max_hold_hours},"
            f"k={self.top_k_pairs},clb={self.coint_lookback_hours},"
            f"alloc={self.pair_gross_alloc})"
        )

    def required_data(self):
        return {"prices": None}

    # ---------- Hurst exponent (q-order, local) ----------
    def _local_hurst(self, x):
        """H from regressing log K_q(tau) on log tau, with K_q(tau) prop to tau^(qH)."""
        x = np.asarray(x, dtype=float)
        if np.isnan(x).any():
            return np.nan
        q = self.hurst_q
        denom = np.mean(np.abs(x) ** q)
        if denom <= 0 or not np.isfinite(denom):
            return np.nan
        log_tau, log_k = [], []
        for tau in self._taus:
            if tau >= len(x):
                return np.nan
            incr = np.abs(x[tau:] - x[:-tau]) ** q
            k = incr.mean() / denom
            if k <= 0 or not np.isfinite(k):
                return np.nan
            log_tau.append(np.log(tau))
            log_k.append(np.log(k))
        slope, _ = np.polyfit(np.array(log_tau), np.array(log_k), 1)
        return slope / q

    # ---------- Per-bar signal computation (no lookahead, rolling b) ----------
    def _signal_at_t(self, prices_hist, a, b_sym):
        """Compute (s_t, m, std, H, b_t) at the current bar t (= last bar in prices_hist).

        b_t   uses returns of (a, b_sym) ending at t-1 over hedge_lookback_hours.
        Each historical spread s_τ in the m/std/H window uses its own contemporaneous b_τ.
        m, std, H are computed over s_{t-tw}..s_{t-1}.
        Returns dict or None if data insufficient / non-finite.
        """
        # Current bar must have both legs priced; else we can't form s_t (engine will
        # zero the asset weight too via its eligibility filter).
        if pd.isna(prices_hist[a].iloc[-1]) or pd.isna(prices_hist[b_sym].iloc[-1]):
            return None

        # Use the most recent contiguous non-NaN bars. Binance has ~0.1% maintenance
        # gaps; dropna and treat surviving rows as adjacent (vol/rolling stats are
        # then computed on non-NaN consecutive samples).
        n_needed = self.hedge_lookback_hours + self.tw_hours + 2
        sub_all = prices_hist[[a, b_sym]].dropna()
        if len(sub_all) < n_needed:
            return None
        sub = sub_all.iloc[-(n_needed + 200):]
        n = len(sub)

        log_a = np.log(sub[a].values)
        log_b = np.log(sub[b_sym].values)
        lr_a = np.diff(log_a)    # length n-1; lr_a[k] is the return at sub bar k+1
        lr_b = np.diff(log_b)

        # Rolling vol: pandas rolling(window).std() at index i uses lr[i-window+1..i]
        vol_a = pd.Series(lr_a).rolling(self.hedge_lookback_hours).std().values
        vol_b = pd.Series(lr_b).rolling(self.hedge_lookback_hours).std().values

        # b at sub bar j (j>=2) uses vol over returns ending at sub bar j-1
        # → window ends at lr-index j-2 → vol_a[j-2] / vol_b[j-2].
        b_at_sub = np.full(n, np.nan)
        if n >= 3:
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(vol_b[:-1] > 0, vol_a[:-1] / vol_b[:-1], np.nan)
            b_at_sub[2:] = ratio  # b_at_sub[j] = vol_a[j-2]/vol_b[j-2] for j in 2..n-1

        s = log_a - b_at_sub * log_b   # spread at each sub bar (NaN where b not yet defined)

        s_t = s[-1]
        s_window = s[-(self.tw_hours + 1):-1]   # length tw_hours, ending at t-1

        if (
            len(s_window) < self.tw_hours
            or np.isnan(s_window).any()
            or not np.isfinite(s_t)
        ):
            return None

        m_val = float(s_window.mean())
        std_val = float((s_window - m_val).std(ddof=1))
        h_val = self._local_hurst(s_window)

        if not (np.isfinite(std_val) and np.isfinite(h_val)) or std_val == 0.0:
            return None

        return {"s": float(s_t), "m": m_val, "std": std_val, "h": float(h_val), "b": float(b_at_sub[-1])}

    # ---------- Monthly cointegration ranking ----------
    def _select_pairs(self, prices_hist):
        """Top-K pairs by lowest ADF p-value on OLS residual, over the last coint_lookback_hours."""
        recent = prices_hist.iloc[-self.coint_lookback_hours:]
        cols = [c for c in recent.columns
                if recent[c].notna().sum() >= self.coint_lookback_hours * 0.9]
        scores = []
        for a, b in combinations(cols, 2):
            sub = recent[[a, b]].dropna()
            if len(sub) < self.coint_lookback_hours * 0.9:
                continue
            la = np.log(sub[a].values)
            lb = np.log(sub[b].values)
            try:
                _, p, _ = coint(la, lb, trend="c", autolag=None, maxlag=1)
            except Exception:
                continue
            if np.isfinite(p):
                scores.append(((a, b), p))
        scores.sort(key=lambda kv: kv[1])
        return [pair for pair, _ in scores[: self.top_k_pairs]]

    # ---------- Engine entry point ----------
    def generate_weights(self, data, t):
        self._event_this_bar = False     # reset; will flip to True iff open_trades mutates
        prices = data.prices
        # Index by panel-wide assets (data.prices.columns), NOT data.assets — the latter
        # excludes assets with NaN at t, but an open trade may involve a leg that's
        # temporarily NaN (Binance maintenance gap). Engine reindexes our output anyway
        # and zeros out ineligible names via its own filter (skills/02-strategy.md §3).
        weights = pd.Series(0.0, index=prices.columns)

        ts = pd.Timestamp(t)
        ym = (ts.year, ts.month)

        # Month change → re-rank pairs (using data through t, inclusive)
        if ym != self._current_month:
            self._current_month = ym
            if len(prices) >= self.coint_lookback_hours:
                self._selected_pairs = self._select_pairs(prices)

        if not self._selected_pairs and not self.open_trades:
            return weights

        k_leg = self.pair_gross_alloc / 2.0   # equal-dollar legs

        # ---- Exit pass: re-evaluate every currently open trade ----
        for pair in list(self.open_trades.keys()):
            a, b = pair
            sig = self._signal_at_t(prices, a, b)
            if sig is None:
                # Can't price the pair right now → hold position; will retry next bar
                trade = self.open_trades[pair]
                side = trade["side"]
                if side == -1:
                    weights[a] += +k_leg; weights[b] += -k_leg
                else:
                    weights[a] += -k_leg; weights[b] += +k_leg
                continue

            s_t = sig["s"]
            trade = self.open_trades[pair]
            m0, std0, side = trade["m0"], trade["std0"], trade["side"]
            hours_held = int((t - trade["entry_t"]) / pd.Timedelta("1h"))

            exit_now = False
            if side == +1 and s_t <= m0:                              # sold pair: spread reverted down
                exit_now = True
            elif side == -1 and s_t >= m0:                            # bought pair: spread reverted up
                exit_now = True
            elif side == +1 and s_t >= m0 + self.stop_std * std0:     # 2σ stop
                exit_now = True
            elif side == -1 and s_t <= m0 - self.stop_std * std0:
                exit_now = True
            elif hours_held >= self.max_hold_hours:                   # 72h expiration
                exit_now = True

            if exit_now:
                del self.open_trades[pair]
                self._event_this_bar = True
            else:
                if side == -1:
                    weights[a] += +k_leg; weights[b] += -k_leg
                else:
                    weights[a] += -k_leg; weights[b] += +k_leg

        # ---- Entry pass: scan selected (top-K) pairs not already open ----
        for pair in self._selected_pairs:
            if pair in self.open_trades:
                continue
            a, b = pair
            sig = self._signal_at_t(prices, a, b)
            if sig is None:
                continue
            if sig["h"] >= self.hurst_threshold:
                continue
            s_t, m_t, std_t = sig["s"], sig["m"], sig["std"]

            side = 0
            if (m_t + self.entry_lower_std * std_t) < s_t < (m_t + self.entry_upper_std * std_t):
                side = +1   # sell pair (short A, long B)
            elif (m_t - self.entry_upper_std * std_t) < s_t < (m_t - self.entry_lower_std * std_t):
                side = -1   # buy  pair (long A, short B)

            if side != 0:
                self.open_trades[pair] = {
                    "entry_t": t, "side": side, "m0": m_t, "std0": std_t,
                }
                self._event_this_bar = True
                if side == -1:
                    weights[a] += +k_leg; weights[b] += -k_leg
                else:
                    weights[a] += -k_leg; weights[b] += +k_leg

        return weights

    # ---------- Risk overlay: suppress micro-rebalance on quiet bars ----------
    def apply_risk(self, proposed, state, data):
        """If no entry/exit fired this bar, return current-weights so the engine's
        `target_$ - positions` step yields zero trades (kills mark-to-market drift trades).
        On event bars, return proposed weights normally. Always pass through the default
        RiskManager so per-asset caps / gross caps stay enforced.
        """
        equity = state["equity"]
        if not self._event_this_bar and equity > 0:
            proposed = state["positions"] / equity
        return super().apply_risk(proposed, state, data)


strat = HurstPairsStrategy()
print(strat)

rb = strat.rebalance_dates(panel.dates)
print(f"\nRebalance bars : {len(rb):,}  (every-hour -> should equal {len(panel.dates):,})")
print(f"First rebalance: {rb[0]}")
print(f"Last rebalance : {rb[-1]}")
print(f"tau for Hurst  : {strat._taus.tolist()}")
print(f"required_data(): {strat.required_data()}")

# Hurst sanity: stationary noise -> H ~ 0; random walk -> H ~ 0.5; trending -> H > 0.5
rng = np.random.default_rng(0)
noise = rng.standard_normal(168)
rw = np.cumsum(noise)
trend = np.cumsum(np.abs(noise))
print("\nHurst sanity (TW=168):")
print(f"  white noise (stationary, anti-persistent): H = {strat._local_hurst(noise):.3f}  (expect ~ 0)")
print(f"  random walk                              : H = {strat._local_hurst(rw):.3f}  (expect ~ 0.5)")
print(f"  trending series                          : H = {strat._local_hurst(trend):.3f}  (expect > 0.5)")

# _signal_at_t smoke test on a real pair from the panel
test_t = panel.dates[6000]   # ~10 months in, enough history for both windows
test_view = panel.as_of(test_t)
sig_btc_eth = strat._signal_at_t(test_view.prices, "BTCUSDT", "ETHUSDT")
print(f"\n_signal_at_t smoke test (BTC,ETH) at {test_t}:")
print(f"  {sig_btc_eth}")

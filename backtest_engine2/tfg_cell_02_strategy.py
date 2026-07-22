# Cell 2 — Stage 2: Forecast-to-Fill gold trend strategy (spec.json).
# Paper sizing in generate_weights; house cap 1% via RiskConfig at apply_risk.
# T+1: compute target at t, return yesterday's target. De-risk rule C (spec §6.3).
import numpy as np
import pandas as pd
from backtest.risk import RiskConfig
from backtest.strategy import Strategy

ASSET = "GC=F"
MAX_POSITION = 0.01  # user policy: no single name above 1% of equity

# spec.json parameters (frozen defaults; λ / Kelly stats set in fit on train only)
P = dict(
    K=50,
    beta=0.6,
    pbull_act=0.52,
    pbear_halve=0.50,
    pbear_close=0.55,
    z_clip=3.0,
    atr_n=14,
    hard_atr=2.0,
    trail_atr=1.5,
    max_age=30,
    vol_ann=0.15,
    w_max=2.0,
    kelly_frac=0.40,
    baseline_frac=0.25,
    cost_rt=0.7e-4,
    eta=0.02,
    n_rt=1,
    tdays=252,
    train_years=10,
    vol_lam=0.94,  # variance EWMA (RiskMetrics standard, spec vol_ewma_window_days=20)
)


def _ema(series: np.ndarray, lam: float) -> np.ndarray:
    out = np.empty_like(series, dtype=float)
    out[0] = series[0]
    for i in range(1, len(series)):
        out[i] = lam * out[i - 1] + (1.0 - lam) * series[i]
    return out


def _true_range(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    tr = np.empty(len(c), dtype=float)
    tr[0] = h[0] - l[0]
    prev = c[:-1]
    tr[1:] = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - prev), np.abs(l[1:] - prev)))
    return tr


def _atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, n: int) -> np.ndarray:
    tr = _true_range(h, l, c)
    out = np.full(len(c), np.nan)
    for i in range(n - 1, len(c)):
        out[i] = tr[i - n + 1 : i + 1].mean()
    return out


def _ewma_var(r: np.ndarray, lam: float, sigma2_0: float) -> np.ndarray:
    out = np.empty(len(r), dtype=float)
    out[0] = sigma2_0
    for i in range(1, len(r)):
        out[i] = lam * out[i - 1] + (1.0 - lam) * r[i - 1] ** 2
    return out


def _kelly_f(mu: float, sigma2: float, k: float, eta: float, n: int, kappa: float) -> float:
    if sigma2 <= 0 or mu <= n * k:
        return 0.0
    disc = 9.0 * eta**2 * n**3 + 16.0 * sigma2 * (mu - n * k)
    if disc < 0:
        return 0.0
    x = (-3.0 * eta * n**1.5 + np.sqrt(disc)) / (4.0 * sigma2)
    return kappa * (x**2)


def _signals(px: np.ndarray, hi: np.ndarray, lo: np.ndarray, lam: float, mu_tr: float,
             sig_tr: float, sigma2_0: float) -> dict:
    """Forward-only signals on arrays ending at t (inclusive). All PIT."""
    y = np.log(np.maximum(px, 1e-12))
    y_t = _ema(y, lam)
    dy = np.diff(y_t, prepend=y_t[0])
    z = (dy - mu_tr) / sig_tr if sig_tr > 0 else np.zeros_like(dy)
    z_bar = np.clip(z, -P["z_clip"], P["z_clip"])
    p_trend = (z_bar + P["z_clip"]) / (2.0 * P["z_clip"])
    m = np.zeros(len(px), dtype=float)
    if len(px) > P["K"]:
        m[P["K"] :] = (px[P["K"] :] / px[: -P["K"]] > 1.0).astype(float)
    p_bull = P["beta"] * p_trend + (1.0 - P["beta"]) * m
    p_bear = 1.0 - p_bull
    r = np.zeros(len(px))
    r[1:] = px[1:] / px[:-1] - 1.0
    sig2 = _ewma_var(r, P["vol_lam"], sigma2_0)  # spec: fixed RiskMetrics λ, not trend λ
    sig_next = np.sqrt(np.maximum(sig2, 1e-16))
    sig_star = P["vol_ann"] / np.sqrt(P["tdays"])
    w_vol = np.minimum(P["w_max"], sig_star / sig_next)
    conf = np.clip((p_bull - 0.5) / 0.5, 0.0, 1.0)
    w_conf = w_vol * conf
    atr = _atr(hi, lo, px, P["atr_n"])
    return dict(
        dy=dy, p_bull=p_bull, p_bear=p_bear, w_conf=w_conf, w_vol=w_vol, atr=atr, r=r,
    )


def _unit_returns(px: np.ndarray, hi: np.ndarray, lo: np.ndarray, lam: float,
                  mu_tr: float, sig_tr: float, sigma2_0: float) -> np.ndarray:
    """Train-only unit-notional return proxy for Kelly μ, σ (weight=1 on activation else 0)."""
    sig = _signals(px, hi, lo, lam, mu_tr, sig_tr, sigma2_0)
    act = (sig["p_bull"] >= P["pbull_act"]) & (sig["dy"] > 0)
    return np.where(act, 1.0, 0.0) * sig["r"]


class ForecastToFillGoldStrategy(Strategy):
    """Forecast-to-Fill gold trend+momentum (Singha et al.); FAST config."""
    rebalance_frequency = "daily"

    def __init__(
        self,
        max_position: float = MAX_POSITION,
        max_gross: float = 1.0,
        max_net: float = 1.0,
        vol_only: bool = False,
    ):
        self._vol_only = vol_only
        self.risk = RiskConfig(
            max_position=max_position,
            max_gross=max_gross,
            max_net=max_net,
            target_vol=None,
        )
        self._lam = 0.94
        self._mu_tr = self._sig_tr = 0.0
        self._mu_k = self._sigma2_k = 0.0
        self._sigma2_0 = 1e-6
        self._fitted = False
        # T+1 execution lag
        self._pending: pd.Series | None = None
        # open-trade state (post-fill position management)
        self._in_trade = False
        self._entry_px = self._peak_px = 0.0
        self._trade_age = 0

    def __repr__(self) -> str:
        mode = "vol_only" if self._vol_only else "full_kelly"
        return (
            f"ForecastToFillGoldStrategy({mode}, K={P['K']}, beta={P['beta']}, "
            f"max_position={self.risk.max_position}, w_max_paper={P['w_max']}, T+1=True)"
        )

    def required_data(self) -> dict:
        return {"prices": None, "volume": None, "open": None, "high": None, "low": None}

    def fit(self, data) -> None:
        # spec: μ_train/σ_train, λ and Kelly stats from the PRECEDING 10-YEAR window only.
        # Engine's train view is end-clipped, not start-clipped — trim here.
        px_s = data.prices[ASSET].dropna()
        cutoff = px_s.index[-1] - pd.DateOffset(years=P["train_years"])
        px_s = px_s[px_s.index >= cutoff]
        px = px_s.to_numpy()
        hi = data.feature("high")[ASSET].reindex(data.prices.index).ffill().reindex(px_s.index).to_numpy()
        lo = data.feature("low")[ASSET].reindex(data.prices.index).ffill().reindex(px_s.index).to_numpy()
        r = np.zeros(len(px))
        r[1:] = px[1:] / px[:-1] - 1.0
        self._sigma2_0 = float(np.var(r[1:])) if len(r) > 2 else 1e-8

        best_lam, best_sr = 0.94, -np.inf
        for lam in np.arange(0.88, 0.995, 0.01):
            y = np.log(np.maximum(px, 1e-12))
            y_t = _ema(y, lam)
            dy = np.diff(y_t, prepend=y_t[0])
            mu, sig = float(np.mean(dy)), float(np.std(dy, ddof=1))
            if sig <= 0:
                continue
            sig_d = _signals(px, hi, lo, lam, mu, sig, self._sigma2_0)
            act = (sig_d["p_bull"] >= P["pbull_act"]) & (sig_d["dy"] > 0)
            ut = np.where(act, sig_d["r"], 0.0)
            if np.std(ut) < 1e-12:
                continue
            sr = np.mean(ut) / np.std(ut) * np.sqrt(P["tdays"])
            if sr > best_sr:
                best_sr, best_lam = sr, float(lam)

        self._lam = best_lam
        y = np.log(np.maximum(px, 1e-12))
        y_t = _ema(y, self._lam)
        dy = np.diff(y_t, prepend=y_t[0])
        self._mu_tr = float(np.mean(dy))
        self._sig_tr = float(np.std(dy, ddof=1)) or 1e-8

        ut = _unit_returns(px, hi, lo, self._lam, self._mu_tr, self._sig_tr, self._sigma2_0)
        self._mu_k = float(np.mean(ut))
        self._sigma2_k = max(float(np.var(ut, ddof=1)), 1e-12) if len(ut) > 1 else 1e-12

        self._fitted = True
        self._pending = None
        self._in_trade = False
        self._trade_age = 0

    def _raw_target(self, data, t: pd.Timestamp) -> float:
        if not self._fitted:
            return 0.0
        idx = data.prices.index
        if t not in idx:
            return 0.0
        i = idx.get_loc(t)
        if i < max(P["K"], P["atr_n"], 20):
            return 0.0

        px = data.prices[ASSET].iloc[: i + 1].to_numpy()
        hi = data.feature("high")[ASSET].iloc[: i + 1].to_numpy()
        lo = data.feature("low")[ASSET].iloc[: i + 1].to_numpy()
        sig = _signals(px, hi, lo, self._lam, self._mu_tr, self._sig_tr, self._sigma2_0)
        j = -1
        c, atr = px[j], sig["atr"][j]
        p_bull, p_bear, dy = sig["p_bull"][j], sig["p_bear"][j], sig["dy"][j]
        w_conf = sig["w_conf"][j]
        w_vol = sig["w_vol"][j]

        f = _kelly_f(self._mu_k, self._sigma2_k, P["cost_rt"], P["eta"], P["n_rt"], P["kelly_frac"])
        if self._vol_only:
            target = min(P["w_max"], w_conf)
        else:
            if f < 1e-8:
                f = P["baseline_frac"] * w_vol
            target = min(P["w_max"], f * w_conf)

        # exit / de-risk rule C on open trade
        if self._in_trade:
            self._trade_age += 1
            self._peak_px = max(self._peak_px, c)
            exit_now = False
            if np.isfinite(atr):
                if c < self._entry_px - P["hard_atr"] * atr:
                    exit_now = True
                elif c < self._peak_px - P["trail_atr"] * atr:
                    exit_now = True
            if self._trade_age >= P["max_age"]:
                exit_now = True
            if p_bear > P["pbear_close"]:
                exit_now = True
                target = 0.0
            elif p_bear > P["pbear_halve"]:
                target *= 0.5
            if exit_now:
                self._in_trade = False
                self._trade_age = 0
                target = 0.0
        else:
            if (p_bull >= P["pbull_act"]) and (dy > 0) and target > 0:
                self._in_trade = True
                self._entry_px = self._peak_px = c
                self._trade_age = 0
            else:
                target = 0.0

        return float(max(0.0, target))

    def generate_weights(self, data, t: pd.Timestamp) -> pd.Series:
        raw = self._raw_target(data, t)
        pending = pd.Series({ASSET: raw})
        if self._pending is None:
            out = pd.Series(0.0, index=data.assets)
        else:
            out = self._pending.reindex(data.assets).fillna(0.0)
        self._pending = pending
        return out


strat = ForecastToFillGoldStrategy()
train_end = panel.dates[panel.dates <= "2014-12-31"][-1]
strat.fit(panel.as_of(train_end))
print(strat)
print(f"fit through {train_end.date()}: lam={strat._lam:.3f} mu_tr={strat._mu_tr:.2e} "
      f"sig_tr={strat._sig_tr:.2e} mu_k={strat._mu_k:.2e} sigma2_k={strat._sigma2_k:.2e}")
print(f"RiskConfig: max_position={strat.risk.max_position} | "
      f"max_gross={strat.risk.max_gross} max_net={strat.risk.max_net}")

oos = panel.dates[(panel.dates >= "2015-01-01") & (panel.dates < "2015-02-01")]
rows = []
for t in oos:
    view = panel.as_of(t)
    proposed = strat.generate_weights(view, t)
    state = {"drawdown": 0.0, "equity": 1e6, "positions": pd.Series(0.0, index=panel.assets_all),
             "cash": 1e6, "peak": 1e6}
    final = strat.apply_risk(proposed, state, view)
    rows.append((t.date(), float(proposed.get(ASSET, 0)), float(final.get(ASSET, 0)),
                 strat._in_trade, float(strat._pending.get(ASSET, 0) if strat._pending is not None else 0)))

diag = pd.DataFrame(rows, columns=["date", "exec_w", "after_risk", "in_trade", "next_pending"])
print("\nJan-2015 T+1 walk (exec_w = weight applied at close; next_pending = computed today for tomorrow):")
print(diag.to_string(index=False))
print(f"\nPaper raw pending max in window: {diag.next_pending.max():.4f} | "
      f"after 1% cap max: {diag.after_risk.max():.4f}")

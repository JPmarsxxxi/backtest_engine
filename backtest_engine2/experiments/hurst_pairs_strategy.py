"""HurstPairsStrategy — spec-flow implementation of the Hurst pairs alpha.

Refined version per hunt 2026-06-02 EDA findings:
- Monthly PIT pair reselection by joint score (coint stability × Hurst-firing frac), K=3 per-asset cap.
- Refined entry gate: H<=0.42, time_in_band<=6, |z| in (1.0, 1.33].
- Per-trade O-U calibration on entry: phi/sigma fit on prior 168h, simulate 1k paths,
  grid-search 20x20 (pi+*, pi-*) for max Sharpe.
- Exit: first of TP / SL / 72h barrier.
- Force-close on de-listed pairs at month boundary.

The strategy consumes pre-cached PIT series (coint_series_df, hurst_series_df) so it
doesn't recompute heavy rolling stats per generate_weights call.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.strategy import Strategy


PI_PLUS_GRID = np.arange(0.5, 10.5, 0.5)
PI_MINUS_GRID = -np.arange(0.5, 10.5, 0.5)
N_STEPS = 72  # vertical barrier


class HurstPairsStrategy(Strategy):
    """Pair mean-reversion using Hurst gate + O-U exits, monthly PIT pair selection."""

    rebalance_frequency = "h"  # hourly: check every bar for entries/exits

    def __init__(
        self,
        prices_pit: pd.DataFrame,
        coint_series_df: pd.DataFrame,
        hurst_series_df: pd.DataFrame,
        gate_params: dict | None = None,
        hedge_window: int = 720,
        hurst_window: int = 168,
        z_window: int = 168,
        hold_max: int = N_STEPS,
        n_ou_paths: int = 1000,
        per_asset_cap: int = 3,
        top_n: int = 30,
        min_coint_anchors: int = 12,
        ou_seed: int = 42,
        verbose: bool = False,
        assert_pit: bool = True,
        per_leg_size: float = 0.01,
        vol_target: bool = True,
        reference_sigma: float = 0.008,
        max_per_leg_size: float = 0.03,
    ):
        self.gate_params = gate_params or {
            "hurst_thr": 0.42, "tib_thr": 6, "z_lo": 1.0, "z_hi": 1.33,
        }
        self.hedge_window = hedge_window
        self.hurst_window = hurst_window
        self.z_window = z_window
        self.hold_max = hold_max
        self.n_ou_paths = n_ou_paths
        self.per_asset_cap = per_asset_cap
        self.top_n = top_n
        self.min_coint_anchors = min_coint_anchors
        self.ou_seed = ou_seed
        self.verbose = verbose
        self.assert_pit = assert_pit
        self.per_leg_size = per_leg_size           # 1% baseline (was 5%)
        self.vol_target = vol_target               # True → scale by reference_sigma / sigma_cal at entry
        self.reference_sigma = reference_sigma     # 0.008 = median observed OU sigma
        self.max_per_leg_size = max_per_leg_size   # cap to prevent runaway on low-vol pairs

        # Counters for verbose progress reporting (reset in fit / __init__).
        self._n_entries_total = 0
        self._n_exits_tp = 0
        self._n_exits_sl = 0
        self._n_exits_barrier = 0
        self._n_exits_delisted = 0
        # Monthly bucket counters reset every month boundary.
        self._month_entries = 0
        self._month_exits_tp = 0
        self._month_exits_sl = 0
        self._month_exits_barrier = 0
        self._month_exits_delisted = 0

        # Pre-process cached data into per-pair lookup-friendly numpy arrays.
        self._prices_pit = prices_pit
        self._log_prices_pit = np.log(prices_pit)

        # Cointegration anchors per pair (PIT - each anchor uses only prior 12-month window).
        self.coint_data: dict[str, dict] = {}
        for pair_id, group in coint_series_df.groupby("pair_id"):
            group = group.sort_values("anchor_t")
            self.coint_data[pair_id] = {
                "asset_a": group["asset_a"].iloc[0],
                "asset_b": group["asset_b"].iloc[0],
                "anchor_t_arr": np.asarray(group["anchor_t"].values, dtype="datetime64[ns]"),
                "p_value_arr": group["p_value"].values.astype(float),
                "hedge_ratio_arr": group["hedge_ratio"].values.astype(float),
            }

        # Hurst + spread per pair (PIT - each bar uses only prior 168h).
        self.hurst_data: dict[str, dict] = {}
        for pair_id, group in hurst_series_df.groupby("pair_id"):
            group = group.sort_values("bar_t")
            self.hurst_data[pair_id] = {
                "bar_t_arr": np.asarray(pd.DatetimeIndex(group["bar_t"]).values, dtype="datetime64[ns]"),
                "hurst_arr": group["hurst"].values.astype(float),
                "spread_arr": group["spread"].values.astype(float),
            }

        # Lazy-derived per-pair: z, abs_z, sign, tib, m, hedge_b — built on first use.
        self._derived: dict[str, dict] = {}

        # State (reset per CPCV fold via deepcopy).
        self.active_trades: dict[tuple[str, str], dict] = {}
        self.current_universe: list[tuple[str, str]] = []
        self.last_selection_month: tuple[int, int] | None = None

    # ----- Strategy interface -----

    def __repr__(self) -> str:
        return (
            f"HurstPairsStrategy(top_n={self.top_n}, cap={self.per_asset_cap}, "
            f"gate={self.gate_params}, ou_paths={self.n_ou_paths}, "
            f"per_leg={self.per_leg_size}, vol_target={self.vol_target}, "
            f"ref_sigma={self.reference_sigma}, max_leg={self.max_per_leg_size})"
        )

    def required_data(self) -> dict:
        return {"prices": None}

    def fit(self, data) -> None:
        T = data.t
        self.current_universe = self._select_pairs_pit(T)
        self.last_selection_month = (T.year, T.month)
        self.active_trades = {}
        self._n_entries_total = 0
        self._n_exits_tp = 0
        self._n_exits_sl = 0
        self._n_exits_barrier = 0
        self._n_exits_delisted = 0
        if self.verbose:
            print(f"[fit @ {T}] initial universe: {len(self.current_universe)} pairs")

    def generate_weights(self, data, t) -> pd.Series:
        # Month boundary: re-select universe, force-close de-listed pairs.
        current_month = (t.year, t.month)
        if self.last_selection_month != current_month:
            # Print monthly progress BEFORE reselecting (summary of the month just ending).
            if self.verbose and self.last_selection_month is not None:
                n_active_before_reselect = len(self.active_trades)
                print(
                    f"[{t}] month boundary | universe={len(self.current_universe)} "
                    f"active_in={n_active_before_reselect} | "
                    f"this_month: opened={self._month_entries} tp={self._month_exits_tp} "
                    f"sl={self._month_exits_sl} barrier={self._month_exits_barrier} | "
                    f"cumulative_entries={self._n_entries_total}"
                )
            self._month_entries = 0
            self._month_exits_tp = 0
            self._month_exits_sl = 0
            self._month_exits_barrier = 0
            self._month_exits_delisted = 0

            new_universe = self._select_pairs_pit(t)
            new_set = set(new_universe)
            for pair in list(self.active_trades.keys()):
                if pair not in new_set:
                    del self.active_trades[pair]
                    self._n_exits_delisted += 1
                    self._month_exits_delisted += 1
            self.current_universe = new_universe
            self.last_selection_month = current_month
            if self.verbose:
                print(
                    f"[{t}] new universe selected: {len(new_universe)} pairs, "
                    f"force-closed {self._month_exits_delisted} de-listed trades; "
                    f"carrying {len(self.active_trades)} active trades into new month"
                )

        weights = pd.Series(0.0, index=data.assets)
        T_np = np.datetime64(t)

        for pair in self.current_universe:
            a, b = pair
            pair_id = f"{a}|{b}"
            derived = self._get_derived(pair_id)

            idx = int(np.searchsorted(derived["bar_t_arr"], T_np))
            if idx >= len(derived["bar_t_arr"]) or derived["bar_t_arr"][idx] != T_np:
                continue

            spread_t = derived["spread_arr"][idx]
            hurst_t = derived["hurst_arr"][idx]
            abs_z_t = derived["abs_z_arr"][idx]
            sign_t = derived["sign_arr"][idx]
            tib_t = derived["tib_arr"][idx]
            hedge_b_t = derived["hedge_b_arr"][idx]

            if pair in self.active_trades:
                trade = self.active_trades[pair]
                if not np.isfinite(spread_t):
                    del self.active_trades[pair]
                    continue
                cur_pnl_sigma = -trade["sign"] * (spread_t - trade["entry_spread"]) / trade["sigma_cal"]
                bars_held = idx - trade["entry_idx"]

                if cur_pnl_sigma >= trade["pi_plus_star"]:
                    del self.active_trades[pair]  # TP
                    self._n_exits_tp += 1
                    self._month_exits_tp += 1
                elif cur_pnl_sigma <= trade["pi_minus_star"]:
                    del self.active_trades[pair]  # SL
                    self._n_exits_sl += 1
                    self._month_exits_sl += 1
                elif bars_held >= self.hold_max:
                    del self.active_trades[pair]  # barrier
                    self._n_exits_barrier += 1
                    self._month_exits_barrier += 1
                else:
                    # Use FIXED per-leg weight stored at entry (possibly vol-scaled).
                    leg_size = trade["leg_size"]
                    if a in weights.index:
                        weights[a] += trade["sign"] * leg_size
                    if b in weights.index:
                        weights[b] += -trade["sign"] * trade["hedge_b"] * leg_size
            else:
                if not (
                    np.isfinite(hurst_t) and np.isfinite(abs_z_t)
                    and np.isfinite(tib_t) and np.isfinite(hedge_b_t)
                ):
                    continue
                if not (
                    hurst_t < self.gate_params["hurst_thr"]
                    and tib_t <= self.gate_params["tib_thr"]
                    and self.gate_params["z_lo"] < abs_z_t < self.gate_params["z_hi"]
                ):
                    continue
                cal = self._calibrate_ou_at(pair_id, idx)
                if cal is None:
                    continue
                trade_sign = float(-sign_t)
                # Vol-targeted leg sizing: scale baseline by reference_sigma / sigma_cal,
                # clipped to max_per_leg_size. Sets risk-per-σ-move ~constant across pairs.
                if self.vol_target and cal["sigma_cal"] > 0:
                    scaled = self.per_leg_size * (self.reference_sigma / cal["sigma_cal"])
                    leg_size = min(scaled, self.max_per_leg_size)
                else:
                    leg_size = self.per_leg_size
                self.active_trades[pair] = {
                    "entry_t": t,
                    "entry_idx": idx,
                    "entry_spread": cal["entry_spread"],
                    "sign": trade_sign,
                    "pi_plus_star": cal["pi_plus_star"],
                    "pi_minus_star": cal["pi_minus_star"],
                    "sigma_cal": cal["sigma_cal"],
                    "hedge_b": float(hedge_b_t),
                    "leg_size": float(leg_size),
                }
                self._n_entries_total += 1
                self._month_entries += 1
                if a in weights.index:
                    weights[a] += trade_sign * leg_size
                if b in weights.index:
                    weights[b] += -trade_sign * float(hedge_b_t) * leg_size

        # NO end-of-bar gross-normalize. Each active trade contributes a fixed
        # per-leg weight set at entry; new/closed trades do not resize others.
        return weights

    # ----- Internals -----

    def _get_derived(self, pair_id: str) -> dict:
        """Lazy-compute and cache: z, abs_z, sign, tib, m, hedge_b (per pair)."""
        if pair_id in self._derived:
            return self._derived[pair_id]

        coint = self.coint_data[pair_id]
        a, b = coint["asset_a"], coint["asset_b"]
        hurst = self.hurst_data[pair_id]

        bar_idx = pd.DatetimeIndex(hurst["bar_t_arr"])
        spread = pd.Series(hurst["spread_arr"], index=bar_idx)

        # Rolling stats with PIT shift.
        m = spread.rolling(self.z_window, min_periods=self.z_window).mean().shift(1)
        sd = spread.rolling(self.z_window, min_periods=self.z_window).std().shift(1)
        z = (spread - m) / sd
        abs_z = z.abs()
        sign_z = np.sign(z.fillna(0))

        # Time-in-band counter (consecutive bars with |z|>1).
        in_band = (abs_z > 1).astype(int).fillna(0).astype(int).values
        tib = np.zeros(len(in_band), dtype=float)
        cur = 0
        for i, ib in enumerate(in_band):
            if ib:
                cur += 1
            else:
                cur = 0
            tib[i] = float(cur) if ib else 0.0

        # Vol-scaled rolling hedge ratio b = std(lr_A) / std(lr_B), aligned to spread index.
        lr_a = self._log_prices_pit[a].reindex(bar_idx).diff()
        lr_b = self._log_prices_pit[b].reindex(bar_idx).diff()
        std_a = lr_a.rolling(self.hedge_window, min_periods=self.hedge_window).std()
        std_b = lr_b.rolling(self.hedge_window, min_periods=self.hedge_window).std()
        hedge_b = (std_a / std_b).values

        derived = {
            "asset_a": a, "asset_b": b,
            "bar_t_arr": np.asarray(bar_idx.values, dtype="datetime64[ns]"),
            "spread_arr": spread.values,
            "hurst_arr": hurst["hurst_arr"],
            "z_arr": z.values,
            "abs_z_arr": abs_z.values,
            "sign_arr": sign_z.values,
            "tib_arr": tib,
            "m_arr": m.values,
            "sd_arr": sd.values,
            "hedge_b_arr": hedge_b,
        }
        self._derived[pair_id] = derived
        return derived

    def _select_pairs_pit(self, T) -> list[tuple[str, str]]:
        """PIT pair selection at time T: rank by joint score, apply K-cap, return top-N."""
        T_np = np.datetime64(T)
        scores = []
        for pair_id, coint in self.coint_data.items():
            mask_c = coint["anchor_t_arr"] <= T_np
            n_c = int(mask_c.sum())
            if n_c < self.min_coint_anchors:
                continue
            # PIT assertion: every anchor used must be <= T.
            if self.assert_pit:
                assert coint["anchor_t_arr"][mask_c].max() <= T_np, \
                    f"PIT violation in coint selection at T={T}: anchor > T"
            stability = float((coint["p_value_arr"][mask_c] < 0.05).mean())

            hurst = self.hurst_data[pair_id]
            mask_h = hurst["bar_t_arr"] <= T_np
            n_h = int(mask_h.sum())
            if n_h < self.hurst_window:
                continue
            if self.assert_pit:
                assert hurst["bar_t_arr"][mask_h].max() <= T_np, \
                    f"PIT violation in hurst selection at T={T}: bar > T"
            h_frac = float((hurst["hurst_arr"][mask_h] < 0.5).mean())

            scores.append({
                "pair_id": pair_id,
                "asset_a": coint["asset_a"],
                "asset_b": coint["asset_b"],
                "joint": stability * h_frac,
            })
        if not scores:
            return []

        df = (
            pd.DataFrame(scores)
            .sort_values("joint", ascending=False)
            .reset_index(drop=True)
        )
        counts: dict[str, int] = {}
        picked: list[tuple[str, str]] = []
        for _, row in df.iterrows():
            if len(picked) >= self.top_n:
                break
            a, b = row["asset_a"], row["asset_b"]
            if counts.get(a, 0) >= self.per_asset_cap or counts.get(b, 0) >= self.per_asset_cap:
                continue
            picked.append((a, b))
            counts[a] = counts.get(a, 0) + 1
            counts[b] = counts.get(b, 0) + 1
        return picked

    def _calibrate_ou_at(self, pair_id: str, idx: int) -> dict | None:
        """Per-trade O-U calibration at bar idx: fit phi/sigma on prior 168h, simulate, grid-search."""
        derived = self._get_derived(pair_id)
        spread_arr = derived["spread_arr"]
        sign_arr = derived["sign_arr"]
        bar_t_arr = derived["bar_t_arr"]

        start = idx - self.hurst_window
        if start < 0:
            return None
        window = spread_arr[start:idx]
        if len(window) < self.hurst_window or np.isnan(window).any():
            return None
        # PIT assertion: the fit window must not include the current bar (idx).
        if self.assert_pit:
            assert bar_t_arr[idx - 1] < bar_t_arr[idx], \
                f"PIT violation in OU calibration at idx={idx}: fit window includes current bar"

        m = float(np.mean(window))
        Y = np.diff(window)
        X = window[:-1] - m
        var_X = float(np.var(X))
        if var_X <= 1e-12:
            return None
        slope = float(np.mean(Y * X)) / var_X
        phi = 1.0 + slope
        if not (-1.0 < phi < 1.0):
            return None
        resid = Y - slope * X
        sigma_sq = float(np.var(resid))
        if sigma_sq <= 0:
            return None
        sigma = float(np.sqrt(sigma_sq))

        s_0 = float(spread_arr[idx])
        sign_t = sign_arr[idx]
        if not np.isfinite(sign_t) or sign_t == 0:
            return None
        sign_dir = -sign_t

        rng = np.random.default_rng(self.ou_seed)
        n_paths = self.n_ou_paths
        n_steps = self.hold_max
        paths = np.empty((n_paths, n_steps + 1))
        paths[:, 0] = s_0
        eps = rng.standard_normal((n_paths, n_steps))
        one_minus_phi_m = (1.0 - phi) * m
        for k in range(1, n_steps + 1):
            paths[:, k] = phi * paths[:, k - 1] + one_minus_phi_m + sigma * eps[:, k - 1]

        pnl_sigma = (sign_dir * (paths - s_0)) / sigma
        pnl_after = pnl_sigma[:, 1:]
        cummax = np.maximum.accumulate(pnl_after, axis=1)
        cummin = np.minimum.accumulate(pnl_after, axis=1)

        cross_pp = np.full((n_paths, len(PI_PLUS_GRID)), n_steps - 1, dtype=np.int32)
        for j, pp in enumerate(PI_PLUS_GRID):
            mask = cummax >= pp
            any_hit = mask.any(axis=1)
            cross_pp[any_hit, j] = mask[any_hit].argmax(axis=1)
        cross_pm = np.full((n_paths, len(PI_MINUS_GRID)), n_steps - 1, dtype=np.int32)
        for j, pm in enumerate(PI_MINUS_GRID):
            mask = cummin <= pm
            any_hit = mask.any(axis=1)
            cross_pm[any_hit, j] = mask[any_hit].argmax(axis=1)

        best_sharpe = -np.inf
        best_pp = best_pm = None
        for i, pp in enumerate(PI_PLUS_GRID):
            for j, pm in enumerate(PI_MINUS_GRID):
                exit_step = np.minimum(cross_pp[:, i], cross_pm[:, j])
                pnl_at_exit = pnl_after[np.arange(n_paths), exit_step]
                mu = float(pnl_at_exit.mean())
                sd_ = float(pnl_at_exit.std())
                if sd_ <= 1e-9:
                    continue
                sh = mu / sd_
                if sh > best_sharpe:
                    best_sharpe = sh
                    best_pp = float(pp)
                    best_pm = float(pm)
        if best_pp is None:
            return None
        return {
            "pi_plus_star": best_pp,
            "pi_minus_star": best_pm,
            "sigma_cal": sigma,
            "entry_spread": s_0,
            "phi": phi,
        }

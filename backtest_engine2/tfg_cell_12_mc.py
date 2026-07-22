# Cell 12 — Stage 8: Monte Carlo stress test (skills/10-simulation.md §15.1).
# Generators fit on real OOS GC returns (2015 -> 2025-10): gaussian (drift baseline),
# block_bootstrap (practical default), permutation (lookahead-leak null — mandatory).
# Surfaced mismatches: synthetic paths are close-only (no OHLC/volume), so this MC
# variant feeds hi=lo=close -> TR collapses to |dC| (Wilder close-only fallback,
# tighter stops than true ATR); costs = Commission only (MarketImpact needs volume;
# impact was ~0.01bp on real data). Strategy pre-fit on real 2005-2014 (MC engine
# does not call fit; deep copies carry frozen train stats). vol_only @ paper caps.
from backtest.simulation import (
    MonteCarloEngine, GaussianGenerator, BlockBootstrapGenerator,
    PermutationGenerator, GeneratorValidator,
)


class ForecastToFillGoldMC(ForecastToFillGoldStrategy):
    """Close-only MC variant: prices-only requirement; hi=lo=close so TR=|dC|."""

    def required_data(self) -> dict:
        return {"prices": None}

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
        sig = _signals(px, px, px, self._lam, self._mu_tr, self._sig_tr, self._sigma2_0)
        j = -1
        c, atr = px[j], sig["atr"][j]
        p_bull, p_bear, dy = sig["p_bull"][j], sig["p_bear"][j], sig["dy"][j]
        w_conf, w_vol = sig["w_conf"][j], sig["w_vol"][j]
        f = _kelly_f(self._mu_k, self._sigma2_k, P["cost_rt"], P["eta"], P["n_rt"], P["kelly_frac"])
        if self._vol_only:
            target = min(P["w_max"], w_conf)
        else:
            if f < 1e-8:
                f = P["baseline_frac"] * w_vol
            target = min(P["w_max"], f * w_conf)
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


# 1. Real OOS returns the generators must mimic (single asset, shape (T, 1)).
r_real = prices[ASSET].loc[eval_dates].pct_change().dropna().to_numpy().reshape(-1, 1)
gens = {
    "gaussian": GaussianGenerator.fit(r_real),
    "block_bootstrap": BlockBootstrapGenerator.fit(r_real),
    "permutation": PermutationGenerator(r_real),
}
print(f"Block length (fit): {gens['block_bootstrap'].block_length}")

# 2. Validate against the Cont-2001 stylized-fact gate.
validator = GeneratorValidator(r_real, n_paths=20, seed=0)
for name, gen in gens.items():
    vr = validator.validate(gen)
    failed = [m for m, p in vr.passed.items() if not p]
    print(f"  {name:16s} overall_passed={vr.overall_passed}  failed={failed}")

# 3. Pre-fit prototype on real data (MC workers deep-copy; fit is NOT re-called).
proto = ForecastToFillGoldMC(max_position=P["w_max"], max_gross=P["w_max"],
                             max_net=P["w_max"], vol_only=True)
proto.fit(panel.as_of(pd.Timestamp("2014-12-31")))
print(f"Prototype fit: lam={proto._lam:.3f} (frozen real-data train stats)")

# 4. Run: 200 paths x 3 generators, costs on (commission leg).
mc = MonteCarloEngine(
    strategy=proto,
    generators=gens,
    assets=list(panel.assets_all),
    dates=eval_dates[1:],  # permutation caps n_steps at T=len(r_real)
    costs=Commission(bps=LEG_BPS),
    initial_capital=1_000_000,
    warmup_bars=60,
).run(n_paths=200, seed=0, n_jobs=-1)

tbl = mc.summary()

perm_sharpes = [r.sharpe for r in mc.metrics_per_path()["permutation"] if np.isfinite(r.sharpe)]
bb_q05 = tbl.loc["block_bootstrap", "sharpe_q05"]
print(f"\nLeak check — permutation-null median Sharpe: {np.median(perm_sharpes):+.3f} "
      f"(|x| >> 0 would mean lookahead)")
print(f"Robustness — block_bootstrap sharpe_q05: {bb_q05:+.3f}")

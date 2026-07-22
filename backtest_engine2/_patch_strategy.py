class CryptoRegimeOuTrendStrategy(OuTrendPullbackStrategy):
    """
    OuTrendPullbackStrategy gated by a BTC DVOL expanding-percentile macro regime filter.
    DVOL > expanding P(dvol_percentile): block new entries only.
    Existing trades run to their natural OU exit (pi_plus / pi_minus / timeout).
    """
    rebalance_frequency = "D"
    risk = RiskConfig(max_position=None, max_gross=1.0, max_net=1.0)

    def __init__(self, dvol_percentile=DVOL_PERCENTILE, dvol_warmup=DVOL_WARMUP_BARS, **kwargs):
        super().__init__(**kwargs)
        self.dvol_percentile = dvol_percentile
        self.dvol_warmup     = dvol_warmup
        self._dvol_hist: list = []

    def __repr__(self):
        return (f"CryptoRegimeOuTrendStrategy("
                f"dvol_pct={self.dvol_percentile}, dvol_warmup={self.dvol_warmup}, "
                f"fast={self.fast_mode}, paths={self.num_simulated_paths}, "
                f"hold={self.max_holding_period}, ols={self.ou_ols_window}, "
                f"maxpos={self.max_positions}, w={self.weight_per_position})")

    def required_data(self):
        return {**super().required_data(), "DVOL": None}

    def generate_weights(self, data, t):
        self._event_this_bar = False

        # ── Macro regime gate (Layer 1) ───────────────────────────────────────
        dvol_row     = data.feature("DVOL").iloc[-1]
        current_dvol = float(dvol_row.iloc[0]) if len(dvol_row) > 0 else np.nan

        if np.isfinite(current_dvol):
            self._dvol_hist.append(current_dvol)

        if len(self._dvol_hist) >= self.dvol_warmup and np.isfinite(current_dvol):
            expanding_threshold = float(np.percentile(self._dvol_hist, self.dvol_percentile))
            regime_off = current_dvol > expanding_threshold
        else:
            regime_off = False   # still in warmup — allow trading

        if regime_off:
            # Block new entries; existing trades run to their natural OU exit
            _saved = self.max_new_entries_per_bar
            self.max_new_entries_per_bar = 0
            weights = super().generate_weights(data, t)
            self.max_new_entries_per_bar = _saved
            return weights

        # ── Regime ON: full strategy ──────────────────────────────────────────
        return super().generate_weights(data, t)


strat = CryptoRegimeOuTrendStrategy()
print("Patched:", strat)

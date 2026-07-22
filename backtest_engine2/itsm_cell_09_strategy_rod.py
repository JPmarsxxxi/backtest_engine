# Cell 9 — Trial 2 (pre-registered queue sketch): rest-of-day variant.
# Same signal (sign of 00:00->00:30 return); hold 00:30->24:00 instead of the last half-hour.
# Enter at the 00:30 bar, exit at the 00:00 (midnight) bar. K=2 for DSR at Stage 9.
def _rod_rebalance_bars(dates):
    hm = dates.hour * 60 + dates.minute
    return dates[(hm == 30) | (hm == 0)]


class IntradayTSMRestOfDay(Strategy):
    """Queue #011 sketch: first-half-hour sign held for the rest of the UTC day."""
    rebalance_frequency = staticmethod(_rod_rebalance_bars)

    def __init__(self, gross: float = 0.5):
        self.gross = gross
        self.risk = RiskConfig(max_position=gross, max_gross=1.0)

    def __repr__(self):
        return f"IntradayTSMRestOfDay(signal=00:00-00:30, trade=00:30-24:00, gross={self.gross})"

    def generate_weights(self, data, t):
        zero = pd.Series(0.0, index=data.assets)
        if not (t.hour == 0 and t.minute == 30):
            return zero                                  # midnight bar: target flat
        day = t.normalize()
        px = data.prices["BTCUSDT"]
        try:
            p_mid = px.at[day]
        except KeyError:
            return zero
        p_0030 = px.at[t]                                # t IS the 00:30 stamp, in-view
        if pd.isna(p_mid) or pd.isna(p_0030) or p_0030 == p_mid:
            return zero
        return pd.Series({"BTCUSDT": self.gross * np.sign(p_0030 / p_mid - 1.0)})


strat_rod = IntradayTSMRestOfDay(gross=0.5)
print(strat_rod)
rb = strat_rod.rebalance_dates(panel.dates)
print(f"Rebalance bars: {len(rb)} (expect ~2/day = {2 * 3220})")

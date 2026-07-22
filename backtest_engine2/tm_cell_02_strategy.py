# Cell 2 — #014 trial 1 strategy: MOP-exact TSMOM (pre-registered).
# sign(252d return) per instrument; w_i = sign_i * (0.40 / sigma_i) / N_alive;
# sigma_i = 60d EWMA ann vol, FLOORED at 5% (peg landmine: EURCHF 2012-14 vol ~1% ->
# 8x position -> Jan-2015 depeg -17%/day; floor is a pre-registered risk rule, not tuning).
# Monthly rebalance (last business day); <252d history -> sit out.
from backtest.strategy import Strategy
from backtest.risk import RiskConfig

LOOKBACK = 252
VOL_SPAN = 60
VOL_FLOOR = 0.05
SCALE = 0.40


def _month_end_bars(dates):
    s = pd.Series(dates, index=dates)
    return pd.DatetimeIndex(s.groupby([dates.year, dates.month]).last().values)


class TSMOM(Strategy):
    """MOP 2012: sign(12m return), 40%/sigma sizing, equal split across live signals."""
    rebalance_frequency = staticmethod(_month_end_bars)

    def __init__(self):
        self.risk = RiskConfig(max_position=0.25, max_gross=6.0)

    def __repr__(self):
        return "TSMOM(sign 252d, 0.40/sigma, vol floor 5%, monthly)"

    def generate_weights(self, data, t):
        px = data.prices
        if len(px) < LOOKBACK + 1:
            return pd.Series(0.0, index=data.assets)
        sig = np.sign(px.iloc[-1] / px.iloc[-(LOOKBACK + 1)] - 1.0)
        vol = (px.pct_change(fill_method=None).ewm(span=VOL_SPAN).std().iloc[-1]
               * np.sqrt(252)).clip(lower=VOL_FLOOR)
        w = (sig * SCALE / vol).fillna(0.0)
        n_alive = int((w != 0).sum())
        if n_alive == 0:
            return pd.Series(0.0, index=data.assets)
        return (w / n_alive).reindex(data.assets).fillna(0.0)


strat = TSMOM()
print(strat)
rb = strat.rebalance_dates(panel.dates)
print(f"Rebalance dates: {len(rb)} month-ends, {rb[0].date()} -> {rb[-1].date()}")

# Dry-run the weight function at three sample dates for sanity.
from backtest.data import DataView
for d in [rb[13], rb[150], rb[-1]]:
    w = strat.generate_weights(DataView(panel, d), d)
    live = w[w != 0]
    print(f"{d.date()}: {len(live)} live | gross {live.abs().sum():.2f} | "
          f"long {int((live > 0).sum())} short {int((live < 0).sum())} | "
          f"max |w| {live.abs().max():.3f} ({live.abs().idxmax()})")

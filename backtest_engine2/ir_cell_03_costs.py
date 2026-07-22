# Cell 3 — #019 cost stack: real US500.cash FTMO costs (no commission, no impact/liqcap).
# Spread: half-spread from specs, floored 0.25bp half (demo prints ~0). Swap: FTMOSwap holding
# cost (reused from #014), per-direction annual rate /252 per daily bar. Baseline models swap as
# the worst case; swap-free Swing shown as a sensitivity at the metrics stage. trading_days=252
# vs ~270 actual bars/yr post-2013 -> minor conservative over-charge of swap.
from backtest.costs import Spread, CompositeCostModel
from backtest.costs.base import CostModel

SPECS = pd.read_parquet(ENG + r"\data\ftmo_specs.parquet")
r = SPECS[SPECS["symbol"] == "US500.cash"].iloc[0]
p = float(prices["US500"].dropna().iloc[-1])

bp_l = 1e4 * r["swap_long"] * r["point"] / p          # swap bp/night, long
bp_s = 1e4 * r["swap_short"] * r["point"] / p          # swap bp/night, short
swap_l = {"US500": bp_l * 364 / 1e4}                   # annual fraction (3x-Wed -> 364 charges)
swap_s = {"US500": bp_s * 364 / 1e4}
half = {"US500": max(1e4 * r["spread_pts"] * r["point"] / p / 2, 0.25)}


class FTMOSwap(CostModel):
    """Per-direction overnight financing; annual fraction (neg = trader pays), rate/td per bar."""
    def __init__(self, long_rate, short_rate, trading_days=252):
        self.long_rate = pd.Series(long_rate)
        self.short_rate = pd.Series(short_rate)
        self.td = trading_days

    def trade_cost(self, trades, view):
        return 0.0

    def holding_cost(self, positions, view):
        pos = positions[positions != 0]
        if pos.empty:
            return 0.0
        rate = pd.Series(np.where(pos > 0, self.long_rate.reindex(pos.index),
                                  self.short_rate.reindex(pos.index)), index=pos.index)
        return float((-rate.fillna(0.0) * pos.abs() / self.td).sum())


costs = CompositeCostModel([Spread(half_bps=half), FTMOSwap(swap_l, swap_s)])
liq = None

print(f"US500 spread: {half['US500']:.2f} bp half ({2*half['US500']:.2f} bp round trip)")
print(f"US500 swap  : long {swap_l['US500']*100:+.2f}%/yr ({bp_l:+.2f} bp/nt) | "
      f"short {swap_s['US500']*100:+.2f}%/yr ({bp_s:+.2f} bp/nt)")
print(f"components: {[type(m).__name__ for m in costs.models]} | liquidity: {liq}")
print(f"panel has volume: {panel.has_field('volume')} (unused — no impact/liqcap)")

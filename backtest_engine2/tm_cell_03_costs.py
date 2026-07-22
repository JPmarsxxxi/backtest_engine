# Cell 3 — #014 real FTMO cost stack from the measured spec table (no sweeps).
# Spread: per-asset half-spread from demo quotes, FLOORED at 0.25bp half (demo prints ~0
# on some indices). Swap: custom holding-cost model, per-instrument AND per-direction,
# annual rate / 252 per daily bar; rates already include the 3x-Wednesday convention;
# credits (USDJPY long, US30 short) are real and count for us. No commission (FX/index/
# metals commission-free), no impact/liquidity cap (size trivial vs FX/index depth).
# Missing instruments get their ASSET-CLASS MEDIAN rates and are flagged loudly.
from backtest.costs import Spread, CompositeCostModel
from backtest.costs.base import CostModel

SPECS = pd.read_parquet(r"C:\Users\User\backtest_engine\backtest_engine2\data\ftmo_specs.parquet")
SPECS["clean"] = SPECS["symbol"].str.replace(".cash", "", regex=False).str.upper()

CLASS = {s: ("fx" if len(s) == 6 and s[:3].isalpha() and not s.startswith(("XA", "UK", "US", "NA"))
        else "other") for s in prices.columns}
CLASS.update({s: "index" for s in ["US500", "US100", "US30", "GER40", "EU50", "FRA40",
                                   "UK100", "JP225", "AUS200", "HK50", "SPA35", "SWI20"]})
CLASS.update({"XAUUSD": "metal", "XAGUSD": "metal", "UKOIL": "energy", "USOIL": "energy",
              "NATGAS": "energy"})

half_bp, swp_l, swp_s, filled = {}, {}, {}, []
for sym in prices.columns:
    row = SPECS[SPECS["clean"] == sym]
    if row.empty:
        filled.append(sym)
        continue
    r = row.iloc[0]
    p = prices[sym].dropna().iloc[-1]
    if r["swap_mode"] == "points":
        bp_l = 1e4 * r["swap_long"] * r["point"] / p
        bp_s = 1e4 * r["swap_short"] * r["point"] / p
        swp_l[sym] = bp_l * 364 / 1e4          # annual fraction
        swp_s[sym] = bp_s * 364 / 1e4
    else:                                       # interest mode: value = annual %
        swp_l[sym] = r["swap_long"] / 100.0
        swp_s[sym] = r["swap_short"] / 100.0
    half_bp[sym] = max(1e4 * r["spread_pts"] * r["point"] / p / 2, 0.25)

for sym in filled:                               # class-median fill, flagged
    peers = [s for s in swp_l if CLASS[s] == CLASS[sym]]
    swp_l[sym] = float(np.median([swp_l[s] for s in peers]))
    swp_s[sym] = float(np.median([swp_s[s] for s in peers]))
    half_bp[sym] = float(np.median([half_bp[s] for s in peers]))
if filled:
    print(f"!! not in specs, filled with {len(filled)} class medians: {filled}")


class FTMOSwap(CostModel):
    """Per-instrument, per-direction overnight financing. Rates = annual fractions
    (negative = trader pays); charged rate/252 per daily bar on held exposure."""

    def __init__(self, long_rate: dict, short_rate: dict, trading_days: int = 252):
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


costs = CompositeCostModel([Spread(half_bps=half_bp), FTMOSwap(swp_l, swp_s)])
liq = None

tbl = pd.DataFrame({"class": pd.Series(CLASS), "half_bp": pd.Series(half_bp).round(2),
                    "swap_long_%yr": (pd.Series(swp_l) * 100).round(2),
                    "swap_short_%yr": (pd.Series(swp_s) * 100).round(2)}).loc[list(prices.columns)]
print(tbl.to_string())
print(f"\nbook-level: gross ~4 x avg|swap| -> rough drag ceiling "
      f"{4 * tbl[['swap_long_%yr', 'swap_short_%yr']].abs().mean().mean():.1f}%/yr if always on the expensive side")

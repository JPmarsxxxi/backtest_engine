# Cell 9 — #013 trial 2 (pre-registered): full close->open hold, swap-parameterized.
# Long at the 16:00 ET bar (cash close), flat at the 9:30 ET bar (cash open). One round
# trip per night collects the diffuse hour-smear from Cell 7. Crosses the ~17:00 ET swap
# cutoff every night -> swap applies; swept as a parameter since real FTMO points unknown.
# Friday entries hold to Monday 9:30 (weekend gap + ~3 swap days) — kept, as in the paper.

def _cv_rebalance_bars(dates):
    et = _et_hour(dates)
    pick = ((et.hour == 16) & (et.minute == 0)) | ((et.hour == 9) & (et.minute == 30))
    return dates[pick]


class CloseToOpenHold(Strategy):
    """Trial 2: hold US500 from cash close (16:00 ET) to cash open (9:30 ET)."""
    rebalance_frequency = staticmethod(_cv_rebalance_bars)

    def __init__(self, gross: float = 1.0):
        self.gross = gross
        self.risk = RiskConfig(max_position=gross, max_gross=1.0)

    def __repr__(self):
        return f"CloseToOpenHold(16:00->9:30 ET, long_only, gross={self.gross})"

    def generate_weights(self, data, t):
        et_t = t.tz_localize("UTC").tz_convert(ET)
        if et_t.hour == 16 and et_t.minute == 0:
            return pd.Series({"USA500": self.gross})
        return pd.Series(0.0, index=data.assets)


strat2 = CloseToOpenHold(gross=1.0)
print(strat2)
rb2 = strat2.rebalance_dates(panel.dates)
et_rb2 = _et_hour(rb2)
print(f"Rebalance bars: {len(rb2)} | entries 16:00: {int((et_rb2.hour == 16).sum())} | "
      f"exits 9:30: {int((et_rb2.hour == 9).sum())}")

engine2 = Engine(panel, strat2, costs=costs, liquidity=None, initial_capital=1_000_000)
result2 = engine2.run(start=panel.dates[48])
print(f"\nEngine (spread-only, no swap): final ${result2.equity_curve.iloc[-1]:,.0f} | "
      f"return {result2.equity_curve.iloc[-1] / 1e6 - 1:+.2%} | "
      f"costs ${result2.costs.sum():,.0f} | rebalances {result2.metadata['n_rebalances']}")

# Gross close->open window return per held night (entry 16:00 -> next 9:30 stamp).
held2 = result2.positions["USA500"]
et_h2 = _et_hour(held2.index)
ent2 = held2.index[(held2.abs() > 1.0).values & (et_h2.hour == 16) & (et_h2.minute == 0)]
opens = px_.index[(et_all.hour == 9) & (et_all.minute == 30)]
nxt = opens.searchsorted(ent2)
valid = nxt < len(opens)
ent2, ext2 = ent2[valid], opens[nxt[valid]]
gross_n = pd.Series(px_.reindex(ext2).values / px_.reindex(ent2).values - 1.0,
                    index=ent2).dropna()
sh2 = gross_n.mean() / gross_n.std() * np.sqrt(250)
print(f"\nGross close->open: {len(gross_n)} nights | hit {100 * (gross_n > 0).mean():.1f}% | "
      f"mean {1e4 * gross_n.mean():+.2f} bps/night | ann {250 * gross_n.mean():+.2%}/yr | "
      f"gross Sharpe {sh2:+.2f}")

print(f"\n{'year':>5} {'nights':>7} {'hit%':>6} {'bps/nt':>8} {'gSharpe':>8}")
for yr, grp in gross_n.groupby(gross_n.index.year):
    shy = grp.mean() / grp.std() * np.sqrt(250) if grp.std() > 0 else float("nan")
    print(f"{yr:>5} {len(grp):>7} {100 * (grp > 0).mean():>6.1f} "
          f"{1e4 * grp.mean():>+8.2f} {shy:>+8.2f}")

# Swap sweep: net/night = gross - 2.0 bp round-trip spread - swap. Fri->Mon holds incur
# ~3 swap days; sweep values are per-HELD-night, so read FTMO's points as wkly avg /5.
print(f"\nSwap sweep (net = gross - 2.0 bp spread - swap), full period and 2022+:")
print(f"{'swap bp/nt':>11} {'net bps/nt':>11} {'net %/yr':>9} {'net Sh':>7} | {'2022+ bps':>10} {'2022+ Sh':>9}")
late = gross_n[gross_n.index.year >= 2022]
for swap in [0.0, 0.5, 1.0, 2.0]:
    net = gross_n - 2.0e-4 - swap * 1e-4
    netl = late - 2.0e-4 - swap * 1e-4
    print(f"{swap:>11.1f} {1e4 * net.mean():>+11.2f} {250 * net.mean():>+9.2%} "
          f"{net.mean() / net.std() * np.sqrt(250):>+7.2f} | "
          f"{1e4 * netl.mean():>+10.2f} {netl.mean() / netl.std() * np.sqrt(250):>+9.2f}")

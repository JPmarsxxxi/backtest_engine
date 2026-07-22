# Cell 8 — Decay check: per-year gross signal quality on the held half-hours.
# Gross trade return = sign(signal) * last-half-hour BTC return, reconstructed from
# prices + the realized position sign at the 23:30 stamp (post-trade, pre-cost).
px = prices["BTCUSDT"]
stamps = px.index
hm = stamps.hour * 60 + stamps.minute

entry_stamps = stamps[hm == 23 * 60 + 30]
pos_sign = np.sign(result.positions["BTCUSDT"].reindex(entry_stamps).fillna(0.0))
exit_stamps = entry_stamps + pd.Timedelta("30min")
ret_lasthh = (px.reindex(exit_stamps).values / px.reindex(entry_stamps).values) - 1.0
ret_lasthh = pd.Series(ret_lasthh, index=entry_stamps)

trade_ret = pos_sign * ret_lasthh          # gross, per held half-hour, sign-only (unit notional)
traded = pos_sign != 0
tr = trade_ret[traded].dropna()

print(f"{'year':>5} {'trades':>7} {'hit%':>6} {'mean bps':>9} {'gross Sh(ann365)':>17}")
for y in range(2017, 2027):
    s = tr[tr.index.year == y]
    if len(s) == 0:
        continue
    sh = s.mean() / s.std() * np.sqrt(365) if s.std() > 0 else float("nan")
    print(f"{y:>5} {len(s):>7} {100 * (s > 0).mean():>6.1f} {1e4 * s.mean():>9.2f} {sh:>17.2f}")

sh_all = tr.mean() / tr.std() * np.sqrt(365)
print(f"\nALL  : {len(tr)} trades | hit {100 * (tr > 0).mean():.1f}% | "
      f"mean {1e4 * tr.mean():+.2f} bps/trade | gross Sharpe {sh_all:+.2f}")
print(f"Cost gate for reference: ~16.5 bps/round trip at FTMO stack")

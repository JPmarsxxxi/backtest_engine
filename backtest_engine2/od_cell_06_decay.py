# Cell 6 — Stage 5 diagnostic: corrected gross window measurement + per-year decay.
# Cell 5's filter caught hour==2 without minute==0 -> double-counted 2:00 and 2:30 stamps
# (3,829 "nights" ~= 2x1,916), blending the true window with an overlapping 2:30-3:30 one.
# Fix: entry = held bar stamped exactly 02:00 ET; exit = entry + 1h. ~1,916 true nights.
px_ = prices["USA500"]
held = result.positions["USA500"]
et_idx = _et_hour(held.index)
entry_stamps = held.index[(held.abs() > 1.0).values
                          & (et_idx.hour == 2) & (et_idx.minute == 0)]
exit_stamps = entry_stamps + pd.Timedelta("1h")
ret_n = pd.Series(px_.reindex(exit_stamps).values / px_.reindex(entry_stamps).values - 1.0,
                  index=entry_stamps).dropna()

sh = ret_n.mean() / ret_n.std() * np.sqrt(250)
print(f"True window (2:00->3:00 ET): {len(ret_n)} nights | hit {100 * (ret_n > 0).mean():.1f}% | "
      f"mean {1e4 * ret_n.mean():+.2f} bps/night | ann {250 * ret_n.mean():+.2%}/yr | "
      f"gross Sharpe {sh:+.2f}")
print("Paper benchmark: ~+1.4 bps/night, ~+3.6%/yr (1998-2019) | cost gate ~2.0 bp/night\n")

# Per-year decay table (the #011-style read: decayed vs never-worked).
print(f"{'year':>5} {'nights':>7} {'hit%':>6} {'bps/nt':>8} {'gSharpe':>8}")
for yr, grp in ret_n.groupby(ret_n.index.year):
    shy = grp.mean() / grp.std() * np.sqrt(250) if grp.std() > 0 else float("nan")
    print(f"{yr:>5} {len(grp):>7} {100 * (grp > 0).mean():>6.1f} "
          f"{1e4 * grp.mean():>+8.2f} {shy:>+8.2f}")

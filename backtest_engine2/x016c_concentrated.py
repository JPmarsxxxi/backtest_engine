# #016 deepening 2 — concentrated book (only train |t|>4 hours: 21,22,23) to cut the 8x cost multiplier.
# Fair last check: does the highest-conviction subset survive realistic spread + show gross stability?
HRS = {21: -1.0, 22: +1.0, 23: +1.0}   # signs from train 2019-2021 (all |t|>4)
n_h = len(HRS)

t2 = df[df.yr >= 2022].copy()
t2["sret"] = [HRS.get(hh, 0.0) for hh in t2.hr] * t2["ret"]
t2["day"] = pd.to_datetime(pd.Series(t2.index.tz_convert("Europe/London").date, index=t2.index))
daily = t2[t2.hr.isin(HRS)].groupby("day")["sret"].sum()

g_sh = daily.mean() / daily.std() * np.sqrt(252)
print(f"CONCENTRATED 3-hour book {sorted(HRS)} | TEST 2022-26 | {len(daily)} days")
print(f"  GROSS {1e4*daily.mean():+.2f} bp/day | Sharpe {g_sh:+.2f} | hit {100*(daily>0).mean():.0f}%")
print("  GROSS per-year (is the signal stable, before cost?):")
for yr, gy in daily.groupby(daily.index.year):
    print(f"    {yr}: gross {1e4*gy.mean():+.2f} bp/day | Sh {gy.mean()/gy.std()*np.sqrt(252):+.2f}")
print(f"\n  NET by half-spread (cost = {n_h} hrs x 2 x half_bp):")
for hb in [0.2, 0.3, 0.5, 1.0]:
    net = daily - n_h * 2 * hb / 1e4
    print(f"    half {hb:.1f}bp: net {1e4*net.mean():+.2f} bp/day | Sharpe {net.mean()/net.std()*np.sqrt(252):+.2f} | ann {net.mean()*252*100:+.1f}%")
print("  NB: 21:00-23:00 London are thin-liquidity hours -> REAL spread there is wider than daytime.")

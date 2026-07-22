# Cell 3 — #017b downside-asymmetry test (mechanism-native). Short-gamma hedging amplifies
# SELLOFFS more than rallies -> low-gamma continuation should be cleaner on DOWN-mornings.
# PIT expanding low-gamma flag (t1). Split low-gamma trade days by morning direction.
# Pre-registered: the DOWN leg should be positive in a majority of years AND not 2025+-only.
d = base[base.t1 == True].copy()
d["r"] = d.morn_sign * d.aft_ret                 # continuation return (>0 = morning move continued)
down = d[d.morn_sign < 0]                         # selloff mornings -> bet selloff continues (SHORT)
up = d[d.morn_sign > 0]                            # rally mornings   -> bet rally continues (LONG)

def leg(x, label):
    r = x["r"].dropna()
    yrs = sorted(set(r.index.year))
    pos = sum(1 for y in yrs if r[r.index.year == y].mean() > 0)
    rec, old = r[r.index.year >= 2025], r[r.index.year < 2025]
    print(f"\n--- {label}")
    print(f"    {len(r)} trades | hit {100*(r>0).mean():.1f}% | gross {1e4*r.mean():+.2f} bps/trade | pos yrs {pos}/{len(yrs)}")
    print(f"    per yr bps: {{{', '.join(f'{int(y)}:{1e4*r[r.index.year==y].mean():+.0f}' for y in yrs)}}}")
    print(f"    pre-2025: {len(old)}tr {1e4*old.mean():+.2f}bps | 2025+: {len(rec)}tr {1e4*rec.mean():+.2f}bps")

leg(down, "DOWN-morning low-gamma (selloff continuation, SHORT) -- mechanism's favored leg")
leg(up, "UP-morning low-gamma (rally continuation, LONG)")

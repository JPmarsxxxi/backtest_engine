# Cell 2 — #012 decay check FIRST (panel-level, no engine): does the first half-hour
# still predict the last half-hour, 2019-2026 (all post-publication)?
# t1: signal = prior 16:00 close -> 10:00 (gap incl., paper headline predictor)
# t2: signal = 9:30 -> 10:00 (gap excl.)
# target: 15:30 -> 16:00 return; strategy return = sign(signal) * target.
def _stamp(hh, mm):
    m = (et_idx.hour == hh) & (et_idx.minute == mm)
    out = px_[m]
    out.index = pd.DatetimeIndex(et_idx[m].date)  # key by ET trading day
    return out

p0930, p1000 = _stamp(9, 30), _stamp(10, 0)
p1530, p1600 = _stamp(15, 30), _stamp(16, 0)

days = p1000.index.intersection(p1530.index).intersection(p1600.index)
prev_close = p1600.reindex(days).shift(1)  # prior trading day's close

sig1 = (p1000.reindex(days) / prev_close - 1).dropna()                 # t1: gap incl.
sig2 = (p1000.reindex(days) / p0930.reindex(days) - 1).dropna()        # t2: gap excl.
tgt = (p1600.reindex(days) / p1530.reindex(days) - 1)

for name, sig in [("t1 gap-incl (paper)", sig1), ("t2 first-30m only", sig2)]:
    r = (np.sign(sig) * tgt.reindex(sig.index)).dropna()
    r = r[np.sign(sig.reindex(r.index)) != 0]
    sh = r.mean() / r.std() * np.sqrt(250)
    print(f"{name}: {len(r)} days | hit {100 * (r > 0).mean():.1f}% | "
          f"gross {1e4 * r.mean():+.2f} bps/trade | Sharpe {sh:+.2f}")
print("paper (1993-2013): ~+3-7 bps/trade | cost gate ~2.0 bps/round trip\n")

print(f"{'year':>5} | {'t1 bps':>7} {'t1 hit':>6} {'t1 Sh':>6} | {'t2 bps':>7} {'t2 hit':>6} {'t2 Sh':>6}")
for yr in range(2019, 2027):
    row = []
    for sig in (sig1, sig2):
        r = (np.sign(sig) * tgt.reindex(sig.index)).dropna()
        r = r[r.index.year == yr]
        if len(r) < 30:
            row.append(None)
            continue
        row.append((1e4 * r.mean(), 100 * (r > 0).mean(),
                    r.mean() / r.std() * np.sqrt(250)))
    f = lambda c: f"{c[0]:>+7.2f} {c[1]:>6.1f} {c[2]:>+6.2f}" if c else " " * 21
    print(f"{yr:>5} | {f(row[0])} | {f(row[1])}")

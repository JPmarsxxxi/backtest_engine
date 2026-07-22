# Cell 7 — Stage 5 diagnostic: hour-of-night decomposition.
# Where does overnight drift live now? Paper (SR917) said the premium migrated across
# hours over its sample. For each 1h window h:00->h+1:00 ET through the non-RTH session,
# mean bps, hit%, gross Sharpe — split early (2019-2021) vs late (2022-2026).
px_ = prices["USA500"]
et_all = _et_hour(px_.index)
HOURS = [18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # futures reopen -> RTH open

def _hour_ret(h):
    ent = px_.index[(et_all.hour == h) & (et_all.minute == 0)]
    ext = ent + pd.Timedelta("1h")
    return pd.Series(px_.reindex(ext).values / px_.reindex(ent).values - 1.0,
                     index=ent).dropna()

def _stats(r):
    if len(r) < 50:
        return None
    return (len(r), 100 * (r > 0).mean(), 1e4 * r.mean(),
            r.mean() / r.std() * np.sqrt(250) if r.std() > 0 else float("nan"))

print(f"{'ET hr':>6} | {'n':>5} {'hit%':>5} {'bps':>6} {'gSh':>6}  <- 2019-2021 | "
      f"{'n':>5} {'hit%':>5} {'bps':>6} {'gSh':>6}  <- 2022-2026")
for h in HOURS:
    r = _hour_ret(h)
    early = _stats(r[r.index.year <= 2021])
    late = _stats(r[r.index.year >= 2022])
    fe = f"{early[0]:>5} {early[1]:>5.1f} {early[2]:>+6.2f} {early[3]:>+6.2f}" if early else " " * 25
    fl = f"{late[0]:>5} {late[1]:>5.1f} {late[2]:>+6.2f} {late[3]:>+6.2f}" if late else " " * 25
    print(f"{h:>4}:00 | {fe} | {fl}")
print("\ncost gate ~2.0 bp/window (1bp half-spread x2) | paper window = 2:00")

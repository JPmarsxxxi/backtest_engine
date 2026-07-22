# Cell 2 — #017 decay-check (DECISION GATE), panel-level, no engine. Depends on Cell 1
# (px_, et_idx, gex_lag, overlap).
# Hypothesis: LOW-gamma days TREND (intraday acf1>0, morning move continues into the close);
# HIGH-gamma days REVERT (acf1<0, morning move reverses). #012 found this flat UNCONDITIONALLY
# (-0.1..-0.6 bps) -- the GEX claim is that the two regimes cancel. Split by lagged-GEX quintile.
# Pre-registered rule: SURVIVES iff acf1 & continuation are MONOTONE in regime AND opposite-signed
# at the extremes (Q1 low-gamma trends, Q5 high-gamma reverts); else KILLED like #012.
from scipy.stats import spearmanr

tmin = et_idx.hour * 60 + et_idx.minute
et_date = pd.DatetimeIndex(et_idx.normalize().date)
df = pd.DataFrame({"px": px_.values, "date": et_date.values, "tmin": tmin.values}, index=px_.index)

# --- per-day intraday character: lag-1 autocorr of RTH 30m log returns + realized vol ---
rth = df[(df.tmin >= 570) & (df.tmin <= 960)]            # 9:30=570 .. 16:00=960 ET

def _day(g):
    r = np.log(g.sort_values("tmin")["px"]).diff().dropna().values
    if len(r) < 6 or r.std() == 0:
        return pd.Series({"acf1": np.nan, "rthvol": np.nan, "n": len(r)})
    return pd.Series({"acf1": np.corrcoef(r[:-1], r[1:])[0, 1], "rthvol": r.std(), "n": len(r)})

ds = rth.groupby("date").apply(_day, include_groups=False)

# --- tradable continuation: open(9:30, fallback 10:00) -> 12:00 -> 16:00 ---
def _at(t):
    sub = df[df.tmin == t]
    return pd.Series(sub["px"].values, index=pd.DatetimeIndex(sub["date"].values))

p_open = _at(570).combine_first(_at(600))
p_noon, p_close = _at(720), _at(960)
morn = (p_noon / p_open - 1)
aft = (p_close / p_noon - 1)
cont = (np.sign(morn) * aft).rename("cont")            # continuation return (signed by morning)

day = ds.join(cont).join(gex_lag.rename("gex")).dropna(subset=["acf1", "gex"])

def _report(sub, label):
    sub = sub.copy()
    sub["q"] = pd.qcut(sub["gex"], 5, labels=[1, 2, 3, 4, 5])
    print(f"\n=== {label}  (n={len(sub)} days) ===")
    print(f"{'Qgex':>4} {'N':>4} {'meanGEX':>8} {'acf1':>7} {'cont_bps':>9} {'cont_hit':>8} {'rthvol_bp':>9}")
    for q in [1, 2, 3, 4, 5]:
        s = sub[sub.q == q]
        c = s["cont"].dropna()
        print(f"{q:>4} {len(s):>4} {s.gex.mean()/1e9:>8.2f} {s.acf1.mean():>+7.3f} "
              f"{1e4*c.mean():>+9.2f} {100*(c>0).mean():>7.1f}% {1e4*s.rthvol.mean():>9.2f}")
    rho_a, p_a = spearmanr(sub["gex"], sub["acf1"])
    cc = sub.dropna(subset=["cont"])
    rho_c, p_c = spearmanr(cc["gex"], cc["cont"])
    q1, q5 = sub[sub.q == 1], sub[sub.q == 5]
    print(f"spearman(GEX, acf1) = {rho_a:+.3f} (p={p_a:.3f})  [neg = hypothesis: more gamma -> more reversion]")
    print(f"spearman(GEX, cont) = {rho_c:+.3f} (p={p_c:.3f})  [neg expected]")
    print(f"Q1(low-gamma) acf1 {q1.acf1.mean():+.3f} cont {1e4*q1.cont.mean():+.2f}bps  |  "
          f"Q5(high-gamma) acf1 {q5.acf1.mean():+.3f} cont {1e4*q5.cont.mean():+.2f}bps")

_report(day[day.index.year >= 2022], "2022+ (post-popularization, the decay question)")
_report(day, "2019-2026 full")

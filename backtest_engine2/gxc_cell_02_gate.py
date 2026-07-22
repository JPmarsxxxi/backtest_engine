# Cell 2 — #017b PIT continuation gate (K=2, gross, no engine — like #012/#013).
# Long-only continuation: on low-gamma days take +sign(9:30->12:00) held 12:00->16:00.
# t1 = expanding/absolute low-gamma (Cell 1, faithful to dollar-gamma magnitude, 2022-concentrated).
# t2 = rolling-1yr RELATIVE low-gamma (spreads regimes, removes the vol-regime confound).
# Pre-registered survive rule: positive in a MAJORITY of years AND positive ex-2022. Cost gate ~1-2 bp.

def _roll_pct(s, win=252, mp=120):
    return s.rolling(win, min_periods=mp).apply(
        lambda x: float((x.iloc[:-1] < x.iloc[-1]).mean()) if len(x) > 1 else np.nan, raw=False)

t2_low = (_roll_pct(gj) < 0.20).rename("t2")
base = pd.concat([feat[["morn_sign", "aft_ret"]], feat["low_g"].rename("t1"), t2_low], axis=1)
base = base[(base.morn_sign != 0)].dropna(subset=["morn_sign", "aft_ret"])

def gate(col, label):
    d = base[base[col] == True]
    r = (d.morn_sign * d.aft_ret).dropna()
    yrs = sorted(set(r.index.year))
    tpy = len(r) / len(yrs)
    sh = r.mean() / r.std() * np.sqrt(tpy) if r.std() > 0 else float("nan")
    print(f"\n--- {label}")
    print(f"  ALL : {len(r)} trades | hit {100*(r>0).mean():.1f}% | gross {1e4*r.mean():+.2f} bps/trade | Sharpe(ann) {sh:+.2f}")
    py = {int(y): (int((r.index.year == y).sum()), round(1e4 * r[r.index.year == y].mean(), 1)) for y in yrs}
    print(f"  per yr (n, bps): {py}")
    ex = r[r.index.year != 2022]
    pos = sum(1 for y in yrs if r[r.index.year == y].mean() > 0)
    print(f"  ex-2022: {len(ex)} trades | hit {100*(ex>0).mean():.1f}% | gross {1e4*ex.mean():+.2f} bps/trade | positive years {pos}/{len(yrs)}")
    return r

r1 = gate("t1", "t1 expanding/absolute low-gamma")
r2 = gate("t2", "t2 rolling-1yr relative low-gamma")
print(f"\ncorr(t1,t2 trade-day returns overlap): "
      f"{pd.concat([(base[base.t1].morn_sign*base[base.t1].aft_ret).rename('a'), (base[base.t2].morn_sign*base[base.t2].aft_ret).rename('b')], axis=1).corr().iloc[0,1]:.2f}")

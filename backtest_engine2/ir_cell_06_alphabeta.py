# Cell 6 — #019 alpha/beta decomposition (DECIDER): is the long-only dip-buyer ALPHA or timed BETA?
# Regress strategy net & gross returns on US500 buy-and-hold. Pre-registered: REAL iff alpha t>2 AND
# alpha positive in 2014-26; if alpha~0 and beta+R2 carry it -> timed equity beta -> kill/downgrade.
mkt = prices["US500"].pct_change().reindex(ret.index)

def ols(y, x):
    X = np.column_stack([np.ones(len(x)), x])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ b
    s2 = (resid @ resid) / (len(y) - 2)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    r2 = 1 - (resid @ resid) / (((y - y.mean()) ** 2).sum())
    return b, b / se, r2

print(f"{'era':>12} {'beta':>6} {'a_net%/yr':>10} {'t(a_net)':>9} {'a_gross%/yr':>12} {'R2':>5} "
      f"{'corr_mkt':>9} {'mkt_Sh':>7}")
for lab, m in [("full 2000-26", ret.index.year >= 2000),
               ("2014-2026", ret.index.year >= 2014),
               ("2022-2025", (ret.index.year >= 2022) & (ret.index.year <= 2025))]:
    rr = ret[m]
    gg = ret_g.reindex(rr.index).fillna(0.0)
    mm = mkt.reindex(rr.index).fillna(0.0)
    bn, tn, r2 = ols(rr.values, mm.values)
    bg, tg, _ = ols(gg.values, mm.values)
    corr = np.corrcoef(rr.values, mm.values)[0, 1]
    mkt_sh = mm.mean() / mm.std() * annf
    print(f"{lab:>12} {bn[1]:>6.2f} {bn[0]*ppy*100:>+10.2f} {tn[0]:>+9.1f} {bg[0]*ppy*100:>+12.2f} "
          f"{r2:>5.2f} {corr:>9.2f} {mkt_sh:>+7.2f}")

print("\nread: high beta + high R2 + alpha t<2  => timed beta (just own the index).")
print("      alpha t>2 at low beta, positive 2014-26 => real (modest) alpha.")

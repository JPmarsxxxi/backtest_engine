# Cell 3 — #018 horse-race: is GEX more than buy-the-dip / a vol proxy?
# Low-gamma days ARE high-vol days that follow selloffs -> the +17bp could be plain short-term
# reversal. Test whether GEX adds info BEYOND trailing realized vol (rv20) and past return.
# Pre-registered "more than buy-the-dip" iff: gex t-stat survives |t|>2 controlling for rv20 AND
# the high-vol double-sort still shows low-GEX > high-GEX.
daily["rv20"] = daily["price"].pct_change().rolling(20).std()
daily["pastret5"] = daily["price"] / daily["price"].shift(5) - 1
dd = daily.dropna(subset=["gex", "rv20", "pastret5", "fwd1"]).copy()

def z(s):
    return ((s - s.mean()) / s.std()).values

def ols(y, X):
    X1 = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.lstsq(X1, y, rcond=None)[0]
    resid = y - X1 @ beta
    s2 = (resid @ resid) / (len(y) - X1.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X1.T @ X1)))
    return beta, beta / se

y = dd["fwd1"].values * 1e4  # bps
print("univariate spearman with fwd1:")
for c in ["gex", "rv20", "pastret5"]:
    rho, p = spearmanr(dd[c], dd["fwd1"])
    print(f"  {c:>9}: {rho:+.3f} (p={p:.3f})")

b, t = ols(y, np.column_stack([z(dd.gex), z(dd.rv20)]))
print(f"\nfwd1(bps) ~ z(gex)+z(rv20):        gex {b[1]:+5.1f} (t {t[1]:+.1f}) | rv20 {b[2]:+5.1f} (t {t[2]:+.1f})")
b, t = ols(y, np.column_stack([z(dd.gex), z(dd.rv20), z(dd.pastret5)]))
print(f"fwd1 ~ z(gex)+z(rv20)+z(past5):    gex {b[1]:+5.1f} (t {t[1]:+.1f}) | rv20 {b[2]:+5.1f} (t {t[2]:+.1f}) | past5 {b[3]:+5.1f} (t {t[3]:+.1f})")

hv = dd[dd.rv20 >= dd.rv20.quantile(0.60)].copy()
hv["gq"] = pd.qcut(hv.gex, 3, labels=["low-γ", "mid", "high-γ"])
print(f"\nHigh-vol days only (rv20 top 40%, n={len(hv)}) — fwd1 by GEX tercile:")
for q, r in hv.groupby("gq", observed=True)["fwd1"].agg(["count", "mean"]).iterrows():
    print(f"   {q:>6}: n={int(r['count']):>4}  fwd1 {1e4*r['mean']:+6.1f} bps")

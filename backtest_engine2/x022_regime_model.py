# #022 — Walk-forward regime model. Predicts P(>10% drawdown within next 63 trading days)
# from 3 PIT features (trend, realized vol, yield curve). Gentle L2 logistic, expanding window,
# retrain every 126 trading days. Strictly causal: each retrain trains only on labels fully
# observed by the retrain date; predictions are out-of-sample. Bands -> position (calm/shaky/danger).
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

ENG = r"C:\Users\User\backtest_engine\backtest_engine2"
rt = pd.read_parquet(ENG + r"\data\rates.parquet")
spx = rt["SPX"].dropna(); ret = spx.pct_change(fill_method=None)
F = pd.DataFrame(index=spx.index)
F["trend"] = spx / spx.rolling(200).mean() - 1
F["rvol"]  = ret.rolling(20).std() * np.sqrt(252)
F["yc"]    = rt["YC_SPREAD"]
H, DD = 63, 0.10                                  # horizon 1 quarter, danger = >10% drop
fwd_min = spx[::-1].rolling(H).min()[::-1].shift(-1)   # min price over next H days (excl today)
label = ((fwd_min / spx - 1) <= -DD).astype(float)
data = pd.concat([F, label.rename("y"), ret.rename("ret")], axis=1).dropna(subset=["trend","rvol","yc"])
data = data[data.index >= "1962-01-01"]
feat = ["trend","rvol","yc"]

idx = data.index
prob = pd.Series(np.nan, index=idx)
MIN_TRAIN, STEP = 1500, 126
starts = list(range(MIN_TRAIN, len(idx), STEP))
for s in starts:
    R = idx[s]
    # train on rows whose forward label is fully observed by R (T + H days <= R) and y not NaN
    train = data.iloc[:s].copy()
    obs = train.index.to_series().shift(-H).bfill() <= R   # crude; safer: only labels with full window
    train = train[train["y"].notna()]
    train = train[train.index <= idx[max(0, s - H)]]       # ensure label window ended before R
    if train["y"].nunique() < 2 or len(train) < 500:
        continue
    sc = StandardScaler().fit(train[feat])
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
    clf.fit(sc.transform(train[feat]), train["y"])
    seg = idx[s: s + STEP]
    prob.loc[seg] = clf.predict_proba(sc.transform(data.loc[seg, feat]))[:, 1]

data["prob"] = prob
oos = data.dropna(subset=["prob"]).copy()

# position from probability bands, lagged 1 day (decide at close T, hold T+1)
def band(p): return np.where(p > 0.60, 0.0, np.where(p > 0.30, 0.5, 1.0))
oos["pos"] = pd.Series(band(oos["prob"].values), index=oos.index).shift(1).fillna(0)

def stats(r):
    r = r.dropna(); eq = (1 + r).cumprod(); dd = (eq / eq.cummax() - 1).min()
    cagr = eq.iloc[-1] ** (252 / len(r)) - 1
    return cagr * 100, r.mean() / r.std() * np.sqrt(252), dd * 100

def report(name, sub):
    bh = stats(sub["ret"]); st = stats(sub["pos"] * sub["ret"])
    tr = (sub["trend"] >= 0).shift(1).fillna(0)                  # simple 200d trend filter baseline
    sf = stats(tr * sub["ret"])
    auc = roc_auc_score(sub["y"], sub["prob"]) if sub["y"].nunique() > 1 else float("nan")
    print(f"\n[{name}]  n={len(sub)}  AUC={auc:.3f}  (0.5=coinflip, model ranks danger correctly above that)")
    print(f"  {'strategy':16s} {'CAGR%':>6s} {'Sharpe':>7s} {'maxDD%':>7s}")
    print(f"  {'buy & hold':16s} {bh[0]:6.1f} {bh[1]:7.2f} {bh[2]:7.1f}")
    print(f"  {'simple 200d':16s} {sf[0]:6.1f} {sf[1]:7.2f} {sf[2]:7.1f}")
    print(f"  {'MODEL (bands)':16s} {st[0]:6.1f} {st[1]:7.2f} {st[2]:7.1f}")

print(f"OOS predictions {oos.index.min().date()} -> {oos.index.max().date()}  | danger base rate {100*data['y'].mean():.0f}%")
report("DEVELOP 1962-2018", oos[oos.index.year <= 2018])
report("VAULT 2019-2026 (one-shot)", oos[oos.index.year >= 2019])

# did danger spike before the big crashes?
print("\nAvg model danger-prob in the 60 days BEFORE major peaks (higher = earlier warning):")
for yr, lbl in [(2008,"GFC"),(2020,"COVID"),(2022,"rate-shock")]:
    w = oos[(oos.index >= f"{yr-1}-09-01") & (oos.index <= f"{yr}-12-31")]
    if len(w): print(f"  {lbl:10s} {yr}: peak danger-prob {w['prob'].max():.2f}, avg {w['prob'].mean():.2f}")

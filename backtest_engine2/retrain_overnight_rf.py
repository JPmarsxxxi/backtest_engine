"""#023-v2 — ANNUAL walk-forward refit of the live RF artifact, automated (user 2026-07-17: "bake
the walk-forward retraining in now").

Runs monthly via run_monthly_health_016.py but acts only when the live artifact is >= MAX_AGE_DAYS
old (age-based, not calendar-based -> self-heals missed Januaries). Recipe is FROZEN (the graduated
walk-forward spec, alpha log 2026-07-17): 30 features via overnight_rf_features (the parity-asserted
module), RandomForest d8/l50/none/seed16, expanding window = all valid nights to date, threshold
untouched. Hyperparameters are NEVER retuned here — that requires a new counted tournament in the
notebook.

SAFETY GATES (any failure -> alert + KEEP the old artifact; a stale model beats a broken one):
  G1 data: SPY zero-gap rate < 3% overall (the fake-open tripwire), base rate in [0.50, 0.62],
     last row within 5 calendar days of today, no NaNs in the final matrix.
  G2 growth: new training set >= old artifact's n_train (data must only accumulate).
  G3 behavior: new model's traded-fraction over the trailing 250 nights in [40%, 95%]
     (a model that trades never/always is broken, whatever its internals say).
Old artifact archived as overnight_rf_v2_prev.joblib; each refit also saved as
overnight_rf_v2_<year>.joblib for the audit trail.

  python retrain_overnight_rf.py            # age check -> maybe refit
  python retrain_overnight_rf.py --force    # refit now regardless of age (still gated)
"""
import os
import shutil
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

ENG = r"C:\Users\User\backtest_engine\backtest_engine2"
DATA = ENG + r"\data"
ARTIFACT = DATA + r"\overnight_rf_v2.joblib"
ALERTLOG = DATA + r"\overnight_alert.log"
LOGFILE = DATA + r"\retrain_overnight_rf.out"
MAX_AGE_DAYS = 350

if sys.stdout is None:                       # pythonw (see ops runbook)
    sys.stdout = sys.stderr = open(LOGFILE, "a", encoding="utf-8")

sys.path.insert(0, ENG)
import overnight_rf_features as orf  # noqa: E402


def alert(msg):
    line = f"{datetime.now():%Y-%m-%d %H:%M} ALERT [023v2-retrain]: {msg}"
    print(">>> " + line, flush=True)
    try:
        with open(ALERTLOG, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def run():
    force = "--force" in sys.argv
    old = joblib.load(ARTIFACT)
    age = (pd.Timestamp.now() - pd.Timestamp(old["frozen_at"])).days
    print(f"\n{datetime.now():%Y-%m-%d %H:%M} retrain check: artifact age {age}d "
          f"(trained through {old['trained_through']}, refit at >={MAX_AGE_DAYS}d)")
    if age < MAX_AGE_DAYS and not force:
        print("   artifact fresh — no refit.")
        return 0

    from sklearn.ensemble import RandomForestClassifier
    try:
        px = orf.yahoo_ohlc("SPY")
        vix = orf.yahoo_ohlc("%5EVIX")["close"]
    except Exception as e:
        alert(f"refit ABORTED — data pull failed: {e}. Keeping old artifact.")
        return 1
    F, on_bp, valid = orf.build_features(px, vix)
    y = (on_bp > 0).astype(int)
    ds = F[valid & F.notna().all(axis=1)]
    yv = y[ds.index]

    # G1 — data sanity (the traps this program has actually been bitten by)
    zero_rate = float(((px.open.shift(-1) / px.close - 1) == 0).mean())
    base = float(yv.mean())
    last_gap = (pd.Timestamp.now().normalize() - ds.index.max()).days
    if zero_rate > 0.03:
        alert(f"G1 FAIL: zero-gap rate {zero_rate:.1%} > 3% — opens look synthetic again. Keeping old.")
        return 1
    if not (0.50 <= base <= 0.62):
        alert(f"G1 FAIL: base rate {base:.3f} outside [0.50, 0.62]. Keeping old.")
        return 1
    if last_gap > 5:
        alert(f"G1 FAIL: newest usable night {ds.index.max().date()} is {last_gap}d old. Keeping old.")
        return 1
    # G2 — data may only accumulate
    if len(ds) < old["n_train"]:
        alert(f"G2 FAIL: new n_train {len(ds):,} < old {old['n_train']:,}. Keeping old.")
        return 1

    model = RandomForestClassifier(n_estimators=500, max_depth=8, min_samples_leaf=50,
                                   class_weight=None, random_state=16, n_jobs=-1)
    model.fit(ds[old["feats"]], yv)

    # G3 — behavioral sanity on the trailing 250 nights
    p_recent = model.predict_proba(ds[old["feats"]].iloc[-250:])[:, 1]
    frac = float((p_recent >= old["threshold"]).mean())
    if not (0.40 <= frac <= 0.95):
        alert(f"G3 FAIL: trailing-250 traded fraction {frac:.1%} outside [40%, 95%]. Keeping old.")
        return 1

    shutil.copy2(ARTIFACT, DATA + r"\overnight_rf_v2_prev.joblib")
    art = {"model": model, "feats": old["feats"], "threshold": old["threshold"],
           "trained_through": str(ds.index.max().date()), "n_train": len(ds),
           "recipe": old["recipe"],
           "frozen_at": f"{pd.Timestamp.now():%Y-%m-%d %H:%M}",
           "refit_lineage": old.get("refit_lineage", []) + [old["frozen_at"]]}
    joblib.dump(art, ARTIFACT)
    joblib.dump(art, DATA + rf"\overnight_rf_v2_{datetime.now().year}.joblib")
    print(f"   REFIT OK: {len(ds):,} nights through {art['trained_through']} "
          f"(was {old['n_train']:,} through {old['trained_through']}) | "
          f"trailing traded-frac {frac:.1%} | prev archived")
    alert(f"annual refit completed: trained through {art['trained_through']}, "
          f"n={len(ds):,} (gates G1-G3 passed). Note it in the alpha log.")
    return 0


if __name__ == "__main__":
    sys.exit(run())

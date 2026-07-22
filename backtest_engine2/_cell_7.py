# Cell 8 — DSR / Selection
# K = 1: only strategy tried this session.
# CPCV paths are one trial measured 5 ways, not 5 independent trials.

import numpy as np
from backtest.selection import dsr, effective_k, expected_max_sr
from backtest.metrics.core import sharpe_var_term

ANN_FACTOR = 8760   # hourly crypto (24h × 365d)
K          = 1      # single trial this session

# ── Per-bar SR variance from CPCV paths (empirical override) ─────────────────
path_sharpes_ann = np.array([rep.sharpe for rep in per_path])
path_sharpes_pb  = path_sharpes_ann / np.sqrt(ANN_FACTOR)
sr_var_empirical = float(np.var(path_sharpes_pb, ddof=1))   # per-bar variance

print(f"CPCV path Sharpes (ann.): {np.round(path_sharpes_ann, 3).tolist()}")
print(f"Empirical per-bar SR var : {sr_var_empirical:.6f}")

# ── DSR on each CPCV path ─────────────────────────────────────────────────────
print(f"\nDSR per CPCV path  (K={K}, empirical sr_variance)")
print(f"{'path':<10}{'sharpe_ann':>12}{'DSR':>10}")
print("-" * 34)
path_dsr_vals = []
for col in res.path_returns.columns:
    ret = res.path_returns[col].dropna()
    d   = dsr(ret, K=K, sr_variance=sr_var_empirical)
    path_dsr_vals.append(d)
    sr_ann = ret.mean() / ret.std(ddof=1) * np.sqrt(ANN_FACTOR)
    print(f"{col:<10}{sr_ann:>12.3f}{d:>10.3f}")

print(f"\nMedian DSR : {np.nanmedian(path_dsr_vals):.3f}")
print(f"Min DSR    : {np.nanmin(path_dsr_vals):.3f}")
print(f"Max DSR    : {np.nanmax(path_dsr_vals):.3f}")

# ── Single-path DSR (reference) ───────────────────────────────────────────────
r_single   = result.returns.dropna().to_numpy()
T          = len(r_single)
sr_pb      = r_single.mean() / r_single.std(ddof=1)
var_term   = sharpe_var_term(r_single, sr_pb)
luck_max_pb  = expected_max_sr(K, var_term / T)
luck_max_ann = luck_max_pb * np.sqrt(ANN_FACTOR)
dsr_single = dsr(result.returns, K=K, sr_variance=sr_var_empirical)

print(f"\n── Single-path reference (K={K}) ──")
print(f"Realized Sharpe (ann.) : {sr_pb * np.sqrt(ANN_FACTOR):.3f}")
print(f"Luck-max threshold     : {luck_max_ann:.4f}  (E[max SR | K={K}])")
print(f"DSR (empirical var)    : {dsr_single:.3f}")
print(f"PSR(0) [= DSR at K=1]  : {single_rep.psr:.3f}")
sig = "significant" if dsr_single > 0.95 else "NOT significant"
print(f"→ {sig} at 95% (DSR = {dsr_single:.3f})")

# ── Effective-K on CPCV paths (transparency) ─────────────────────────────────
k_eff = effective_k(res.path_returns.dropna(), threshold=0.5)
print(f"\neffective_k(path_returns, threshold=0.5) = {k_eff}  "
      f"(paths are correlated variants of one trial — K stays {K})")

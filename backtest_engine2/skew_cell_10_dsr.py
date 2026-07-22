# Cell 10 — DSR for flip30 (baseline costs), deflated by this session's trials
import numpy as np
from backtest.selection import dsr, effective_k, expected_max_sr
from backtest.metrics.core import sharpe_var_term

trials = pd.concat([
    result_skew.returns.rename("short21"),
    result_flip.returns.rename("flip21"),
    result_flip30.returns.rename("flip30"),
    result_m30.returns.rename("mom30"),
], axis=1).dropna()

K_raw = 4
K_eff = effective_k(trials, threshold=0.5)

r = result_flip30.returns.dropna().to_numpy()
T = len(r)
sr_pb = r.mean() / r.std(ddof=1)
var_term = sharpe_var_term(r, sr_pb)
print("Trial corr matrix:")
print(trials.corr().round(2).to_string())
print()
print(f"K raw = {K_raw} | K effective (thr 0.5) = {K_eff}")
for K in sorted({K_raw, K_eff, 1}):
    luck_ann = expected_max_sr(K, var_term / T) * np.sqrt(365)
    print(f"  K={K}: luck-max {luck_ann:.3f} ann | DSR {dsr(result_flip30.returns, K=K):.3f}")
print(f"Realized Sharpe (ann.): {sr_pb * np.sqrt(365):+.3f}")

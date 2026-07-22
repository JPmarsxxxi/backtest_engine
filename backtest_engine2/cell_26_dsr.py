# Cell 26 — DSR deflation for the 30d crypto momentum candidate
import numpy as np
from backtest.selection import dsr, expected_max_sr
from backtest.metrics.core import sharpe_var_term
from backtest.metrics import psr as psr_fn

r = result_best.returns.dropna().to_numpy()
T = len(r)
sr_pb = r.mean() / r.std(ddof=1)
var_term = sharpe_var_term(r, sr_pb)
A = 365 ** 0.5

print(f"Realized Sharpe (ann.): {sr_pb*A:+.3f}   over {T} bars")
print(f"Undeflated PSR(0):      {psr_fn(result_best.returns):.3f}")
print(f"{'K':>4} | luck-max(ann) | DSR")
for K in [1, 9, 20]:
    lm = expected_max_sr(K, var_term / T) * A
    d = dsr(result_best.returns, K=K)
    print(f"{K:>4} | {lm:>+12.3f} | {d:.3f}  {'SIG' if d>0.95 else 'not sig'} at 95%")
print()
print("Note: K=9 treats all 9 horizon trials as independent (over-deflates, since they correlate).")
print("True effective-K is lower -> DSR between the K=1 and K=9 rows.")

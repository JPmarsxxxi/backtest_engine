# Cell 10 — Stage 9 DSR + #013 session verdict.
# Best trial this session = trial 2 (close->open, spread-only engine run, NO swap —
# the most favorable case). K=2 pre-registered trials. DSR on engine bar returns.
from backtest.selection import dsr, expected_max_sr
from backtest.metrics.core import sharpe_var_term

ret2 = result2.returns
ANN = 252 * 45  # ~45 30m bars/day median
r = ret2.dropna().to_numpy()
sr_pb = r.mean() / r.std(ddof=1)
var_term = sharpe_var_term(r, sr_pb)
print(f"Trial 2 engine returns: ann Sharpe {sr_pb * np.sqrt(ANN):+.2f} (spread-only, swap=0)")
for K in [1, 2, 6]:
    print(f"  K={K}: luck-max {expected_max_sr(K, var_term / len(r)) * np.sqrt(ANN):+.3f} ann "
          f"| DSR {dsr(ret2, K=K):.3f}")
print("(K=2 = this session's pre-registered trials; K=6 = #011+#013 combined)")

print("\n#013 VERDICT: KILLED (both trials)")
print("  trial 1 (2-3am window): decayed — only 2020 positive, dead 2021+; gross < costs")
print("  trial 2 (close->open):  gross +2.75 bp/nt real, but 2022+ nets ~0 at swap=0;")
print("                          any positive swap or spread > 1bp half makes it negative")
print("  swap lookup NO LONGER GATES #013 — dead even in the best swap world")

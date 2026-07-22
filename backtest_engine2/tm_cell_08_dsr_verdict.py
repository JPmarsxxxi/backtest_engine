# Cell 8 — #014 DSR (best trial) + verdict.
# Best NET trial = t2 (net-of-financing, -0.17). All trials negative net -> DSR on the
# best is the formal nail. K=3 session trials; also report K=8 (incl #011/#012/#013 leads).
from backtest.selection import dsr, expected_max_sr
from backtest.metrics.core import sharpe_var_term

best = result2.returns.dropna()
r = best.to_numpy()
sr_pb = r.mean() / r.std(ddof=1)
vt = sharpe_var_term(r, sr_pb)
print(f"Best trial (t2 net-of-financing): ann net Sharpe {sr_pb * np.sqrt(252):+.2f}")
for K in [1, 3, 8]:
    print(f"  K={K}: luck-max {expected_max_sr(K, vt / len(r)) * np.sqrt(252):+.3f} ann | "
          f"DSR {dsr(best, K=K):.3f}")

print("\n#014 VERDICT: KILLED (all 3 trials net-negative; gross too weak to clear swap)")
print("  t1 MOP-exact:  net -0.34 | gross +0.31 | swap drag 8.3%/yr")
print("  t2 net-carry:  net -0.17 | gross +0.17 | swap drag 3.9%/yr (best net)")
print("  t3 blend:      net -0.43 | gross +0.20 | swap drag ~6%/yr")
print("  Root cause: UNIVERSE, not costs. Gross trend on FTMO-45 ~+0.2-0.3 Sharpe and a")
print("  NEGATIVE-gross decade 2015-23; the published TSMOM profit lives in bonds/rates +")
print("  commodity breadth that FTMO does not list. Need gross ~0.6 to clear swap+matter.")
print("  Implementation validated (2022 book correctly short EUR/GBP, long JPY, short indices).")

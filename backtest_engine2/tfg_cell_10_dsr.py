# Cell 10 — Stage 9: Deflated Sharpe Ratio (skills/09-selection.md).
# K = 2 distinct configs evaluated OOS this session: full-Kelly (Cells 4-5) and
# vol_only paper-vol (Cells 6, 8-9; the reported one). CPCV paths are NOT trials
# (one trial measured 5 ways); in-train λ grid search is NOT a trial (in-sample).
# Default asymptotic per-trial SR variance (LdP 2014 recipe, no registry yet).
from backtest.selection import dsr, expected_max_sr
from backtest.metrics.core import sharpe_var_term

K = 2

deflated = {c: dsr(mp_res.path_returns[c].dropna(), K=K)
            for c in mp_res.path_returns.columns}

# Luck-max threshold (annualized) per path, for display
rows = []
for c in mp_res.path_returns.columns:
    r = mp_res.path_returns[c].dropna().to_numpy()
    T = len(r)
    sr_pb = r.mean() / r.std(ddof=1)
    var_term = sharpe_var_term(r, sr_pb)
    luck_ann = expected_max_sr(K, var_term / T) * np.sqrt(P["tdays"])
    rows.append({"path": c, "sharpe_ann": sr_pb * np.sqrt(P["tdays"]),
                 "luck_max_ann": luck_ann, "psr": per_path[int(c[-1])].psr,
                 "dsr": deflated[c]})
dsr_df = pd.DataFrame(rows).set_index("path")

print(f"Trials K = {K} (full-Kelly config + vol_only config, this session)")
print(dsr_df.to_string(float_format=lambda v: f"{v:.4f}"))
vals = list(deflated.values())
print(f"\nDSR across {mp_res.n_paths} paths: median={np.median(vals):.4f} "
      f"min={np.min(vals):.4f} max={np.max(vals):.4f}")
print(f"Sanity: DSR < PSR on every path (K>1 deflation): "
      f"{all(dsr_df.loc[c, 'dsr'] < dsr_df.loc[c, 'psr'] for c in dsr_df.index)}")
print(f"Realized Sharpe beats luck-max on every path: "
      f"{bool((dsr_df.sharpe_ann > dsr_df.luck_max_ann).all())}")

import json
NB = r"C:\Users\User\backtest_engine\backtest_engine2\reversal_alpha_ftmo_universe.ipynb"
nb = json.load(open(NB, encoding="utf-8"))
code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]

# Indices (0-based) of cells needed to define result_best + helpers, skipping
# equity sweeps and the Wikipedia sector fetch (cell 15). sector_map is unused by
# the momentum path (neutralize=False), so stub it empty.
NEEDED = [0, 1, 2, 3, 4, 5, 6, 7, 11, 15, 17, 18, 20, 23]  # cells 1,2,3,4,5,6,7,8,12,16,18,19,21,24
g = globals()
g["sector_map"] = {}  # stub before cell 16 (idx 15)

for idx in NEEDED:
    src = "".join(code_cells[idx]["source"])
    try:
        exec(src, g)
        print(f"[cell idx {idx}] OK")
    except Exception as e:
        print(f"[cell idx {idx}] ERROR: {type(e).__name__}: {e}")
        raise

print("---")
print("result_best defined:", "result_best" in g)
print("panel_crypto assets:", len(panel_crypto.assets_all))
# benchmark sanity vs alpha_log (#006: net Sharpe ~1.10)
from backtest.metrics import compute_metrics, sharpe_ratio
_rep = compute_metrics(result_best.returns, result_best.equity_curve,
                       costs=result_best.costs, trades=result_best.trades, ann_factor=365)
print(f"BENCHMARK full-universe 30d momentum  net Sharpe = {_rep.sharpe:+.3f}  (log says ~+1.10)")

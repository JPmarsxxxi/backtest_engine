"""Quick sample replay (last 8760 bars ~1y) to validate trade log + EDA tables."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

nb = json.loads(Path("hurst_optimal_exit_pairs.ipynb").read_text(encoding="utf-8"))
ns = {}
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code" and i <= 2:
        exec("".join(cell["source"]), ns)

panel = ns["panel"]
WARMUP_BARS = 5400
Strategy = ns["HurstOptimalExitPairsStrategy"]

strat = Strategy()
dates = panel.dates[-2000:]  # ~3 months hourly; set -8760 for ~1y
if dates[0] < panel.dates[WARMUP_BARS]:
    dates = panel.dates[WARMUP_BARS:]

print(f"Replay {len(dates)} bars: {dates[0]} -> {dates[-1]}")
t0 = time.time()
for i, t in enumerate(dates):
    strat.generate_weights(panel.as_of(t), t)
elapsed = time.time() - t0
print(f"Done in {elapsed/60:.1f} min | closed={len(strat.trade_log)} open={len(strat.open_trades)}")

df = pd.DataFrame(strat.trade_log)
out = Path("data/hurst_ou_trade_log_sample.parquet")
df.to_parquet(out, index=False)
print(f"Saved {len(df)} trades -> {out}")
if len(df):
    print(df["exit_reason"].value_counts())
    print("win rate", df["win"].mean())
    print(df.groupby((np.floor(df["h_entry"] * 10) / 10))["win"].agg(["count", "mean"]))

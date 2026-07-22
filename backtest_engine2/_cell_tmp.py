# Cell 2 â€” DVOL calibration reference + strategy parameters
# DVOL_PERCENTILE / DVOL_WARMUP_BARS feed directly into the strategy (Cell 3).
# The expanding percentile is computed live inside generate_weights â€” fully PIT-safe.

import numpy as np

# â”€â”€ Strategy parameters â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
DVOL_PERCENTILE  = 65    # regime OFF when current DVOL > P65 of all DVOL seen so far
DVOL_WARMUP_BARS = 720   # 30 days Ã— 24 h â€” gate inactive until this many bars seen

# â”€â”€ Reference: full-sample distribution (context only, NOT used by the strategy) â”€
_vals = dvol_aligned.dropna().values
print("BTC DVOL full-sample distribution (reference only):")
for _p in [10, 25, 50, 65, 75, 85, 90, 95, 99]:
    print(f"  p{_p:2d}: {np.percentile(_vals, _p):.1f}")
print(f"\n  mean : {_vals.mean():.1f}   std : {_vals.std():.1f}")
print(f"  min  : {_vals.min():.1f}   max : {_vals.max():.1f}")

_p65_full = np.percentile(_vals, DVOL_PERCENTILE)
print(f"\nFull-sample p{DVOL_PERCENTILE}: {_p65_full:.1f}  "
      f"(~{(_vals > _p65_full).mean():.0%} of bars would be regime-OFF under this fixed level)")

# â”€â”€ Expanding-percentile trace â€” shows how the threshold evolves PIT-style â”€â”€â”€â”€
_dvol_daily = dvol_aligned.resample("D").last().dropna()
_exp_thresh  = [
    float(np.percentile(_dvol_daily.values[:i+1], DVOL_PERCENTILE))
    for i in range(len(_dvol_daily))
]
import pandas as pd
_trace = pd.Series(_exp_thresh, index=_dvol_daily.index, name="expanding_p65")

print(f"\nExpanding p{DVOL_PERCENTILE} â€” first/last 5 daily values:")
print(_trace.head())
print(_trace.tail())
print(f"\nGate warmup: {DVOL_WARMUP_BARS} bars  "
      f"(â‰ˆ {DVOL_WARMUP_BARS/24:.0f} days â€” regime ON until then regardless of DVOL)")
print(f"\nDVOL_PERCENTILE  = {DVOL_PERCENTILE}")
print(f"DVOL_WARMUP_BARS = {DVOL_WARMUP_BARS}")

"""One-shot runner for the monthly maintenance (invoked by the FX_016_monthly_health task; despite
the _016 name it is the program-wide monthly pass). Sequential; each script writes its own logs:
  edge_health_016.py      — #016 mid-based edge recompute (report only)
  recal_caps_016.py       — #016 entry-cap re-measurement from MT5 ticks
  retrain_overnight_rf.py — #023-v2 annual RF refit (age-gated: acts only when the live artifact
                            is >=350d old; safety gates G1-G3; alpha log 2026-07-17)
"""
import os
import subprocess
import sys

d = os.path.dirname(os.path.abspath(__file__))
rc = 0
for script in ("edge_health_016.py", "recal_caps_016.py", "retrain_overnight_rf.py"):
    r = subprocess.run([sys.executable, os.path.join(d, script)], cwd=d)
    rc = rc or r.returncode
sys.exit(rc)

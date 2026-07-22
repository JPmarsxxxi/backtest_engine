"""Start a persistent Jupyter kernel for trend_following_gold.ipynb.
Run in background; execute cells with:
  python krun.py _tfg_kernel.json trend_following_gold.ipynb <cell_idx>"""
import shutil
import time
from pathlib import Path

from jupyter_client import KernelManager

OUT = Path(__file__).with_name("_tfg_kernel.json")
km = KernelManager(kernel_name="python3")
km.start_kernel()
shutil.copy(km.connection_file, OUT)
print(f"KERNEL_READY {OUT}", flush=True)
try:
    while True:
        time.sleep(3600)
except KeyboardInterrupt:
    km.shutdown_kernel()

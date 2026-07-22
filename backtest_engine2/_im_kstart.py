"""Start a persistent Jupyter kernel for the intraday_mom_us500 session.
Run in background; execute cells against it with _krun.py _im_kernel.json <code.py>."""
import shutil, time
from pathlib import Path
from jupyter_client import KernelManager

OUT = Path(__file__).with_name("_im_kernel.json")
km = KernelManager(kernel_name="python3")
km.start_kernel()
shutil.copy(km.connection_file, OUT)
print(f"KERNEL_READY {OUT}", flush=True)
try:
    while True:
        time.sleep(3600)
except KeyboardInterrupt:
    km.shutdown_kernel()

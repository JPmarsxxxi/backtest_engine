"""Execute a code string against a running Jupyter kernel and print output."""
import sys, re
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
from jupyter_client import BlockingKernelClient

conn_file    = sys.argv[1]
code_file    = sys.argv[2]
msg_timeout  = int(sys.argv[3]) if len(sys.argv) > 3 else 7200   # seconds; default 2h

with open(code_file, "r", encoding="utf-8") as f:
    code = f.read()

kc = BlockingKernelClient()
kc.load_connection_file(conn_file)
kc.start_channels()
kc.wait_for_ready(timeout=60)

kc.execute(code)

while True:
    try:
        msg = kc.get_iopub_msg(timeout=msg_timeout)
    except Exception:
        break
    mt   = msg["msg_type"]
    data = msg.get("content", {})

    if mt == "stream":
        print(data.get("text", ""), end="", flush=True)
    elif mt in ("execute_result", "display_data"):
        txt = data.get("data", {}).get("text/plain", "")
        if txt:
            print(txt, flush=True)
    elif mt == "error":
        print("ERROR:", data.get("ename"), data.get("evalue"), file=sys.stderr)
        for line in data.get("traceback", []):
            print(re.sub(r"\x1b\[[0-9;]*m", "", line), file=sys.stderr)
    elif mt == "status" and data.get("execution_state") == "idle":
        break

kc.stop_channels()

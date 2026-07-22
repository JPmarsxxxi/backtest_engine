"""Connect to a running kernel and listen for output without sending new code."""
import sys, re
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
from jupyter_client import BlockingKernelClient

conn_file   = sys.argv[1]
msg_timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 7200   # seconds

kc = BlockingKernelClient()
kc.load_connection_file(conn_file)
kc.start_channels()
# skip wait_for_ready — kernel may be busy; just listen

while True:
    try:
        msg = kc.get_iopub_msg(timeout=msg_timeout)
    except Exception:
        print("TIMEOUT — no message within", msg_timeout, "s")
        break
    mt   = msg["msg_type"]
    data = msg.get("content", {})

    if mt == "stream":
        print(data.get("text", ""), end="")
    elif mt in ("execute_result", "display_data"):
        txt = data.get("data", {}).get("text/plain", "")
        if txt:
            print(txt)
    elif mt == "error":
        print("ERROR:", data.get("ename"), data.get("evalue"), file=sys.stderr)
        for line in data.get("traceback", []):
            print(re.sub(r"\x1b\[[0-9;]*m", "", line), file=sys.stderr)
    elif mt == "status" and data.get("execution_state") == "idle":
        print("[kernel idle]")
        break

kc.stop_channels()

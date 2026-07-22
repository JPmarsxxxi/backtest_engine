"""Run a single notebook cell against a persistent Jupyter kernel.

Usage: python krun.py <connection_file> <notebook.ipynb> <abs_cell_idx>

Executes the source of cells[idx] in the already-running kernel, writes the
resulting outputs back into the notebook (valid nbformat), and echoes stdout /
results so the caller can inspect them. The kernel stays alive between calls,
so in-memory state persists (no re-running from the top).
"""
import sys
import json
import queue

from jupyter_client import BlockingKernelClient


def main():
    conn, nb_path, idx = sys.argv[1], sys.argv[2], int(sys.argv[3])
    nb = json.load(open(nb_path, encoding="utf-8"))
    cell = nb["cells"][idx]
    if cell.get("cell_type") != "code":
        print(f"cell {idx} is not code", file=sys.stderr)
        sys.exit(2)
    code = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]

    kc = BlockingKernelClient(connection_file=conn)
    kc.load_connection_file()
    kc.start_channels()
    kc.wait_for_ready(timeout=60)

    msg_id = kc.execute(code)
    outputs, status = [], "ok"
    exec_count = None
    while True:
        try:
            msg = kc.get_iopub_msg(timeout=3600)
        except queue.Empty:
            break
        if msg["parent_header"].get("msg_id") != msg_id:
            continue
        mt, c = msg["msg_type"], msg["content"]
        if mt == "stream":
            outputs.append({"output_type": "stream", "name": c.get("name", "stdout"), "text": c["text"]})
            sys.stdout.write(c["text"])
        elif mt in ("execute_result", "display_data"):
            data = c.get("data", {})
            o = {"output_type": mt, "data": data, "metadata": c.get("metadata", {})}
            if mt == "execute_result":
                o["execution_count"] = c.get("execution_count")
            outputs.append(o)
            if "text/plain" in data:
                sys.stdout.write(data["text/plain"] + "\n")
        elif mt == "error":
            outputs.append({"output_type": "error", "ename": c["ename"],
                            "evalue": c["evalue"], "traceback": c["traceback"]})
            sys.stdout.write("\n".join(c["traceback"]) + "\n")
            status = "error"
        elif mt == "execute_input":
            exec_count = c.get("execution_count")
        elif mt == "status" and c["execution_state"] == "idle":
            break

    cell["outputs"] = outputs
    cell["execution_count"] = exec_count
    json.dump(nb, open(nb_path, "w", encoding="utf-8"), indent=1)
    kc.stop_channels()
    print(f"\n[krun] cell {idx} -> {status} (exec_count={exec_count})", file=sys.stderr)
    sys.exit(1 if status == "error" else 0)


if __name__ == "__main__":
    main()

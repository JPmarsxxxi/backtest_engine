"""Run hurst_optimal_exit_pairs.ipynb Cell 1 standalone (for validation)."""
import json
import runpy
from pathlib import Path

nb = json.loads(Path("hurst_optimal_exit_pairs.ipynb").read_text(encoding="utf-8"))
code = nb["cells"][0]["source"]
if isinstance(code, list):
    code = "".join(code)
exec(compile(code, "cell1", "exec"), {"__name__": "__main__"})

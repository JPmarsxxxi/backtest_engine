"""Append a code cell to a notebook (create it if missing).
Usage: python _mm_nb.py <notebook.ipynb> <source.py>"""
import json, sys, uuid
from pathlib import Path

nb_path = Path(sys.argv[1])
src = Path(sys.argv[2]).read_text(encoding="utf-8")

if nb_path.exists():
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
else:
    nb = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3",
          "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}

nb["cells"].append({
    "cell_type": "code", "execution_count": None, "id": uuid.uuid4().hex[:8],
    "metadata": {}, "outputs": [], "source": src.splitlines(keepends=True),
})
nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Appended {sys.argv[2]} -> {nb_path.name}; now {len(nb['cells'])} cell(s).")

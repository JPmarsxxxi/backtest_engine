"""Helper: append a code cell from a .py file to a notebook. Usage: _nb_append.py <nb.ipynb> <src.py>"""
import json
import sys
import uuid
from pathlib import Path

NB_PATH = Path(sys.argv[1])
SRC_PATH = Path(sys.argv[2])

src = SRC_PATH.read_text(encoding="utf-8")

if NB_PATH.exists():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
else:
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

nb["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "id": uuid.uuid4().hex[:8],
    "metadata": {},
    "outputs": [],
    "source": src.splitlines(keepends=True),
})

NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Appended {SRC_PATH.name} -> {NB_PATH.name}; notebook now has {len(nb['cells'])} cells.")

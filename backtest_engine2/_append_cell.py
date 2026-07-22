"""Helper: append a new code cell to hurst_pairs_crypto.ipynb."""
import json
import sys
import uuid
from pathlib import Path

NB_PATH = Path("hurst_pairs_crypto.ipynb")
SRC_PATH = Path(sys.argv[1])

with open(SRC_PATH, "r", encoding="utf-8") as f:
    src = f.read()

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": uuid.uuid4().hex[:8],
    "metadata": {},
    "outputs": [],
    "source": src.splitlines(keepends=True),
}
nb["cells"].append(new_cell)

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Appended cell from {SRC_PATH}. Notebook now has {len(nb['cells'])} cells.")

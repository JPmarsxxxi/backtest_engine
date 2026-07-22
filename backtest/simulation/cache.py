from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np

from backtest.simulation.base import PathGenerator


class PathTensorCache:
    """Disk cache for path tensors keyed by (generator.config, seed, dims).

    Stores .npy files (numpy native, 3-D friendly). Avoids regenerating
    identical paths across MonteCarloEngine runs — useful for reusing the
    same paths when comparing multiple strategies (CRN) or iterating on
    analysis without rerunning generation.
    """

    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(
        self,
        gen_config: dict,
        seed: int,
        n_paths: int,
        n_steps: int,
        n_assets: int,
    ) -> str:
        rep = (
            json.dumps(gen_config, sort_keys=True)
            + f"|{seed}|{n_paths}|{n_steps}|{n_assets}"
        )
        return hashlib.sha256(rep.encode()).hexdigest()[:16]

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.npy"

    def get(
        self,
        generator: PathGenerator,
        seed: int,
        n_paths: int,
        n_steps: int,
        n_assets: int,
    ) -> Optional[np.ndarray]:
        key = self._key(generator.config(), seed, n_paths, n_steps, n_assets)
        p = self._path(key)
        if not p.exists():
            return None
        return np.load(p)

    def put(
        self,
        generator: PathGenerator,
        seed: int,
        n_paths: int,
        n_steps: int,
        n_assets: int,
        tensor: np.ndarray,
    ) -> None:
        key = self._key(generator.config(), seed, n_paths, n_steps, n_assets)
        np.save(self._path(key), tensor)

    def clear(self) -> None:
        for f in self.cache_dir.glob("*.npy"):
            f.unlink()

    def stats(self) -> dict:
        files = list(self.cache_dir.glob("*.npy"))
        total_bytes = sum(f.stat().st_size for f in files)
        return {"count": len(files), "total_bytes": total_bytes}

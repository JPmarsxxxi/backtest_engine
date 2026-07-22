from __future__ import annotations

import numpy as np

from backtest.simulation.adapters import JumpOverlayAdapter
from backtest.simulation.base import PathGenerator
from backtest.simulation.generators import (
    BlockBootstrapGenerator,
    GarchGenerator,
    GaussianGenerator,
    MultivariateGenerator,
    PermutationGenerator,
)


def select_generators(
    fingerprint_scores: dict[str, float],
    real_returns: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, PathGenerator]:
    """Translate fingerprint scores into a generators dict for MonteCarloEngine.

    Always includes gaussian (baseline) + permutation (null). Routes by
    fingerprint axis per MC_DESIGN.md:

        any non-baseline > threshold     → block_bootstrap (preserves all stylized
                                            facts of real returns cheaply)
        jump_diffusion   > threshold     → block_bootstrap_jumps (jump overlay)
        vol_clustering   > threshold     → garch (explicit GJR-GARCH(1,1) vol
                                            dynamics; method-of-moments fit)
        cross_section    > threshold     → multivariate (empirical Σ + shocks;
                                            requires K >= 2 assets)
    """
    if real_returns.ndim != 2:
        raise ValueError(
            f"real_returns must be 2-D (T, K), got {real_returns.shape}"
        )
    K = real_returns.shape[1]
    gens: dict[str, PathGenerator] = {
        "gaussian": GaussianGenerator.fit(real_returns),
        "permutation": PermutationGenerator(real_returns),
    }
    nonbaseline = {k: v for k, v in fingerprint_scores.items() if k != "iid_gaussian"}
    if any(v > threshold for v in nonbaseline.values()):
        gens["block_bootstrap"] = BlockBootstrapGenerator.fit(real_returns)
    if fingerprint_scores.get("jump_diffusion", 0.0) > threshold:
        gens["block_bootstrap_jumps"] = JumpOverlayAdapter(
            BlockBootstrapGenerator.fit(real_returns),
            jump_intensity=0.01,
            jump_sigma=0.03,
        )
    if fingerprint_scores.get("vol_clustering", 0.0) > threshold:
        gens["garch"] = GarchGenerator.fit(real_returns)
    if K >= 2 and fingerprint_scores.get("cross_section", 0.0) > threshold:
        gens["multivariate"] = MultivariateGenerator.fit(real_returns)
    return gens

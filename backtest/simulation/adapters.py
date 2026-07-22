from __future__ import annotations

import numpy as np

from backtest.simulation.base import PathGenerator

# Offset so the adapter's RNG stream doesn't overlap the base generator's draws.
_JUMP_SEED_OFFSET = 999_983


class JumpOverlayAdapter(PathGenerator):
    """Adds Poisson-thinned jumps on top of a base generator's returns.

    On each (path, step), with probability `jump_intensity` a jump is added.
    If correlate_assets=True, a single Gaussian jump value is broadcast across
    all assets on that bar — a market-wide event. If False, each asset draws
    its own independent jump per step.

    Use to stress-test for tail risk beyond what the base generator's sample
    contains (e.g., crash overlays on top of historical-replay paths).
    """

    def __init__(
        self,
        base: PathGenerator,
        jump_intensity: float,
        jump_mu: float = 0.0,
        jump_sigma: float = 0.05,
        correlate_assets: bool = False,
    ):
        if not 0.0 <= jump_intensity <= 1.0:
            raise ValueError(
                f"jump_intensity must be in [0, 1], got {jump_intensity}"
            )
        if jump_sigma < 0:
            raise ValueError(f"jump_sigma must be non-negative, got {jump_sigma}")
        self.base = base
        self.jump_intensity = float(jump_intensity)
        self.jump_mu = float(jump_mu)
        self.jump_sigma = float(jump_sigma)
        self.correlate_assets = bool(correlate_assets)

    def sample(
        self, n_paths: int, n_steps: int, n_assets: int, seed: int
    ) -> np.ndarray:
        base_rets = self.base.sample(n_paths, n_steps, n_assets, seed)
        if self.jump_intensity == 0.0:
            return base_rets
        rng = np.random.default_rng(seed + _JUMP_SEED_OFFSET)
        if self.correlate_assets:
            shape = (n_paths, n_steps, 1)
        else:
            shape = (n_paths, n_steps, n_assets)
        jump_mask = rng.random(shape) < self.jump_intensity
        jump_size = rng.normal(self.jump_mu, self.jump_sigma, shape)
        return base_rets + jump_mask * jump_size

    def config(self) -> dict:
        return {
            "kind": "jump_overlay",
            "base": self.base.config(),
            "jump_intensity": self.jump_intensity,
            "jump_mu": self.jump_mu,
            "jump_sigma": self.jump_sigma,
            "correlate_assets": self.correlate_assets,
        }


class AntitheticAdapter(PathGenerator):
    """Pair each path with its antithetic for ~2x variance reduction.

    Specialized for Gaussian-based bases (must expose .mu attribute).
    For r = mu + sigma*z, antithetic = 2*mu - r. n_paths must be even;
    pairs (2k, 2k+1) are antithetic.
    """

    def __init__(self, base: PathGenerator):
        if not hasattr(base, "mu"):
            raise TypeError(
                "AntitheticAdapter requires base with a .mu attribute "
                f"(typically a Gaussian-based generator); got {type(base).__name__}"
            )
        self.base = base

    def sample(
        self, n_paths: int, n_steps: int, n_assets: int, seed: int
    ) -> np.ndarray:
        if n_paths % 2 != 0:
            raise ValueError(
                f"n_paths must be even for AntitheticAdapter, got {n_paths}"
            )
        half = n_paths // 2
        base_paths = self.base.sample(half, n_steps, n_assets, seed)
        mu = np.asarray(self.base.mu, dtype=np.float64)
        antithetic = 2.0 * mu - base_paths
        out = np.empty((n_paths, n_steps, n_assets), dtype=np.float64)
        out[0::2] = base_paths
        out[1::2] = antithetic
        return out

    def config(self) -> dict:
        return {
            "kind": "antithetic",
            "base": self.base.config(),
        }


_LW_MAX_DELTA = 0.25  # kurtosis well-defined for delta < 0.25


class LambertWTailAdapter(PathGenerator):
    """Goerg's Lambert W x Gaussian heavy-tail wrapper.

    Per asset: standardize base output, apply Y = U * exp(delta/2 * U^2),
    re-standardize empirically, rescale to base mean/std. Delta = 0 is a
    bit-identical passthrough; delta in [0, 0.25) keeps kurtosis finite.
    Composes with any base; tails are added via the marginal transform.
    """

    def __init__(self, base: PathGenerator, delta: float = 0.0):
        if delta < 0:
            raise ValueError("delta must be non-negative")
        if delta >= _LW_MAX_DELTA:
            raise ValueError(
                f"delta must be < {_LW_MAX_DELTA} for finite kurtosis"
            )
        self.base = base
        self.delta = float(delta)

    def sample(
        self, n_paths: int, n_steps: int, n_assets: int, seed: int
    ) -> np.ndarray:
        base_rets = self.base.sample(n_paths, n_steps, n_assets, seed)
        if self.delta == 0.0:
            return base_rets
        out = np.empty_like(base_rets)
        for k in range(n_assets):
            x = base_rets[..., k]
            mu = float(x.mean())
            sigma = float(x.std())
            if sigma == 0.0:
                out[..., k] = x
                continue
            u = (x - mu) / sigma
            y = u * np.exp(0.5 * self.delta * u * u)
            y_std = float(y.std())
            if y_std == 0.0:
                out[..., k] = x
                continue
            y_normalized = (y - y.mean()) / y_std
            out[..., k] = mu + sigma * y_normalized
        return out

    @staticmethod
    def gaussianize(y: np.ndarray, delta: float) -> np.ndarray:
        from scipy.special import lambertw

        if delta == 0.0:
            return np.asarray(y)
        y_arr = np.asarray(y, dtype=np.float64)
        return np.sign(y_arr) * np.sqrt(
            lambertw(delta * y_arr * y_arr).real / delta
        )

    @classmethod
    def fit(
        cls, real_returns: np.ndarray, base: PathGenerator
    ) -> "LambertWTailAdapter":
        """Method-of-moments: solve EK(delta) = excess_kurtosis(real_returns)
        for delta in [0, 0.24] via brentq. Returns delta=0 if real has
        non-positive excess kurtosis."""
        from scipy.optimize import brentq
        from scipy.stats import kurtosis

        ek_target = float(kurtosis(real_returns.flatten(), fisher=True))
        if not np.isfinite(ek_target) or ek_target <= 0:
            return cls(base, delta=0.0)

        def fn(d: float) -> float:
            return (
                3.0 * (1.0 - 2.0 * d) ** 3 / (1.0 - 4.0 * d) ** 2.5 - 3.0 - ek_target
            )

        try:
            delta = float(brentq(fn, 1e-6, 0.24))
        except ValueError:
            delta = 0.0
        return cls(base, delta=delta)

    def config(self) -> dict:
        return {
            "kind": "lambert_w_tail",
            "base": self.base.config(),
            "delta": self.delta,
        }

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from backtest.data import DataPanel
from backtest.simulation.adapters import JumpOverlayAdapter
from backtest.simulation.base import panel_from_returns
from backtest.simulation.generators import GaussianGenerator


@dataclass(frozen=True)
class Probe:
    name: str
    description: str
    panel: DataPanel
    isolates: str


class ProbeBattery:
    """Fixed deterministic synthetic datasets for strategy fingerprinting.

    Each probe isolates one market property. Built from a single seed so
    fingerprints are reproducible across runs and machines.
    """

    def __init__(
        self,
        n_assets: int = 5,
        n_steps: int = 520,
        seed: int = 42,
        init_price: float = 100.0,
    ):
        if n_assets < 2:
            raise ValueError("n_assets must be >= 2")
        if n_steps < 60:
            raise ValueError("n_steps must be >= 60")
        self.n_assets = int(n_assets)
        self.n_steps = int(n_steps)
        self.seed = int(seed)
        self.init_price = float(init_price)
        self._assets = [f"P{i}" for i in range(self.n_assets)]
        self._dates = pd.date_range("2020-01-01", periods=self.n_steps, freq="B")

    def baseline(self) -> Probe:
        return self._iid_gaussian()

    def probes(self) -> List[Probe]:
        return [
            self._iid_gaussian(),
            self._momentum(),
            self._mean_reversion(),
            self._vol_clustering(),
            self._jump_diffusion(),
            self._cross_section(),
        ]

    def battery_id(self) -> str:
        rep = f"v1|{self.n_assets}|{self.n_steps}|{self.seed}|{self.init_price:.4f}"
        return hashlib.sha256(rep.encode()).hexdigest()[:16]

    # --- probe builders --------------------------------------------------

    def _iid_gaussian(self) -> Probe:
        rng = np.random.default_rng(self.seed)
        rets = rng.normal(0.0003, 0.015, (self.n_steps, self.n_assets))
        return self._make("iid_gaussian", "IID N(0.0003, 0.015)", rets, "none")

    def _momentum(self) -> Probe:
        """Persistent per-asset alphas. Cross-sectional momentum profits by
        identifying and holding the structurally-positive alpha assets."""
        rng = np.random.default_rng(self.seed + 11)
        K, T = self.n_assets, self.n_steps
        alphas = np.linspace(-0.0015, 0.0015, K)
        rets = np.tile(alphas, (T, 1)) + rng.normal(0, 0.013, (T, K))
        return self._make(
            "momentum",
            "Persistent per-asset alphas (momentum-friendly)",
            rets, "momentum",
        )

    def _mean_reversion(self) -> Probe:
        """Mean-reverting per-asset log-prices around a common attractor.
        Past winners (now above attractor) have negative expected next return;
        cross-sectional momentum is decisively wrong."""
        rng = np.random.default_rng(self.seed + 12)
        K, T = self.n_assets, self.n_steps
        log_attractor = np.log(100.0)
        log_prices = np.full((T, K), log_attractor)
        sigma_eps = 0.013
        strength = 0.10
        for t in range(1, T):
            log_prices[t] = (
                log_prices[t - 1]
                - strength * (log_prices[t - 1] - log_attractor)
                + rng.normal(0, sigma_eps, K)
            )
        prices = np.exp(log_prices)
        rets = np.zeros((T, K))
        rets[1:] = prices[1:] / prices[:-1] - 1.0
        return self._make(
            "mean_reversion",
            "Mean-reverting log-prices around a common attractor",
            rets, "mean_reversion",
        )

    def _vol_clustering(self) -> Probe:
        rng = np.random.default_rng(self.seed + 100)
        omega = 1e-6
        alpha = 0.10
        beta = 0.85
        sigma0 = np.sqrt(omega / (1.0 - alpha - beta))
        sigma = np.full((self.n_steps, self.n_assets), sigma0)
        eps = rng.standard_normal((self.n_steps, self.n_assets))
        rets = np.zeros((self.n_steps, self.n_assets))
        for t in range(1, self.n_steps):
            sigma[t] = np.sqrt(
                omega
                + alpha * (eps[t - 1] * sigma[t - 1]) ** 2
                + beta * sigma[t - 1] ** 2
            )
            rets[t] = sigma[t] * eps[t]
        return self._make(
            "vol_clustering",
            "GARCH(1,1)-like volatility clustering",
            rets, "vol_clustering",
        )

    def _jump_diffusion(self) -> Probe:
        base = GaussianGenerator(mu=0.0003, sigma=0.010)
        overlay = JumpOverlayAdapter(
            base, jump_intensity=0.02, jump_mu=0.0, jump_sigma=0.05,
            correlate_assets=False,
        )
        rets = overlay.sample(
            n_paths=1, n_steps=self.n_steps, n_assets=self.n_assets,
            seed=self.seed + 200,
        )[0]
        return self._make(
            "jump_diffusion",
            "Gaussian + Poisson jumps (intensity 0.02)",
            rets, "tails",
        )

    def _cross_section(self) -> Probe:
        K = self.n_assets
        rng = np.random.default_rng(self.seed + 300)
        corr = np.full((K, K), 0.2)
        h = K // 2
        corr[:h, :h] = 0.7
        corr[h:, h:] = 0.7
        np.fill_diagonal(corr, 1.0)
        sigma = 0.015
        cov = corr * sigma * sigma
        L = np.linalg.cholesky(cov)
        z = rng.standard_normal((self.n_steps, K))
        rets = z @ L.T
        return self._make(
            "cross_section",
            "Two-block cross-sectional correlation",
            rets, "cross_section",
        )

    def _make(self, name: str, desc: str, rets: np.ndarray, isolates: str) -> Probe:
        panel = panel_from_returns(
            rets, self._assets, self._dates, init_price=self.init_price
        )
        return Probe(name=name, description=desc, panel=panel, isolates=isolates)

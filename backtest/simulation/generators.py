from __future__ import annotations

from typing import Union

import numpy as np

from backtest.simulation.base import PathGenerator

_Scalar = Union[float, np.ndarray]


class GaussianGenerator(PathGenerator):
    """IID Gaussian returns. Baseline / null — preserves μ, σ only.

    mu and sigma may be scalars or per-asset 1-D arrays of length n_assets.
    """

    def __init__(self, mu: _Scalar, sigma: _Scalar):
        self.mu = np.asarray(mu, dtype=np.float64)
        self.sigma = np.asarray(sigma, dtype=np.float64)
        if np.any(self.sigma < 0):
            raise ValueError("sigma must be non-negative")

    def sample(
        self, n_paths: int, n_steps: int, n_assets: int, seed: int
    ) -> np.ndarray:
        self._check_shape(n_assets)
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((n_paths, n_steps, n_assets))
        return z * self.sigma + self.mu

    def _check_shape(self, n_assets: int) -> None:
        for arr, name in ((self.mu, "mu"), (self.sigma, "sigma")):
            if arr.ndim == 0:
                continue
            if arr.shape != (n_assets,):
                raise ValueError(
                    f"{name} shape {arr.shape} incompatible with n_assets={n_assets}"
                )

    def config(self) -> dict:
        return {
            "kind": "gaussian",
            "mu": self.mu.tolist(),
            "sigma": self.sigma.tolist(),
        }

    @classmethod
    def fit(cls, real_returns: np.ndarray) -> "GaussianGenerator":
        if real_returns.ndim != 2:
            raise ValueError(
                f"real_returns must be 2-D (T, K), got {real_returns.shape}"
            )
        return cls(mu=real_returns.mean(axis=0), sigma=real_returns.std(axis=0))


class PermutationGenerator(PathGenerator):
    """Time-shuffled real returns.

    Same per-path permutation across all assets preserves contemporaneous
    correlation; permutation destroys autocorrelation. The look-ahead-leak null:
    any strategy that "works" on permuted returns has a leak.
    """

    def __init__(self, real_returns: np.ndarray):
        if real_returns.ndim != 2:
            raise ValueError(
                f"real_returns must be 2-D (T, K), got {real_returns.shape}"
            )
        self.real = np.asarray(real_returns, dtype=np.float64)

    def sample(
        self, n_paths: int, n_steps: int, n_assets: int, seed: int
    ) -> np.ndarray:
        T, K = self.real.shape
        if K != n_assets:
            raise ValueError(f"real has {K} assets, requested n_assets={n_assets}")
        if n_steps > T:
            raise ValueError(f"n_steps={n_steps} exceeds available history T={T}")
        rng = np.random.default_rng(seed)
        out = np.empty((n_paths, n_steps, K), dtype=np.float64)
        for p in range(n_paths):
            idx = rng.permutation(T)[:n_steps]
            out[p] = self.real[idx]
        return out

    def config(self) -> dict:
        return {
            "kind": "permutation",
            "real_shape": list(self.real.shape),
            "real_checksum": float(self.real.sum()),
        }


class HistoricalReplayGenerator(PathGenerator):
    """Random contiguous slices of real returns. No synthesis — sanity baseline."""

    def __init__(self, real_returns: np.ndarray):
        if real_returns.ndim != 2:
            raise ValueError(
                f"real_returns must be 2-D (T, K), got {real_returns.shape}"
            )
        self.real = np.asarray(real_returns, dtype=np.float64)

    def sample(
        self, n_paths: int, n_steps: int, n_assets: int, seed: int
    ) -> np.ndarray:
        T, K = self.real.shape
        if K != n_assets:
            raise ValueError(f"real has {K} assets, requested n_assets={n_assets}")
        if n_steps > T:
            raise ValueError(f"n_steps={n_steps} exceeds available history T={T}")
        rng = np.random.default_rng(seed)
        out = np.empty((n_paths, n_steps, K), dtype=np.float64)
        max_start = T - n_steps
        for p in range(n_paths):
            start = int(rng.integers(0, max_start + 1))
            out[p] = self.real[start : start + n_steps]
        return out

    def config(self) -> dict:
        return {
            "kind": "historical_replay",
            "real_shape": list(self.real.shape),
            "real_checksum": float(self.real.sum()),
        }


class BlockBootstrapGenerator(PathGenerator):
    """Stationary bootstrap (Politis & Romano 1994) with geometric block lengths.

    At each step, with probability 1/L̄ start a new block at a random index;
    otherwise advance the previous index by 1 (mod T). Wrap-around handles
    the boundary cleanly, so n_steps may exceed T.

    Inherits all stylized facts present in the input returns: heavy tails, vol
    clustering, autocorrelation, cross-section. The practical default for MC.
    """

    def __init__(self, real_returns: np.ndarray, block_length: int):
        if real_returns.ndim != 2:
            raise ValueError(
                f"real_returns must be 2-D (T, K), got {real_returns.shape}"
            )
        if block_length <= 0:
            raise ValueError(f"block_length must be positive, got {block_length}")
        self.real = np.asarray(real_returns, dtype=np.float64)
        self.block_length = int(block_length)

    def sample(
        self, n_paths: int, n_steps: int, n_assets: int, seed: int
    ) -> np.ndarray:
        T, K = self.real.shape
        if K != n_assets:
            raise ValueError(f"real has {K} assets, requested n_assets={n_assets}")
        rng = np.random.default_rng(seed)
        out = np.empty((n_paths, n_steps, K), dtype=np.float64)
        p_new = 1.0 / self.block_length
        for p in range(n_paths):
            new_starts = rng.random(n_steps) < p_new
            new_starts[0] = True  # always start a fresh block at t=0
            random_idx = rng.integers(0, T, size=n_steps)
            idx = np.empty(n_steps, dtype=np.int64)
            cur = 0
            for t in range(n_steps):
                if new_starts[t]:
                    cur = int(random_idx[t])
                else:
                    cur = (cur + 1) % T
                idx[t] = cur
            out[p] = self.real[idx]
        return out

    def config(self) -> dict:
        return {
            "kind": "block_bootstrap",
            "block_length": self.block_length,
            "real_shape": list(self.real.shape),
            "real_checksum": float(self.real.sum()),
        }

    @classmethod
    def fit(cls, real_returns: np.ndarray) -> "BlockBootstrapGenerator":
        """Heuristic block length: 2 × average per-asset first-insignificant
        ACF lag. Politis-White-inspired simplification.
        """
        if real_returns.ndim != 2:
            raise ValueError(
                f"real_returns must be 2-D (T, K), got {real_returns.shape}"
            )
        n, k = real_returns.shape
        threshold = 2.0 / np.sqrt(n)
        max_lag = max(2, min(50, n // 4))
        lags = []
        for col in range(k):
            x = real_returns[:, col].astype(np.float64)
            x = x - x.mean()
            var = float((x * x).mean())
            if var == 0:
                lags.append(1)
                continue
            chosen = max_lag  # fallback if never decorrelates
            for lag in range(1, max_lag + 1):
                acf = float((x[:-lag] * x[lag:]).mean()) / var
                if abs(acf) < threshold:
                    chosen = lag
                    break
            lags.append(chosen)
        block = max(2, int(round(2.0 * float(np.mean(lags)))))
        return cls(real_returns=real_returns, block_length=block)


class SobolGaussianGenerator(GaussianGenerator):
    """Gaussian returns via scrambled Sobol QMC for O((log N)^d / N) ≈ O(1/N)
    convergence (vs O(1/√N) for plain Gaussian).

    Drop-in for GaussianGenerator. Dimension d = n_steps * n_assets per path;
    works well up to d ~ 1000-2000. Best when n_paths is a power of 2 (scipy
    emits a UserWarning otherwise — suppressed locally).
    """

    def __init__(self, mu: _Scalar, sigma: _Scalar, scramble: bool = True):
        super().__init__(mu, sigma)
        self.scramble = bool(scramble)

    def sample(
        self, n_paths: int, n_steps: int, n_assets: int, seed: int
    ) -> np.ndarray:
        import warnings

        from scipy.special import ndtri
        from scipy.stats import qmc

        self._check_shape(n_assets)
        d = n_steps * n_assets
        sobol = qmc.Sobol(d=d, scramble=self.scramble, seed=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            u = sobol.random(n_paths)
        # Avoid 0 and 1 which map to ±inf under the inverse normal CDF.
        u = np.clip(u, 1e-15, 1.0 - 1e-15)
        z = ndtri(u).reshape(n_paths, n_steps, n_assets)
        return z * self.sigma + self.mu

    def config(self) -> dict:
        return {
            "kind": "sobol_gaussian",
            "mu": self.mu.tolist(),
            "sigma": self.sigma.tolist(),
            "scramble": self.scramble,
        }

    @classmethod
    def fit(
        cls, real_returns: np.ndarray, scramble: bool = True
    ) -> "SobolGaussianGenerator":
        if real_returns.ndim != 2:
            raise ValueError(
                f"real_returns must be 2-D (T, K), got {real_returns.shape}"
            )
        return cls(
            mu=real_returns.mean(axis=0),
            sigma=real_returns.std(axis=0),
            scramble=scramble,
        )


class GarchGenerator(PathGenerator):
    """GJR-GARCH(1,1) returns with optional Student-t innovations.

    sigma_t^2 = omega + (alpha + gamma * I[r_{t-1}-mu < 0]) * (r_{t-1}-mu)^2
                      + beta * sigma_{t-1}^2

    Captures vol clustering, leverage effect (gamma > 0), fat tails (Student-t).
    Persistence alpha + beta + gamma/2 must be < 1 for stationarity.
    nu: Student-t df (must be > 2). None = Gaussian innovations.
    """

    def __init__(
        self,
        mu: float = 0.0,
        omega: float = 1e-6,
        alpha: float = 0.05,
        beta: float = 0.90,
        gamma: float = 0.05,
        nu: "float | None" = None,
    ):
        if omega <= 0:
            raise ValueError("omega must be positive")
        if alpha < 0 or beta < 0 or gamma < 0:
            raise ValueError("alpha, beta, gamma must be non-negative")
        persistence = alpha + beta + gamma / 2.0
        if persistence >= 1.0:
            raise ValueError(
                f"alpha + beta + gamma/2 = {persistence:.4f} must be < 1 for stationarity"
            )
        if nu is not None and nu <= 2:
            raise ValueError("nu must be > 2 for finite variance")
        self.mu = float(mu)
        self.omega = float(omega)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.nu = float(nu) if nu is not None else None

    def sample(
        self, n_paths: int, n_steps: int, n_assets: int, seed: int
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        if self.nu is not None:
            # Standardize Student-t to unit variance.
            eps = rng.standard_t(self.nu, (n_paths, n_steps, n_assets)) * np.sqrt(
                (self.nu - 2.0) / self.nu
            )
        else:
            eps = rng.standard_normal((n_paths, n_steps, n_assets))

        unconditional_var = self.omega / (
            1.0 - self.alpha - self.beta - self.gamma / 2.0
        )
        sigma_sq = np.empty((n_paths, n_steps, n_assets), dtype=np.float64)
        rets = np.empty((n_paths, n_steps, n_assets), dtype=np.float64)
        sigma_sq[:, 0, :] = unconditional_var
        rets[:, 0, :] = self.mu + np.sqrt(unconditional_var) * eps[:, 0, :]
        for t in range(1, n_steps):
            r_prev_centered = rets[:, t - 1, :] - self.mu
            indicator = (r_prev_centered < 0).astype(np.float64)
            sigma_sq[:, t, :] = (
                self.omega
                + (self.alpha + self.gamma * indicator) * r_prev_centered ** 2
                + self.beta * sigma_sq[:, t - 1, :]
            )
            rets[:, t, :] = self.mu + np.sqrt(sigma_sq[:, t, :]) * eps[:, t, :]
        return rets

    def config(self) -> dict:
        return {
            "kind": "garch",
            "mu": self.mu,
            "omega": self.omega,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "nu": self.nu,
        }

    @classmethod
    def fit(
        cls,
        real_returns: np.ndarray,
        alpha: float = 0.05,
        beta: float = 0.90,
        gamma: float = 0.05,
        nu: "float | None" = None,
    ) -> "GarchGenerator":
        """Method-of-moments fit. Sets omega and mu from pooled mean/var of
        real_returns; alpha/beta/gamma/nu are inputs (typical equity values).
        """
        if real_returns.ndim != 2:
            raise ValueError(
                f"real_returns must be 2-D (T, K), got {real_returns.shape}"
            )
        persistence = alpha + beta + gamma / 2.0
        if persistence >= 1.0:
            raise ValueError(
                f"alpha + beta + gamma/2 = {persistence:.4f} must be < 1"
            )
        var = float(real_returns.var())
        omega = var * (1.0 - persistence)
        mu = float(real_returns.mean())
        return cls(mu=mu, omega=omega, alpha=alpha, beta=beta, gamma=gamma, nu=nu)


class MultivariateGenerator(PathGenerator):
    """Gaussian returns with explicit covariance via Cholesky factorization.

    Vectorized: factorize Σ once, then z @ L.T + mu samples all paths in one
    matmul. mu (K,) is per-asset mean; cov (K, K) the covariance.
    Shrinkage in [0, 1] applies linear shrinkage toward the diagonal:
    (1 - α) * cov + α * diag(diag(cov)).
    """

    def __init__(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        shrinkage: float = 0.0,
    ):
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError(f"shrinkage must be in [0, 1], got {shrinkage}")
        mu_arr = np.asarray(mu, dtype=np.float64)
        cov_arr = np.asarray(cov, dtype=np.float64)
        if mu_arr.ndim != 1:
            raise ValueError(f"mu must be 1-D (K,), got shape {mu_arr.shape}")
        K = mu_arr.shape[0]
        if cov_arr.shape != (K, K):
            raise ValueError(
                f"cov must be ({K}, {K}) to match mu, got {cov_arr.shape}"
            )
        if shrinkage > 0:
            cov_arr = (1.0 - shrinkage) * cov_arr + shrinkage * np.diag(np.diag(cov_arr))
        self.mu = mu_arr
        self.cov = cov_arr
        self.shrinkage = float(shrinkage)
        self._L = np.linalg.cholesky(cov_arr)

    def sample(
        self, n_paths: int, n_steps: int, n_assets: int, seed: int
    ) -> np.ndarray:
        if n_assets != self.mu.shape[0]:
            raise ValueError(
                f"n_assets={n_assets} doesn't match mu shape {self.mu.shape}"
            )
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((n_paths, n_steps, n_assets))
        return z @ self._L.T + self.mu

    def config(self) -> dict:
        return {
            "kind": "multivariate",
            "mu": self.mu.tolist(),
            "cov": self.cov.tolist(),
            "shrinkage": self.shrinkage,
        }

    @classmethod
    def fit(
        cls, real_returns: np.ndarray, shrinkage: float = 0.0
    ) -> "MultivariateGenerator":
        if real_returns.ndim != 2:
            raise ValueError(
                f"real_returns must be 2-D (T, K), got {real_returns.shape}"
            )
        mu = real_returns.mean(axis=0)
        cov = np.cov(real_returns, rowvar=False)
        return cls(mu=mu, cov=cov, shrinkage=shrinkage)


class SobolMultivariateGenerator(MultivariateGenerator):
    """Multivariate Gaussian via scrambled Sobol QMC.

    Inherits Cholesky factorization from MultivariateGenerator. Standard
    normals come from inverse-transform of scrambled Sobol uniforms instead
    of IID, giving O(1/N) convergence on linear functionals.
    """

    def __init__(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        shrinkage: float = 0.0,
        scramble: bool = True,
    ):
        super().__init__(mu, cov, shrinkage)
        self.scramble = bool(scramble)

    def sample(
        self, n_paths: int, n_steps: int, n_assets: int, seed: int
    ) -> np.ndarray:
        import warnings

        from scipy.special import ndtri
        from scipy.stats import qmc

        if n_assets != self.mu.shape[0]:
            raise ValueError(
                f"n_assets={n_assets} doesn't match mu shape {self.mu.shape}"
            )
        d = n_steps * n_assets
        sobol = qmc.Sobol(d=d, scramble=self.scramble, seed=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            u = sobol.random(n_paths)
        u = np.clip(u, 1e-15, 1.0 - 1e-15)
        z = ndtri(u).reshape(n_paths, n_steps, n_assets)
        return z @ self._L.T + self.mu

    def config(self) -> dict:
        cfg = super().config()
        cfg["kind"] = "sobol_multivariate"
        cfg["scramble"] = self.scramble
        return cfg

    @classmethod
    def fit(
        cls,
        real_returns: np.ndarray,
        shrinkage: float = 0.0,
        scramble: bool = True,
    ) -> "SobolMultivariateGenerator":
        if real_returns.ndim != 2:
            raise ValueError(
                f"real_returns must be 2-D (T, K), got {real_returns.shape}"
            )
        mu = real_returns.mean(axis=0)
        cov = np.cov(real_returns, rowvar=False)
        return cls(mu=mu, cov=cov, shrinkage=shrinkage, scramble=scramble)

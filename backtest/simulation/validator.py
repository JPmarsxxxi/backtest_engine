from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import chi2, kurtosis, ks_2samp, skew

from backtest.simulation.base import PathGenerator


# --- Threshold + ValidationResult ----------------------------------------

@dataclass(frozen=True)
class Threshold:
    """Closed-interval bound. Either side may be None to leave open."""
    lower: Optional[float] = None
    upper: Optional[float] = None

    def passes(self, value: float) -> bool:
        if not np.isfinite(value):
            return False
        if self.lower is not None and value < self.lower:
            return False
        if self.upper is not None and value > self.upper:
            return False
        return True


@dataclass
class ValidationResult:
    scores: dict[str, float]
    thresholds: dict[str, Threshold]
    passed: dict[str, bool]
    overall_passed: bool
    generator_config: dict
    n_paths: int
    n_steps: int
    real_returns: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    synth_returns: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    def report(self) -> pd.DataFrame:
        rows = []
        for name, score in self.scores.items():
            t = self.thresholds.get(name)
            rows.append({
                "metric": name,
                "score": score,
                "lower": t.lower if t is not None else None,
                "upper": t.upper if t is not None else None,
                "passed": self.passed[name],
            })
        return pd.DataFrame(rows).set_index("metric")

    def plot(self, figsize: tuple = (14, 8), max_lag: int = 20):
        """6-panel matplotlib tearsheet: QQ, marginal histogram overlay,
        ACF of returns, ACF of |returns| (vol clustering), correlation
        eigenvalue spectrum (K>=2), and pass/fail score table.
        Requires real_returns and synth_returns on the result.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "matplotlib is required for plot(). "
                "Install with: pip install backtest[viz]"
            ) from e

        if self.real_returns is None or self.synth_returns is None:
            raise ValueError(
                "plot() requires real_returns and synth_returns on the "
                "result; ensure validate() populated them."
            )

        real = self.real_returns
        synth = self.synth_returns
        K = real.shape[1]
        synth_pooled = synth.reshape(-1, K)

        fig, axes = plt.subplots(2, 3, figsize=figsize)

        # 1. QQ plot (pooled across all assets)
        ax = axes[0, 0]
        n = min(len(real.flatten()), len(synth_pooled.flatten()), 1000)
        qs = np.linspace(0.01, 0.99, n)
        real_q = np.quantile(real.flatten(), qs)
        synth_q = np.quantile(synth_pooled.flatten(), qs)
        ax.scatter(real_q, synth_q, s=8, alpha=0.5)
        lims = [min(real_q.min(), synth_q.min()), max(real_q.max(), synth_q.max())]
        ax.plot(lims, lims, "k--", lw=0.8)
        ax.set_xlabel("real quantile")
        ax.set_ylabel("synth quantile")
        ax.set_title("QQ plot")

        # 2. Marginal histogram overlay
        ax = axes[0, 1]
        ax.hist(real.flatten(), bins=80, alpha=0.5, label="real", density=True)
        ax.hist(
            synth_pooled.flatten(), bins=80, alpha=0.5,
            label="synth", density=True,
        )
        ax.set_xlabel("returns")
        ax.set_ylabel("density")
        ax.set_title("Marginal distribution")
        ax.legend(loc="best", frameon=False)

        # 3. ACF of returns (pooled)
        ax = axes[0, 2]
        real_pool = real.flatten()
        synth_pool = synth_pooled.flatten()
        real_acf = _acf_lags(real_pool, max_lag)
        synth_acf = _acf_lags(synth_pool, max_lag)
        lags = np.arange(1, max_lag + 1)
        ax.bar(lags - 0.2, real_acf, 0.4, label="real", alpha=0.7)
        ax.bar(lags + 0.2, synth_acf, 0.4, label="synth", alpha=0.7)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xlabel("lag")
        ax.set_ylabel("ACF")
        ax.set_title("ACF of returns")
        ax.legend(loc="best", frameon=False)

        # 4. ACF of |returns| (vol clustering)
        ax = axes[1, 0]
        real_abs_acf = _acf_lags(np.abs(real_pool), max_lag)
        synth_abs_acf = _acf_lags(np.abs(synth_pool), max_lag)
        ax.bar(lags - 0.2, real_abs_acf, 0.4, label="real", alpha=0.7)
        ax.bar(lags + 0.2, synth_abs_acf, 0.4, label="synth", alpha=0.7)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xlabel("lag")
        ax.set_ylabel("ACF |r|")
        ax.set_title("Volatility clustering")
        ax.legend(loc="best", frameon=False)

        # 5. Eigenvalue spectrum
        ax = axes[1, 1]
        if K >= 2:
            real_corr = np.corrcoef(real.T)
            synth_corr = np.corrcoef(synth_pooled.T)
            real_eig = np.sort(np.linalg.eigvalsh(real_corr))[::-1]
            synth_eig = np.sort(np.linalg.eigvalsh(synth_corr))[::-1]
            x = np.arange(1, K + 1)
            ax.plot(x, real_eig, "o-", label="real", lw=1.2)
            ax.plot(x, synth_eig, "s-", label="synth", lw=1.2)
            ax.set_xlabel("rank")
            ax.set_ylabel("eigenvalue")
            ax.set_title("Correlation eigenvalue spectrum")
            ax.legend(loc="best", frameon=False)
        else:
            ax.text(
                0.5, 0.5, "univariate (no Σ spectrum)",
                ha="center", va="center", transform=ax.transAxes,
            )
            ax.set_title("Correlation eigenvalue spectrum")
            ax.axis("off")

        # 6. Pass/fail score table
        ax = axes[1, 2]
        ax.axis("off")
        table_data = []
        cell_colours = []
        for m, s in self.scores.items():
            p = self.passed.get(m, True)
            table_data.append([m, f"{s:.4f}", "PASS" if p else "FAIL"])
            c = "#a3e4a3" if p else "#f4a3a3"
            cell_colours.append(["white", "white", c])
        table = ax.table(
            cellText=table_data,
            colLabels=["metric", "score", "result"],
            cellLoc="left",
            loc="center",
            cellColours=cell_colours,
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        ax.set_title(f"Validation: {'PASS' if self.overall_passed else 'FAIL'}")

        fig.tight_layout()
        return fig


# --- default thresholds (Cont 2001 stylized facts) -----------------------

DEFAULT_THRESHOLDS: dict[str, Threshold] = {
    "mean_z":               Threshold(upper=0.1),
    "std_ratio":            Threshold(lower=0.9, upper=1.1),
    "skew_diff":            Threshold(lower=-0.3, upper=0.3),
    "kurt_ratio":           Threshold(lower=0.5, upper=1.5),
    "ks_pvalue":            Threshold(lower=0.05),
    "hill_tail_diff":       Threshold(lower=-0.5, upper=0.5),
    "acf_r_lag1_diff":      Threshold(upper=0.05),
    "acf_abs_r_lag1_diff":  Threshold(upper=0.10),
    "ljung_box_r2_pvalue":  Threshold(upper=0.01),
    "corr_frobenius_ratio": Threshold(upper=0.15),
}


# --- numerical helpers ---------------------------------------------------

def _hill_index(x: np.ndarray, top_pct: float = 0.05) -> float:
    """Hill estimator of tail index (1/decay rate). Higher = lighter tails."""
    abs_x = np.abs(np.asarray(x, dtype=np.float64))
    abs_x = abs_x[abs_x > 0]
    if len(abs_x) < 20:
        return float("nan")
    abs_x = np.sort(abs_x)[::-1]
    k = max(2, int(top_pct * len(abs_x)))
    if k >= len(abs_x):
        return float("nan")
    threshold = abs_x[k]
    if threshold <= 0:
        return float("nan")
    log_ratio = np.log(abs_x[:k] / threshold)
    m = float(log_ratio.mean())
    if m <= 0:
        return float("nan")
    return 1.0 / m


def _acf_lag1(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    var = float((x * x).mean())
    if var <= 0:
        return 0.0
    return float((x[:-1] * x[1:]).mean()) / var


def _acf_lags(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Sample ACF at lags 1..max_lag for a 1-D array."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    var = float((x * x).mean())
    if var <= 0:
        return np.zeros(max_lag)
    out = np.empty(max_lag)
    for k in range(1, max_lag + 1):
        out[k - 1] = float((x[:-k] * x[k:]).mean()) / var
    return out


def _ljung_box_pvalue(x: np.ndarray, max_lag: int = 10) -> float:
    """Q-stat p-value under H0 of no serial correlation up to max_lag."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n <= max_lag + 1:
        return float("nan")
    x = x - x.mean()
    var = float((x * x).mean())
    if var <= 0:
        return 1.0
    Q = 0.0
    for k in range(1, max_lag + 1):
        cov = float((x[:-k] * x[k:]).mean())
        rho_k = cov / var
        Q += (rho_k * rho_k) / (n - k)
    Q *= n * (n + 2)
    return float(1.0 - chi2.cdf(Q, max_lag))


# --- validator -----------------------------------------------------------

class GeneratorValidator:
    """Scores a generator's output against real reference returns and a
    stylized-facts battery. Generators with scores outside threshold bounds
    fail the gate (overall_passed=False).
    """

    def __init__(
        self,
        real_returns: np.ndarray,
        n_paths: int = 20,
        n_steps: Optional[int] = None,
        seed: int = 0,
        thresholds: Optional[dict[str, Threshold]] = None,
    ):
        if real_returns.ndim != 2:
            raise ValueError(
                f"real_returns must be 2-D (T, K), got {real_returns.shape}"
            )
        if n_paths <= 0:
            raise ValueError("n_paths must be positive")
        self.real = np.asarray(real_returns, dtype=np.float64)
        self.n_paths = int(n_paths)
        self.n_steps = int(n_steps) if n_steps is not None else len(self.real)
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive")
        self.seed = int(seed)
        self.thresholds = (
            dict(thresholds) if thresholds is not None else dict(DEFAULT_THRESHOLDS)
        )

    def validate(self, generator: PathGenerator) -> ValidationResult:
        K = self.real.shape[1]
        synth = generator.sample(
            n_paths=self.n_paths,
            n_steps=self.n_steps,
            n_assets=K,
            seed=self.seed,
        )
        if synth.shape != (self.n_paths, self.n_steps, K):
            raise ValueError(
                f"generator returned shape {synth.shape}, "
                f"expected ({self.n_paths}, {self.n_steps}, {K})"
            )
        scores = self._compute_scores(synth)
        passed = {
            m: (self.thresholds[m].passes(s) if m in self.thresholds else True)
            for m, s in scores.items()
        }
        return ValidationResult(
            scores=scores,
            thresholds=dict(self.thresholds),
            passed=passed,
            overall_passed=all(passed.values()),
            generator_config=generator.config(),
            n_paths=self.n_paths,
            n_steps=self.n_steps,
            real_returns=self.real,
            synth_returns=synth,
        )

    def _compute_scores(self, synth: np.ndarray) -> dict[str, float]:
        K = self.real.shape[1]
        synth_pooled = synth.reshape(-1, K)  # (N*T, K) for marginal metrics

        return {
            "mean_z":               self._mean_z(synth_pooled),
            "std_ratio":            self._std_ratio(synth_pooled),
            "skew_diff":            self._skew_diff(synth_pooled),
            "kurt_ratio":           self._kurt_ratio(synth_pooled),
            "ks_pvalue":            self._ks_pvalue(synth_pooled),
            "hill_tail_diff":       self._hill_tail_diff(synth_pooled),
            "acf_r_lag1_diff":      self._acf_lag1_diff(synth, abs_returns=False),
            "acf_abs_r_lag1_diff":  self._acf_lag1_diff(synth, abs_returns=True),
            "ljung_box_r2_pvalue":  self._ljung_box_r2_pvalue(synth),
            "corr_frobenius_ratio": self._corr_frobenius_ratio(synth),
        }

    def _mean_z(self, synth_pooled: np.ndarray) -> float:
        real_mean = self.real.mean(axis=0)
        synth_mean = synth_pooled.mean(axis=0)
        real_std = self.real.std(axis=0)
        denom = np.where(real_std > 0, real_std, 1.0)
        return float(np.mean(np.abs(synth_mean - real_mean) / denom))

    def _std_ratio(self, synth_pooled: np.ndarray) -> float:
        real_std = self.real.std(axis=0)
        synth_std = synth_pooled.std(axis=0)
        denom = np.where(real_std > 0, real_std, 1.0)
        return float(np.mean(synth_std / denom))

    def _skew_diff(self, synth_pooled: np.ndarray) -> float:
        return float(np.mean(skew(synth_pooled, axis=0) - skew(self.real, axis=0)))

    def _kurt_ratio(self, synth_pooled: np.ndarray) -> float:
        synth_k = kurtosis(synth_pooled, axis=0, fisher=True) + 3.0
        real_k = kurtosis(self.real, axis=0, fisher=True) + 3.0
        denom = np.where(real_k > 0, real_k, 1.0)
        return float(np.mean(synth_k / denom))

    def _ks_pvalue(self, synth_pooled: np.ndarray) -> float:
        K = self.real.shape[1]
        ps = []
        for k in range(K):
            _, p = ks_2samp(synth_pooled[:, k], self.real[:, k])
            ps.append(p)
        return float(np.mean(ps))

    def _hill_tail_diff(self, synth_pooled: np.ndarray) -> float:
        K = self.real.shape[1]
        diffs = []
        for k in range(K):
            sh = _hill_index(synth_pooled[:, k])
            rh = _hill_index(self.real[:, k])
            if np.isfinite(sh) and np.isfinite(rh):
                diffs.append(sh - rh)
        if not diffs:
            return float("nan")
        return float(np.mean(diffs))

    def _acf_lag1_diff(self, synth: np.ndarray, abs_returns: bool) -> float:
        K = self.real.shape[1]
        N = synth.shape[0]
        real_acf = np.array([
            _acf_lag1(np.abs(self.real[:, k]) if abs_returns else self.real[:, k])
            for k in range(K)
        ])
        synth_acf = np.zeros(K)
        for k in range(K):
            per_path = [
                _acf_lag1(np.abs(synth[p, :, k]) if abs_returns else synth[p, :, k])
                for p in range(N)
            ]
            synth_acf[k] = float(np.mean(per_path))
        return float(np.mean(np.abs(synth_acf - real_acf)))

    def _ljung_box_r2_pvalue(self, synth: np.ndarray, max_lag: int = 10) -> float:
        K = self.real.shape[1]
        N = synth.shape[0]
        ps = []
        for p in range(N):
            for k in range(K):
                ps.append(_ljung_box_pvalue(synth[p, :, k] ** 2, max_lag))
        ps = [v for v in ps if np.isfinite(v)]
        if not ps:
            return float("nan")
        return float(np.mean(ps))

    def _corr_frobenius_ratio(self, synth: np.ndarray) -> float:
        K = self.real.shape[1]
        if K < 2:
            return 0.0
        real_corr = np.corrcoef(self.real.T)
        N = synth.shape[0]
        synth_corrs = np.empty((N, K, K))
        for p in range(N):
            synth_corrs[p] = np.corrcoef(synth[p].T)
        synth_corr = synth_corrs.mean(axis=0)
        diff = float(np.linalg.norm(synth_corr - real_corr, ord="fro"))
        denom = float(np.linalg.norm(real_corr, ord="fro"))
        if denom <= 0:
            return float("nan")
        return diff / denom

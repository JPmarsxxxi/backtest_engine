from __future__ import annotations


def _check_alpha(alpha: float) -> None:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")


def _check_pvalue(p: float) -> None:
    if not 0 <= p <= 1:
        raise ValueError("p must be in [0, 1]")


def _check_k(K: int) -> None:
    if K < 1:
        raise ValueError("K must be >= 1")


def sidak_alpha(alpha: float, K: int) -> float:
    """Per-trial significance threshold for FWER alpha across K independent trials."""
    _check_alpha(alpha)
    _check_k(K)
    return 1.0 - (1.0 - alpha) ** (1.0 / K)


def sidak_pvalue(p: float, K: int) -> float:
    """Sidak-corrected p-value for K trials."""
    _check_pvalue(p)
    _check_k(K)
    return 1.0 - (1.0 - p) ** K


def bonferroni_alpha(alpha: float, K: int) -> float:
    """Per-trial threshold under Bonferroni: alpha / K."""
    _check_alpha(alpha)
    _check_k(K)
    return alpha / K


def bonferroni_pvalue(p: float, K: int) -> float:
    """Bonferroni-corrected p-value, capped at 1."""
    _check_pvalue(p)
    _check_k(K)
    return min(p * K, 1.0)

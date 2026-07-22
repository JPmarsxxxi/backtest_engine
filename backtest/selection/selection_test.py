import numpy as np
import pandas as pd
import pytest

from backtest.metrics import psr, sharpe_ratio
from backtest.selection import (
    bonferroni_alpha,
    bonferroni_pvalue,
    dsr,
    effective_k,
    expected_max_sr,
    sidak_alpha,
    sidak_pvalue,
)


def _normal_returns(seed=0, n=2000, mu=0.0005, sigma=0.01):
    rng = np.random.default_rng(seed)
    return rng.normal(mu, sigma, n)


def test_sidak_alpha_k1_unchanged():
    assert sidak_alpha(0.05, K=1) == pytest.approx(0.05)


def test_sidak_alpha_decreases_with_k():
    a1 = sidak_alpha(0.05, K=1)
    a10 = sidak_alpha(0.05, K=10)
    a100 = sidak_alpha(0.05, K=100)
    assert a1 > a10 > a100


def test_sidak_pvalue_roundtrip():
    p = sidak_alpha(0.05, K=20)
    assert sidak_pvalue(p, K=20) == pytest.approx(0.05)


def test_sidak_pvalue_at_zero():
    assert sidak_pvalue(0.0, K=10) == 0.0


def test_bonferroni_alpha():
    assert bonferroni_alpha(0.05, K=10) == pytest.approx(0.005)


def test_bonferroni_pvalue_capped():
    assert bonferroni_pvalue(0.5, K=10) == 1.0
    assert bonferroni_pvalue(0.01, K=10) == pytest.approx(0.10)


def test_corrections_validation():
    with pytest.raises(ValueError):
        sidak_alpha(0.0, K=10)
    with pytest.raises(ValueError):
        sidak_alpha(1.0, K=10)
    with pytest.raises(ValueError):
        sidak_alpha(0.05, K=0)
    with pytest.raises(ValueError):
        sidak_pvalue(-0.1, K=10)
    with pytest.raises(ValueError):
        sidak_pvalue(1.1, K=10)
    with pytest.raises(ValueError):
        bonferroni_alpha(0.05, K=0)
    with pytest.raises(ValueError):
        bonferroni_pvalue(0.5, K=0)


def test_expected_max_sr_k1_zero():
    assert expected_max_sr(K=1, var_sr=0.001) == 0.0


def test_expected_max_sr_zero_variance_zero():
    assert expected_max_sr(K=10, var_sr=0.0) == 0.0


def test_expected_max_sr_monotonic_in_k():
    var = 0.001
    s2 = expected_max_sr(K=2, var_sr=var)
    s10 = expected_max_sr(K=10, var_sr=var)
    s100 = expected_max_sr(K=100, var_sr=var)
    assert s2 < s10 < s100


def test_expected_max_sr_monotonic_in_var():
    K = 20
    s_low = expected_max_sr(K, var_sr=0.0001)
    s_high = expected_max_sr(K, var_sr=0.01)
    assert s_low < s_high


def test_expected_max_sr_validation():
    with pytest.raises(ValueError):
        expected_max_sr(K=0, var_sr=0.001)
    with pytest.raises(ValueError):
        expected_max_sr(K=10, var_sr=-0.001)


def test_dsr_in_unit_interval():
    r = _normal_returns(mu=0.0008, n=2000, seed=1)
    val = dsr(r, K=10)
    assert 0.0 <= val <= 1.0


def test_dsr_k1_matches_psr_zero():
    r = _normal_returns(mu=0.0005, n=2000, seed=2)
    assert dsr(r, K=1) == pytest.approx(psr(r, sr_star=0.0), rel=1e-9)


def test_dsr_decreases_with_k():
    r = _normal_returns(mu=0.0008, n=2000, seed=3)
    d1 = dsr(r, K=1)
    d20 = dsr(r, K=20)
    d100 = dsr(r, K=100)
    assert d1 > d20 > d100


def test_dsr_increases_with_realized_sr():
    weak = _normal_returns(mu=0.0002, n=3000, seed=4)
    strong = _normal_returns(mu=0.0015, n=3000, seed=4)
    assert dsr(strong, K=20) > dsr(weak, K=20)


def test_dsr_with_sample_sr_variance():
    r = _normal_returns(mu=0.001, n=2000, seed=5)
    # Pretend we observed multiple trials with this variance
    val = dsr(r, K=20, sr_variance=0.001)
    assert 0.0 <= val <= 1.0


def test_dsr_validation():
    r = _normal_returns(n=500)
    with pytest.raises(ValueError):
        dsr(r, K=0)


def test_effective_k_identical_returns():
    rng = np.random.default_rng(10)
    base = rng.normal(0, 0.01, 500)
    df = pd.DataFrame({f"t{i}": base for i in range(10)})
    assert effective_k(df, threshold=0.5) == 1


def test_effective_k_uncorrelated_returns():
    rng = np.random.default_rng(11)
    df = pd.DataFrame(
        {f"t{i}": rng.normal(0, 0.01, 1000) for i in range(15)}
    )
    k_eff = effective_k(df, threshold=0.5)
    # Random series at threshold=0.5 should mostly be in their own clusters
    assert k_eff >= 10


def test_effective_k_two_clear_clusters():
    rng = np.random.default_rng(12)
    a = rng.normal(0, 0.01, 1000)
    b = rng.normal(0, 0.01, 1000)
    cluster_a = pd.DataFrame({f"a{i}": a + rng.normal(0, 0.0005, 1000) for i in range(5)})
    cluster_b = pd.DataFrame({f"b{i}": b + rng.normal(0, 0.0005, 1000) for i in range(5)})
    df = pd.concat([cluster_a, cluster_b], axis=1)
    assert effective_k(df, threshold=0.5) == 2


def test_effective_k_single_column():
    df = pd.DataFrame({"only": _normal_returns(n=500)})
    assert effective_k(df) == 1


def test_effective_k_empty():
    df = pd.DataFrame()
    assert effective_k(df) == 0


def test_effective_k_threshold_validation():
    df = pd.DataFrame({"a": _normal_returns(n=200), "b": _normal_returns(n=200, seed=1)})
    with pytest.raises(ValueError):
        effective_k(df, threshold=-0.1)
    with pytest.raises(ValueError):
        effective_k(df, threshold=1.5)

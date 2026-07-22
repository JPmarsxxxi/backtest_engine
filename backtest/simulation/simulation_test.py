import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kurtosis

from backtest.engine import Engine
from backtest.risk import RiskConfig
from backtest.simulation import (
    AntitheticAdapter,
    BlockBootstrapGenerator,
    GarchGenerator,
    GaussianGenerator,
    HistoricalReplayGenerator,
    JumpOverlayAdapter,
    LambertWTailAdapter,
    MultivariateGenerator,
    PathGenerator,
    PermutationGenerator,
    SobolGaussianGenerator,
    SobolMultivariateGenerator,
    panel_from_returns,
)
from backtest.strategy import Strategy


# --- shared fixtures -----------------------------------------------------

@pytest.fixture
def real_returns():
    rng = np.random.default_rng(42)
    return rng.normal(0.0003, 0.015, (1260, 5))


@pytest.fixture
def assets():
    return ["A", "B", "C", "D", "E"]


@pytest.fixture
def dates():
    return pd.date_range("2020-01-01", periods=520, freq="B")


# --- shared shape / dtype / reproducibility battery ----------------------

GENERATOR_CTORS = [
    pytest.param(lambda r: GaussianGenerator.fit(r), id="gaussian"),
    pytest.param(lambda r: PermutationGenerator(r), id="permutation"),
    pytest.param(lambda r: HistoricalReplayGenerator(r), id="historical"),
    pytest.param(lambda r: BlockBootstrapGenerator.fit(r), id="block_bootstrap"),
    pytest.param(lambda r: SobolGaussianGenerator.fit(r), id="sobol_gaussian"),
    pytest.param(lambda r: GarchGenerator.fit(r), id="garch"),
    pytest.param(
        lambda r: LambertWTailAdapter(GaussianGenerator.fit(r), delta=0.15),
        id="lambert_w",
    ),
    pytest.param(lambda r: MultivariateGenerator.fit(r), id="multivariate"),
    pytest.param(
        lambda r: SobolMultivariateGenerator.fit(r), id="sobol_multivariate"
    ),
]


@pytest.mark.parametrize("ctor", GENERATOR_CTORS)
def test_generator_shape_and_dtype(ctor, real_returns):
    gen: PathGenerator = ctor(real_returns)
    out = gen.sample(n_paths=4, n_steps=200, n_assets=5, seed=0)
    assert out.shape == (4, 200, 5)
    assert out.dtype == np.float64


@pytest.mark.parametrize("ctor", GENERATOR_CTORS)
def test_generator_reproducibility(ctor, real_returns):
    gen: PathGenerator = ctor(real_returns)
    a = gen.sample(n_paths=4, n_steps=100, n_assets=5, seed=7)
    b = gen.sample(n_paths=4, n_steps=100, n_assets=5, seed=7)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("ctor", GENERATOR_CTORS)
def test_generator_seeds_differ(ctor, real_returns):
    gen: PathGenerator = ctor(real_returns)
    a = gen.sample(n_paths=4, n_steps=50, n_assets=5, seed=1)
    b = gen.sample(n_paths=4, n_steps=50, n_assets=5, seed=2)
    assert not np.array_equal(a, b)


@pytest.mark.parametrize("ctor", GENERATOR_CTORS)
def test_config_json_serializable(ctor, real_returns):
    gen: PathGenerator = ctor(real_returns)
    cfg = gen.config()
    assert "kind" in cfg
    json.dumps(cfg)


# --- Gaussian-specific ---------------------------------------------------

def test_gaussian_moments_match_fit(real_returns):
    gen = GaussianGenerator.fit(real_returns)
    out = gen.sample(n_paths=200, n_steps=500, n_assets=5, seed=0)
    sample_mu = out.mean(axis=(0, 1))
    sample_sigma = out.std(axis=(0, 1))
    np.testing.assert_allclose(sample_mu, gen.mu, atol=5e-4)
    np.testing.assert_allclose(sample_sigma, gen.sigma, rtol=0.05)


def test_gaussian_scalar_args_broadcast():
    gen = GaussianGenerator(mu=0.0, sigma=0.01)
    out = gen.sample(n_paths=3, n_steps=50, n_assets=4, seed=0)
    assert out.shape == (3, 50, 4)


def test_gaussian_rejects_negative_sigma():
    with pytest.raises(ValueError, match="non-negative"):
        GaussianGenerator(mu=0.0, sigma=-0.01)


def test_gaussian_rejects_mismatched_per_asset():
    gen = GaussianGenerator(mu=np.zeros(3), sigma=np.ones(3))
    with pytest.raises(ValueError, match="incompatible"):
        gen.sample(n_paths=1, n_steps=10, n_assets=5, seed=0)


# --- Permutation-specific ------------------------------------------------

def test_permutation_destroys_autocorrelation():
    rng = np.random.default_rng(0)
    eps = rng.normal(0, 0.01, (5000, 1))
    ar = np.empty_like(eps)
    ar[0] = eps[0]
    for k in range(1, len(eps)):
        ar[k] = 0.7 * ar[k - 1] + eps[k]
    real_acf1 = np.corrcoef(ar[:-1, 0], ar[1:, 0])[0, 1]
    assert real_acf1 > 0.5

    gen = PermutationGenerator(ar)
    out = gen.sample(n_paths=1, n_steps=5000, n_assets=1, seed=0)[0]
    synth_acf1 = np.corrcoef(out[:-1, 0], out[1:, 0])[0, 1]
    assert abs(synth_acf1) < 0.05


def test_permutation_preserves_marginal_moments(real_returns):
    gen = PermutationGenerator(real_returns)
    out = gen.sample(n_paths=20, n_steps=1260, n_assets=5, seed=0)
    np.testing.assert_allclose(
        out.mean(axis=(0, 1)), real_returns.mean(axis=0), atol=5e-5
    )
    np.testing.assert_allclose(
        out.std(axis=(0, 1)), real_returns.std(axis=0), rtol=0.02
    )


def test_permutation_preserves_contemporaneous_correlation():
    rng = np.random.default_rng(0)
    L = np.array([[1.0, 0.0, 0.0], [0.7, 0.7, 0.0], [0.5, 0.0, 0.87]])
    z = rng.standard_normal((1260, 3))
    correlated = z @ L.T * 0.01
    gen = PermutationGenerator(correlated)
    out = gen.sample(n_paths=1, n_steps=1260, n_assets=3, seed=0)[0]
    np.testing.assert_allclose(np.corrcoef(out.T), np.corrcoef(correlated.T), atol=0.05)


# --- HistoricalReplay-specific -------------------------------------------

def test_historical_replay_returns_real_rows(real_returns):
    gen = HistoricalReplayGenerator(real_returns)
    out = gen.sample(n_paths=5, n_steps=100, n_assets=5, seed=0)
    real_set = {tuple(r) for r in real_returns}
    for path in out:
        for row in path:
            assert tuple(row) in real_set


def test_historical_replay_contiguous(real_returns):
    gen = HistoricalReplayGenerator(real_returns)
    out = gen.sample(n_paths=3, n_steps=50, n_assets=5, seed=0)
    T = len(real_returns)
    for path in out:
        found = False
        for start in range(T - 50 + 1):
            if np.array_equal(path, real_returns[start : start + 50]):
                found = True
                break
        assert found, "path is not a contiguous slice of real"


def test_historical_replay_rejects_too_many_steps(real_returns):
    gen = HistoricalReplayGenerator(real_returns)
    with pytest.raises(ValueError, match="exceeds available history"):
        gen.sample(n_paths=1, n_steps=2000, n_assets=5, seed=0)


# --- BlockBootstrap-specific ---------------------------------------------

def _ar1(rho: float, n: int, sigma: float = 0.01, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, sigma, (n, 1))
    ar = np.empty_like(eps)
    ar[0] = eps[0]
    for k in range(1, n):
        ar[k] = rho * ar[k - 1] + eps[k]
    return ar


def test_block_bootstrap_preserves_marginal_moments(real_returns):
    gen = BlockBootstrapGenerator(real_returns, block_length=20)
    out = gen.sample(n_paths=20, n_steps=1260, n_assets=5, seed=0)
    np.testing.assert_allclose(
        out.mean(axis=(0, 1)), real_returns.mean(axis=0), atol=5e-4
    )
    np.testing.assert_allclose(
        out.std(axis=(0, 1)), real_returns.std(axis=0), rtol=0.05
    )


def test_block_bootstrap_preserves_autocorrelation():
    ar = _ar1(rho=0.7, n=5000)
    real_acf1 = np.corrcoef(ar[:-1, 0], ar[1:, 0])[0, 1]
    assert real_acf1 > 0.5

    gen = BlockBootstrapGenerator(ar, block_length=20)
    out = gen.sample(n_paths=1, n_steps=5000, n_assets=1, seed=0)[0]
    synth_acf1 = np.corrcoef(out[:-1, 0], out[1:, 0])[0, 1]
    assert synth_acf1 > 0.3


def test_block_bootstrap_block_length_one_destroys_autocorrelation():
    ar = _ar1(rho=0.7, n=5000)
    gen = BlockBootstrapGenerator(ar, block_length=1)
    out = gen.sample(n_paths=1, n_steps=5000, n_assets=1, seed=0)[0]
    synth_acf1 = np.corrcoef(out[:-1, 0], out[1:, 0])[0, 1]
    assert abs(synth_acf1) < 0.05


def test_block_bootstrap_preserves_cross_section():
    rng = np.random.default_rng(0)
    L = np.array([[1.0, 0.0, 0.0], [0.7, 0.7, 0.0], [0.5, 0.0, 0.87]])
    z = rng.standard_normal((1260, 3))
    correlated = z @ L.T * 0.01
    gen = BlockBootstrapGenerator(correlated, block_length=10)
    out = gen.sample(n_paths=1, n_steps=2000, n_assets=3, seed=0)[0]
    np.testing.assert_allclose(
        np.corrcoef(out.T), np.corrcoef(correlated.T), atol=0.05
    )


def test_block_bootstrap_fit_iid(real_returns):
    gen = BlockBootstrapGenerator.fit(real_returns)
    assert 2 <= gen.block_length <= 6


def test_block_bootstrap_fit_autocorrelated():
    ar = _ar1(rho=0.7, n=1260)
    gen = BlockBootstrapGenerator.fit(ar)
    assert gen.block_length > 5


def test_block_bootstrap_rejects_invalid_block_length(real_returns):
    with pytest.raises(ValueError, match="positive"):
        BlockBootstrapGenerator(real_returns, block_length=0)
    with pytest.raises(ValueError, match="positive"):
        BlockBootstrapGenerator(real_returns, block_length=-1)


def test_block_bootstrap_handles_n_steps_greater_than_T(real_returns):
    T = len(real_returns)
    gen = BlockBootstrapGenerator(real_returns, block_length=20)
    out = gen.sample(n_paths=2, n_steps=T * 2, n_assets=5, seed=0)
    assert out.shape == (2, T * 2, 5)
    assert np.isfinite(out).all()


# --- SobolGaussian-specific ----------------------------------------------

def test_sobol_gaussian_low_variance_vs_iid():
    """Sobol QMC sample-mean estimator should have lower trial-to-trial variance
    than plain IID Gaussian at the same N."""
    mu, sigma = 0.0003, 0.015
    iid_gen = GaussianGenerator(mu, sigma)
    qmc_gen = SobolGaussianGenerator(mu, sigma)
    n_trials = 30
    iid_means = []
    qmc_means = []
    for trial in range(n_trials):
        iid_out = iid_gen.sample(n_paths=64, n_steps=200, n_assets=1, seed=trial)
        qmc_out = qmc_gen.sample(n_paths=64, n_steps=200, n_assets=1, seed=trial)
        iid_means.append(iid_out.mean())
        qmc_means.append(qmc_out.mean())
    iid_var = float(np.var(iid_means, ddof=1))
    qmc_var = float(np.var(qmc_means, ddof=1))
    assert qmc_var < iid_var, (
        f"Sobol var {qmc_var:.3e} not less than IID var {iid_var:.3e}"
    )


def test_sobol_gaussian_unscrambled_is_deterministic():
    gen = SobolGaussianGenerator(0.0, 0.01, scramble=False)
    a = gen.sample(n_paths=8, n_steps=100, n_assets=1, seed=0)
    b = gen.sample(n_paths=8, n_steps=100, n_assets=1, seed=999)
    np.testing.assert_array_equal(a, b)


def test_sobol_gaussian_scrambled_depends_on_seed():
    gen = SobolGaussianGenerator(0.0, 0.01, scramble=True)
    a = gen.sample(n_paths=8, n_steps=100, n_assets=1, seed=0)
    b = gen.sample(n_paths=8, n_steps=100, n_assets=1, seed=1)
    assert not np.array_equal(a, b)


def test_sobol_gaussian_fit_matches_moments(real_returns):
    gen = SobolGaussianGenerator.fit(real_returns)
    out = gen.sample(n_paths=64, n_steps=500, n_assets=5, seed=0)
    np.testing.assert_allclose(
        out.mean(axis=(0, 1)), real_returns.mean(axis=0), atol=5e-4
    )
    np.testing.assert_allclose(
        out.std(axis=(0, 1)), real_returns.std(axis=0), rtol=0.10
    )


def test_sobol_gaussian_config_includes_scramble():
    gen = SobolGaussianGenerator(0.0, 0.01, scramble=False)
    cfg = gen.config()
    assert cfg["kind"] == "sobol_gaussian"
    assert cfg["scramble"] is False


# --- JumpOverlayAdapter --------------------------------------------------

def test_jump_overlay_zero_intensity_is_passthrough():
    base = GaussianGenerator(mu=0.0, sigma=0.01)
    overlay = JumpOverlayAdapter(base, jump_intensity=0.0)
    a = base.sample(n_paths=2, n_steps=100, n_assets=5, seed=0)
    b = overlay.sample(n_paths=2, n_steps=100, n_assets=5, seed=0)
    np.testing.assert_array_equal(a, b)


def test_jump_overlay_increases_kurtosis():
    base = GaussianGenerator(mu=0.0, sigma=0.01)
    overlay = JumpOverlayAdapter(base, jump_intensity=0.05, jump_sigma=0.05)
    base_out = base.sample(n_paths=20, n_steps=2000, n_assets=1, seed=0).flatten()
    over_out = overlay.sample(n_paths=20, n_steps=2000, n_assets=1, seed=0).flatten()
    base_kurt = float(kurtosis(base_out, fisher=True))
    over_kurt = float(kurtosis(over_out, fisher=True))
    assert over_kurt > base_kurt + 1.0


def test_jump_overlay_correlate_assets_aligns_jump_times():
    base = GaussianGenerator(mu=0.0, sigma=0.0)
    overlay = JumpOverlayAdapter(
        base,
        jump_intensity=0.1,
        jump_mu=0.0,
        jump_sigma=0.05,
        correlate_assets=True,
    )
    out = overlay.sample(n_paths=1, n_steps=200, n_assets=5, seed=0)[0]
    for t in range(out.shape[0]):
        if (out[t] != 0.0).any():
            assert (out[t] != 0.0).all()
            assert np.allclose(out[t], out[t, 0])


def test_jump_overlay_independent_assets_have_independent_jumps():
    base = GaussianGenerator(mu=0.0, sigma=0.0)
    overlay = JumpOverlayAdapter(
        base,
        jump_intensity=0.5,
        correlate_assets=False,
    )
    out = overlay.sample(n_paths=1, n_steps=500, n_assets=5, seed=0)[0]
    nz = (out != 0.0).sum(axis=1)
    assert ((nz > 0) & (nz < 5)).any()


def test_jump_overlay_reproducibility():
    base = GaussianGenerator(mu=0.0, sigma=0.01)
    overlay = JumpOverlayAdapter(base, jump_intensity=0.05)
    a = overlay.sample(n_paths=3, n_steps=100, n_assets=5, seed=7)
    b = overlay.sample(n_paths=3, n_steps=100, n_assets=5, seed=7)
    np.testing.assert_array_equal(a, b)


def test_jump_overlay_config_includes_base():
    base = GaussianGenerator(mu=0.0, sigma=0.01)
    overlay = JumpOverlayAdapter(base, jump_intensity=0.05)
    cfg = overlay.config()
    assert cfg["kind"] == "jump_overlay"
    assert cfg["base"]["kind"] == "gaussian"
    json.dumps(cfg)


def test_jump_overlay_composes_with_block_bootstrap(real_returns):
    base = BlockBootstrapGenerator(real_returns, block_length=20)
    overlay = JumpOverlayAdapter(base, jump_intensity=0.01, jump_sigma=0.03)
    out = overlay.sample(n_paths=2, n_steps=500, n_assets=5, seed=0)
    assert out.shape == (2, 500, 5)
    assert np.isfinite(out).all()


def test_jump_overlay_validation():
    base = GaussianGenerator(mu=0.0, sigma=0.01)
    with pytest.raises(ValueError, match="jump_intensity"):
        JumpOverlayAdapter(base, jump_intensity=-0.1)
    with pytest.raises(ValueError, match="jump_intensity"):
        JumpOverlayAdapter(base, jump_intensity=1.5)
    with pytest.raises(ValueError, match="jump_sigma"):
        JumpOverlayAdapter(base, jump_intensity=0.05, jump_sigma=-0.01)


# --- AntitheticAdapter ---------------------------------------------------

def test_antithetic_pairs_are_exact():
    base = GaussianGenerator(mu=0.001, sigma=0.02)
    adapter = AntitheticAdapter(base)
    out = adapter.sample(n_paths=10, n_steps=50, n_assets=3, seed=0)
    mu = float(base.mu)
    for k in range(0, 10, 2):
        np.testing.assert_allclose(out[k + 1], 2 * mu - out[k])


def test_antithetic_zero_mu_is_negation():
    base = GaussianGenerator(mu=0.0, sigma=0.02)
    adapter = AntitheticAdapter(base)
    out = adapter.sample(n_paths=4, n_steps=20, n_assets=2, seed=0)
    np.testing.assert_array_equal(out[1], -out[0])
    np.testing.assert_array_equal(out[3], -out[2])


def test_antithetic_per_asset_mu():
    mu = np.array([0.001, -0.002, 0.0])
    sigma = np.array([0.01, 0.02, 0.015])
    base = GaussianGenerator(mu=mu, sigma=sigma)
    adapter = AntitheticAdapter(base)
    out = adapter.sample(n_paths=4, n_steps=20, n_assets=3, seed=0)
    for k in range(0, 4, 2):
        np.testing.assert_allclose(out[k + 1], 2 * mu - out[k])


def test_antithetic_rejects_odd_n_paths():
    base = GaussianGenerator(mu=0.0, sigma=0.01)
    adapter = AntitheticAdapter(base)
    with pytest.raises(ValueError, match="even"):
        adapter.sample(n_paths=3, n_steps=10, n_assets=2, seed=0)


def test_antithetic_rejects_base_without_mu(real_returns):
    base = BlockBootstrapGenerator(real_returns, block_length=10)
    with pytest.raises(TypeError, match=".mu"):
        AntitheticAdapter(base)


def test_antithetic_composes_with_sobol():
    base = SobolGaussianGenerator(mu=0.0, sigma=0.01)
    adapter = AntitheticAdapter(base)
    out = adapter.sample(n_paths=8, n_steps=50, n_assets=2, seed=0)
    assert out.shape == (8, 50, 2)
    for k in range(0, 8, 2):
        np.testing.assert_array_equal(out[k + 1], -out[k])


def test_antithetic_reproducibility():
    base = GaussianGenerator(mu=0.0, sigma=0.01)
    adapter = AntitheticAdapter(base)
    a = adapter.sample(n_paths=4, n_steps=20, n_assets=2, seed=7)
    b = adapter.sample(n_paths=4, n_steps=20, n_assets=2, seed=7)
    np.testing.assert_array_equal(a, b)


def test_antithetic_config_includes_base():
    base = GaussianGenerator(mu=0.0, sigma=0.01)
    adapter = AntitheticAdapter(base)
    cfg = adapter.config()
    assert cfg["kind"] == "antithetic"
    assert cfg["base"]["kind"] == "gaussian"
    json.dumps(cfg)


# --- panel_from_returns + engine integration -----------------------------

def test_panel_from_returns_shape(assets, dates):
    rng = np.random.default_rng(0)
    rets = rng.normal(0, 0.01, (len(dates), len(assets)))
    panel = panel_from_returns(rets, assets, dates)
    assert len(panel.dates) == len(dates)
    assert list(panel.assets_all) == assets
    assert panel._prices.iloc[0, 0] == pytest.approx(100.0 * (1 + rets[0, 0]))


def test_panel_from_returns_validates_shapes(assets, dates):
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="must be 2-D"):
        panel_from_returns(rng.normal(0, 0.01, (10,)), assets, dates)
    with pytest.raises(ValueError, match="steps"):
        panel_from_returns(rng.normal(0, 0.01, (10, 5)), assets, dates)
    with pytest.raises(ValueError, match="assets"):
        panel_from_returns(rng.normal(0, 0.01, (len(dates), 3)), assets, dates)


class _Mom(Strategy):
    rebalance_frequency = "monthly"
    risk = RiskConfig(max_position=0.30, max_gross=1.0, max_net=1.0)

    def __init__(self, lookback: int = 60):
        self.lookback = lookback

    def __repr__(self) -> str:
        return f"_Mom(lookback={self.lookback})"

    def generate_weights(self, data, t):
        if len(data.prices) < self.lookback + 1:
            return pd.Series(0.0, index=data.assets)
        rets = data.prices.iloc[-1] / data.prices.iloc[-self.lookback - 1] - 1
        rets = rets.reindex(data.assets).dropna()
        if len(rets) < 2:
            return pd.Series(0.0, index=data.assets)
        median = rets.median()
        n = len(rets)
        w = pd.Series(0.0, index=rets.index)
        w[rets > median] = 1.0 / n
        w[rets < median] = -1.0 / n
        return w


def test_engine_runs_on_synthetic_panel(assets, dates):
    gen = GaussianGenerator(mu=0.0003, sigma=0.015)
    paths = gen.sample(
        n_paths=1, n_steps=len(dates), n_assets=len(assets), seed=0
    )
    panel = panel_from_returns(paths[0], assets, dates)
    result = Engine(panel, _Mom(lookback=60)).run(start=panel.dates[120])
    assert len(result.equity_curve) > 0
    assert np.isfinite(result.equity_curve.iloc[-1])


# --- GarchGenerator-specific --------------------------------------------

def test_garch_rejects_non_stationary():
    with pytest.raises(ValueError, match="stationarity"):
        GarchGenerator(alpha=0.5, beta=0.5)


def test_garch_rejects_bad_omega():
    with pytest.raises(ValueError, match="omega"):
        GarchGenerator(omega=-1)


def test_garch_rejects_bad_nu():
    with pytest.raises(ValueError, match="nu"):
        GarchGenerator(nu=2.0)


def test_garch_fit_matches_variance(real_returns):
    gen = GarchGenerator.fit(real_returns)
    out = gen.sample(n_paths=20, n_steps=2000, n_assets=5, seed=0)
    real_var = float(real_returns.var())
    synth_var = float(out.var())
    assert 0.7 * real_var < synth_var < 1.3 * real_var


def test_garch_vol_clustering():
    gen = GarchGenerator(omega=1e-6, alpha=0.10, beta=0.85, gamma=0.0)
    out = gen.sample(n_paths=1, n_steps=5000, n_assets=1, seed=0)[0, :, 0]
    abs_r = np.abs(out)
    acf1 = np.corrcoef(abs_r[:-1], abs_r[1:])[0, 1]
    assert acf1 > 0.1


def test_garch_leverage_effect():
    # E[|r_t| | r_{t-1}<0] should exceed E[|r_t| | r_{t-1}>0] under gamma>0.
    # Theoretical ratio for these params is ~1.11; assert > 1.05 for safety.
    gen = GarchGenerator(omega=1e-6, alpha=0.05, beta=0.80, gamma=0.20)
    out = gen.sample(n_paths=20, n_steps=3000, n_assets=1, seed=0)
    flat_r = out[:, :-1, 0].flatten()
    flat_next_abs = np.abs(out[:, 1:, 0]).flatten()
    after_neg = flat_next_abs[flat_r < 0].mean()
    after_pos = flat_next_abs[flat_r > 0].mean()
    assert after_neg > after_pos * 1.05


def test_garch_student_t_increases_kurtosis():
    gauss_gen = GarchGenerator(omega=1e-6, alpha=0.05, beta=0.90)
    student_gen = GarchGenerator(omega=1e-6, alpha=0.05, beta=0.90, nu=4.0)
    g_out = gauss_gen.sample(n_paths=10, n_steps=2000, n_assets=1, seed=0).flatten()
    s_out = student_gen.sample(n_paths=10, n_steps=2000, n_assets=1, seed=0).flatten()
    g_kurt = float(kurtosis(g_out, fisher=True))
    s_kurt = float(kurtosis(s_out, fisher=True))
    assert s_kurt > g_kurt + 1.0


def test_garch_fit_rejects_bad_inputs():
    with pytest.raises(ValueError, match="2-D"):
        GarchGenerator.fit(np.zeros((10,)))
    with pytest.raises(ValueError, match="< 1"):
        GarchGenerator.fit(np.zeros((10, 2)), alpha=0.5, beta=0.5)


# --- LambertWTailAdapter ------------------------------------------------

def test_lambert_w_zero_delta_passthrough():
    base = GaussianGenerator(mu=0.001, sigma=0.02)
    adapter = LambertWTailAdapter(base, delta=0.0)
    a = base.sample(n_paths=4, n_steps=100, n_assets=3, seed=0)
    b = adapter.sample(n_paths=4, n_steps=100, n_assets=3, seed=0)
    np.testing.assert_array_equal(a, b)


def test_lambert_w_increases_kurtosis():
    base = GaussianGenerator(mu=0.0, sigma=0.01)
    base_out = base.sample(n_paths=20, n_steps=2000, n_assets=1, seed=0).flatten()
    base_kurt = float(kurtosis(base_out, fisher=True))
    adapter = LambertWTailAdapter(base, delta=0.15)
    adapter_out = adapter.sample(n_paths=20, n_steps=2000, n_assets=1, seed=0).flatten()
    adapter_kurt = float(kurtosis(adapter_out, fisher=True))
    assert adapter_kurt > base_kurt + 1.0


def test_lambert_w_rejects_bad_delta():
    base = GaussianGenerator(mu=0.0, sigma=0.01)
    with pytest.raises(ValueError, match="non-negative"):
        LambertWTailAdapter(base, delta=-0.1)
    with pytest.raises(ValueError, match="finite kurtosis"):
        LambertWTailAdapter(base, delta=0.25)


def test_lambert_w_preserves_mean_and_std():
    base = GaussianGenerator(mu=0.001, sigma=0.02)
    adapter = LambertWTailAdapter(base, delta=0.15)
    out = adapter.sample(n_paths=20, n_steps=1000, n_assets=3, seed=0)
    for k in range(3):
        np.testing.assert_allclose(out[..., k].mean(), 0.001, atol=5e-4)
        np.testing.assert_allclose(out[..., k].std(), 0.02, rtol=0.10)


def test_lambert_w_gaussianize_inverts():
    rng = np.random.default_rng(0)
    u = rng.standard_normal(10000)
    delta = 0.15
    y = u * np.exp(0.5 * delta * u * u)
    u_recovered = LambertWTailAdapter.gaussianize(y, delta)
    np.testing.assert_allclose(u_recovered, u, atol=1e-8)


def test_lambert_w_composes_with_sobol():
    base = SobolGaussianGenerator(mu=0.0, sigma=0.01)
    adapter = LambertWTailAdapter(base, delta=0.15)
    out = adapter.sample(n_paths=8, n_steps=100, n_assets=2, seed=0)
    assert out.shape == (8, 100, 2)
    assert np.isfinite(out).all()


def test_lambert_w_fit_produces_valid_delta(real_returns):
    base = GaussianGenerator.fit(real_returns)
    adapter = LambertWTailAdapter.fit(real_returns, base)
    assert 0 <= adapter.delta < 0.25


def test_lambert_w_fit_zero_for_negative_kurtosis():
    rng = np.random.default_rng(0)
    light_tail = rng.uniform(-1, 1, (10000, 1))
    base = GaussianGenerator(mu=0.0, sigma=1.0)
    adapter = LambertWTailAdapter.fit(light_tail, base)
    assert adapter.delta == 0.0


def test_lambert_w_config_includes_base():
    base = GaussianGenerator(mu=0.0, sigma=0.01)
    adapter = LambertWTailAdapter(base, delta=0.10)
    cfg = adapter.config()
    assert cfg["kind"] == "lambert_w_tail"
    assert cfg["base"]["kind"] == "gaussian"
    assert cfg["delta"] == 0.10
    json.dumps(cfg)


# --- MultivariateGenerator ----------------------------------------------

def test_multivariate_recovers_cov():
    K = 5
    cov_true = np.full((K, K), 0.6 * 0.015 ** 2)
    np.fill_diagonal(cov_true, 0.015 ** 2)
    mu_true = np.zeros(K)
    gen = MultivariateGenerator(mu=mu_true, cov=cov_true)
    out = gen.sample(n_paths=20, n_steps=2000, n_assets=K, seed=0)
    flat = out.reshape(-1, K)
    cov_emp = np.cov(flat, rowvar=False)
    np.testing.assert_allclose(cov_emp, cov_true, rtol=0.10)


def test_multivariate_recovers_mean():
    K = 4
    mu_true = np.array([0.001, 0.002, 0.0, -0.001])
    cov_true = np.eye(K) * 0.01 ** 2
    gen = MultivariateGenerator(mu=mu_true, cov=cov_true)
    out = gen.sample(n_paths=30, n_steps=1500, n_assets=K, seed=0)
    flat = out.reshape(-1, K)
    np.testing.assert_allclose(flat.mean(axis=0), mu_true, atol=5e-4)


def test_multivariate_rejects_bad_shapes():
    with pytest.raises(ValueError, match="mu must be"):
        MultivariateGenerator(mu=np.zeros((3, 3)), cov=np.eye(3))
    with pytest.raises(ValueError, match="cov must be"):
        MultivariateGenerator(mu=np.zeros(3), cov=np.eye(4))


def test_multivariate_rejects_non_psd_cov():
    bad_cov = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalue -1 -> not PSD
    with pytest.raises(np.linalg.LinAlgError):
        MultivariateGenerator(mu=np.zeros(2), cov=bad_cov)


def test_multivariate_rejects_bad_shrinkage():
    with pytest.raises(ValueError, match="shrinkage"):
        MultivariateGenerator(mu=np.zeros(2), cov=np.eye(2), shrinkage=-0.1)
    with pytest.raises(ValueError, match="shrinkage"):
        MultivariateGenerator(mu=np.zeros(2), cov=np.eye(2), shrinkage=1.5)


def test_multivariate_full_shrinkage_diagonalizes():
    K = 4
    cov = np.full((K, K), 0.5)
    np.fill_diagonal(cov, 1.0)
    gen = MultivariateGenerator(mu=np.zeros(K), cov=cov, shrinkage=1.0)
    out = gen.sample(n_paths=20, n_steps=2000, n_assets=K, seed=0)
    flat = out.reshape(-1, K)
    cov_emp = np.cov(flat, rowvar=False)
    off_diag = cov_emp - np.diag(np.diag(cov_emp))
    assert np.abs(off_diag).max() < 0.05


def test_multivariate_fit_recovers_input(real_returns):
    gen = MultivariateGenerator.fit(real_returns)
    np.testing.assert_allclose(gen.mu, real_returns.mean(axis=0))
    cov_real = np.cov(real_returns, rowvar=False)
    np.testing.assert_allclose(gen.cov, cov_real)


def test_multivariate_sample_n_assets_mismatch_raises():
    K = 3
    gen = MultivariateGenerator(mu=np.zeros(K), cov=np.eye(K))
    with pytest.raises(ValueError, match="n_assets"):
        gen.sample(n_paths=2, n_steps=10, n_assets=5, seed=0)


# --- SobolMultivariateGenerator -----------------------------------------

def test_sobol_multivariate_recovers_cov():
    K = 5
    cov_true = np.full((K, K), 0.6 * 0.015 ** 2)
    np.fill_diagonal(cov_true, 0.015 ** 2)
    mu_true = np.zeros(K)
    gen = SobolMultivariateGenerator(mu=mu_true, cov=cov_true)
    out = gen.sample(n_paths=64, n_steps=500, n_assets=K, seed=0)
    flat = out.reshape(-1, K)
    cov_emp = np.cov(flat, rowvar=False)
    np.testing.assert_allclose(cov_emp, cov_true, rtol=0.15)


def test_sobol_multivariate_unscrambled_is_deterministic():
    K = 3
    cov = np.eye(K) * 0.01 ** 2
    gen = SobolMultivariateGenerator(
        mu=np.zeros(K), cov=cov, scramble=False
    )
    a = gen.sample(n_paths=8, n_steps=100, n_assets=K, seed=0)
    b = gen.sample(n_paths=8, n_steps=100, n_assets=K, seed=999)
    np.testing.assert_array_equal(a, b)


def test_sobol_multivariate_config_includes_scramble():
    K = 2
    gen = SobolMultivariateGenerator(
        mu=np.zeros(K), cov=np.eye(K), scramble=False
    )
    cfg = gen.config()
    assert cfg["kind"] == "sobol_multivariate"
    assert cfg["scramble"] is False


def test_sobol_multivariate_composes_with_lambert_w():
    K = 3
    cov = np.eye(K) * 0.01 ** 2
    base = SobolMultivariateGenerator(mu=np.zeros(K), cov=cov)
    adapter = LambertWTailAdapter(base, delta=0.15)
    out = adapter.sample(n_paths=8, n_steps=100, n_assets=K, seed=0)
    assert out.shape == (8, 100, K)
    assert np.isfinite(out).all()

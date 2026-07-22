import numpy as np
import pandas as pd
import pytest

from backtest.simulation import (
    BlockBootstrapGenerator,
    GaussianGenerator,
    HistoricalReplayGenerator,
    PermutationGenerator,
)
from backtest.simulation.validator import (
    DEFAULT_THRESHOLDS,
    GeneratorValidator,
    Threshold,
    ValidationResult,
)


# --- fixtures ------------------------------------------------------------

@pytest.fixture
def garch_real():
    """5-asset GARCH(1,1)-like returns: vol clustering present, no autocorr in r."""
    rng = np.random.default_rng(0)
    K = 5
    T = 1500
    omega = 1e-6
    alpha = 0.10
    beta = 0.85
    sigma = np.full((T, K), np.sqrt(omega / (1 - alpha - beta)))
    eps = rng.standard_normal((T, K))
    rets = np.zeros((T, K))
    for t in range(1, T):
        sigma[t] = np.sqrt(
            omega
            + alpha * (eps[t - 1] * sigma[t - 1]) ** 2
            + beta * sigma[t - 1] ** 2
        )
        rets[t] = sigma[t] * eps[t]
    return rets


@pytest.fixture
def iid_real():
    rng = np.random.default_rng(1)
    return rng.normal(0.0003, 0.015, (1500, 5))


# --- Threshold -----------------------------------------------------------

def test_threshold_two_sided():
    t = Threshold(lower=0.9, upper=1.1)
    assert t.passes(1.0)
    assert t.passes(0.9)
    assert t.passes(1.1)
    assert not t.passes(0.89)
    assert not t.passes(1.11)


def test_threshold_one_sided_upper():
    t = Threshold(upper=0.05)
    assert t.passes(0.0)
    assert t.passes(-100.0)
    assert not t.passes(0.06)


def test_threshold_one_sided_lower():
    t = Threshold(lower=0.05)
    assert t.passes(0.05)
    assert t.passes(100.0)
    assert not t.passes(0.04)


def test_threshold_rejects_nan():
    assert not Threshold(upper=1.0).passes(float("nan"))
    assert not Threshold(lower=0.0).passes(float("nan"))


def test_threshold_open():
    assert Threshold().passes(1e9)
    assert Threshold().passes(-1e9)
    assert not Threshold().passes(float("nan"))


# --- validator smoke -----------------------------------------------------

def test_validator_runs_on_each_generator(iid_real):
    val = GeneratorValidator(iid_real, n_paths=5, seed=0)
    for gen in [
        GaussianGenerator.fit(iid_real),
        PermutationGenerator(iid_real),
        HistoricalReplayGenerator(iid_real),
        BlockBootstrapGenerator.fit(iid_real),
    ]:
        result = val.validate(gen)
        assert isinstance(result, ValidationResult)
        assert set(result.scores.keys()) == set(DEFAULT_THRESHOLDS.keys())
        for name in result.scores:
            assert name in result.passed
            assert isinstance(result.passed[name], bool)


def test_report_dataframe_shape(iid_real):
    val = GeneratorValidator(iid_real, n_paths=3, seed=0)
    result = val.validate(GaussianGenerator.fit(iid_real))
    df = result.report()
    assert isinstance(df, pd.DataFrame)
    assert df.index.name == "metric"
    assert set(df.columns) == {"score", "lower", "upper", "passed"}
    assert len(df) == len(DEFAULT_THRESHOLDS)


def test_validator_reproducibility(iid_real):
    val_a = GeneratorValidator(iid_real, n_paths=5, seed=42)
    val_b = GeneratorValidator(iid_real, n_paths=5, seed=42)
    gen = GaussianGenerator.fit(iid_real)
    a = val_a.validate(gen)
    b = val_b.validate(gen)
    for m in a.scores:
        assert a.scores[m] == b.scores[m] or (
            np.isnan(a.scores[m]) and np.isnan(b.scores[m])
        )


# --- semantic gates ------------------------------------------------------

def test_historical_replay_passes_marginals_and_vol_clustering(garch_real):
    val = GeneratorValidator(garch_real, n_paths=10, seed=0)
    result = val.validate(HistoricalReplayGenerator(garch_real))
    # HistoricalReplay returns slices of real -> all stylized facts inherited.
    for m in (
        "mean_z", "std_ratio", "skew_diff", "kurt_ratio",
        "ks_pvalue", "acf_r_lag1_diff", "acf_abs_r_lag1_diff",
        "ljung_box_r2_pvalue", "corr_frobenius_ratio",
    ):
        assert result.passed[m], f"HistoricalReplay should pass {m}, got score={result.scores[m]}"


def test_gaussian_fails_vol_clustering(garch_real):
    val = GeneratorValidator(garch_real, n_paths=10, seed=0)
    gen = GaussianGenerator.fit(garch_real)
    result = val.validate(gen)
    # Gaussian has zero r^2 autocorrelation by construction -> LB doesn't reject.
    assert not result.passed["ljung_box_r2_pvalue"]
    # but matches marginals (since fit on this data).
    assert result.passed["mean_z"]
    assert result.passed["std_ratio"]


def test_permutation_fails_vol_clustering(garch_real):
    val = GeneratorValidator(garch_real, n_paths=10, seed=0)
    gen = PermutationGenerator(garch_real)
    result = val.validate(gen)
    # Permutation destroys temporal structure -> no vol clustering.
    assert not result.passed["ljung_box_r2_pvalue"]
    # marginals still match.
    assert result.passed["mean_z"]
    assert result.passed["std_ratio"]


def test_block_bootstrap_passes_vol_clustering(garch_real):
    val = GeneratorValidator(garch_real, n_paths=10, seed=0)
    gen = BlockBootstrapGenerator(garch_real, block_length=20)
    result = val.validate(gen)
    assert result.passed["ljung_box_r2_pvalue"]
    assert result.passed["acf_abs_r_lag1_diff"]
    assert result.passed["mean_z"]
    assert result.passed["std_ratio"]


# --- threshold customization ---------------------------------------------

def test_custom_thresholds_override_defaults(iid_real):
    # A nonsense-loose threshold for ljung_box_r2_pvalue: anything passes.
    custom = dict(DEFAULT_THRESHOLDS)
    custom["ljung_box_r2_pvalue"] = Threshold()  # open
    val = GeneratorValidator(iid_real, n_paths=5, seed=0, thresholds=custom)
    result = val.validate(GaussianGenerator.fit(iid_real))
    assert result.passed["ljung_box_r2_pvalue"]


def test_empty_thresholds_means_report_only(iid_real):
    val = GeneratorValidator(iid_real, n_paths=5, seed=0, thresholds={})
    result = val.validate(GaussianGenerator.fit(iid_real))
    for m in result.scores:
        assert result.passed[m] is True
    assert result.overall_passed is True


def test_overall_passed_is_and_of_per_metric(iid_real):
    val = GeneratorValidator(iid_real, n_paths=5, seed=0)
    result = val.validate(GaussianGenerator.fit(iid_real))
    expected = all(result.passed.values())
    assert result.overall_passed == expected


# --- input validation ----------------------------------------------------

def test_validator_rejects_bad_inputs():
    with pytest.raises(ValueError, match="must be 2-D"):
        GeneratorValidator(np.zeros((10,)))
    with pytest.raises(ValueError, match="n_paths"):
        GeneratorValidator(np.zeros((10, 3)), n_paths=0)
    with pytest.raises(ValueError, match="n_steps"):
        GeneratorValidator(np.zeros((10, 3)), n_steps=0)


# --- plot() smoke tests -------------------------------------------------

def test_validation_result_carries_returns(garch_real):
    val = GeneratorValidator(garch_real, n_paths=3, seed=0)
    result = val.validate(BlockBootstrapGenerator(garch_real, block_length=20))
    assert result.real_returns is not None
    assert result.synth_returns is not None
    assert result.synth_returns.shape == (3, val.n_steps, garch_real.shape[1])


def test_validation_plot_smoke(garch_real):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    val = GeneratorValidator(garch_real, n_paths=5, seed=0)
    result = val.validate(BlockBootstrapGenerator(garch_real, block_length=20))
    fig = result.plot()
    assert fig is not None
    assert len(fig.axes) == 6
    plt.close(fig)


def test_validation_plot_univariate(garch_real):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    real_1d = garch_real[:, [0]]
    val = GeneratorValidator(real_1d, n_paths=5, seed=0)
    result = val.validate(BlockBootstrapGenerator(real_1d, block_length=20))
    fig = result.plot()
    assert fig is not None
    plt.close(fig)


def test_validation_plot_requires_returns():
    pytest.importorskip("matplotlib")
    result = ValidationResult(
        scores={}, thresholds={}, passed={},
        overall_passed=True, generator_config={},
        n_paths=0, n_steps=0,
    )
    with pytest.raises(ValueError, match="real_returns"):
        result.plot()

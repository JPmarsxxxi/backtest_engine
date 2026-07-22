import numpy as np
import pytest

from backtest.simulation import GaussianGenerator, PermutationGenerator
from backtest.simulation.cache import PathTensorCache


@pytest.fixture
def cache(tmp_path):
    return PathTensorCache(tmp_path / "path_cache")


@pytest.fixture
def real_returns():
    rng = np.random.default_rng(0)
    return rng.normal(0.0003, 0.015, (1260, 5))


# --- basic operations ---------------------------------------------------

def test_cache_miss_returns_none(cache):
    gen = GaussianGenerator(mu=0.0, sigma=0.01)
    assert cache.get(gen, seed=0, n_paths=4, n_steps=100, n_assets=3) is None


def test_cache_round_trip(cache):
    gen = GaussianGenerator(mu=0.0, sigma=0.01)
    tensor = gen.sample(n_paths=4, n_steps=100, n_assets=3, seed=0)
    cache.put(gen, seed=0, n_paths=4, n_steps=100, n_assets=3, tensor=tensor)
    retrieved = cache.get(gen, seed=0, n_paths=4, n_steps=100, n_assets=3)
    assert retrieved is not None
    np.testing.assert_array_equal(tensor, retrieved)


def test_cache_dtype_and_shape_preserved(cache):
    gen = GaussianGenerator(mu=0.0, sigma=0.01)
    tensor = gen.sample(n_paths=2, n_steps=50, n_assets=3, seed=0)
    cache.put(gen, 0, 2, 50, 3, tensor)
    retrieved = cache.get(gen, 0, 2, 50, 3)
    assert retrieved.dtype == np.float64
    assert retrieved.shape == tensor.shape


# --- key uniqueness -----------------------------------------------------

def test_different_generator_configs_have_different_keys(cache, real_returns):
    g1 = GaussianGenerator(mu=0.0, sigma=0.01)
    g2 = GaussianGenerator(mu=0.0, sigma=0.02)  # different sigma
    g3 = PermutationGenerator(real_returns)
    t = np.zeros((2, 50, 3))
    cache.put(g1, 0, 2, 50, 3, t)
    cache.put(g2, 0, 2, 50, 3, t)
    cache.put(g3, 0, 2, 50, 3, t)
    assert cache.stats()["count"] == 3


def test_different_seeds_have_different_keys(cache):
    gen = GaussianGenerator(mu=0.0, sigma=0.01)
    t = np.zeros((2, 50, 3))
    cache.put(gen, 0, 2, 50, 3, t)
    cache.put(gen, 1, 2, 50, 3, t)
    assert cache.stats()["count"] == 2


def test_different_dims_have_different_keys(cache):
    gen = GaussianGenerator(mu=0.0, sigma=0.01)
    cache.put(gen, 0, 2, 50, 3, np.zeros((2, 50, 3)))
    cache.put(gen, 0, 4, 50, 3, np.zeros((4, 50, 3)))
    cache.put(gen, 0, 2, 100, 3, np.zeros((2, 100, 3)))
    cache.put(gen, 0, 2, 50, 5, np.zeros((2, 50, 5)))
    assert cache.stats()["count"] == 4


def test_same_config_same_key_overwrites(cache):
    gen = GaussianGenerator(mu=0.0, sigma=0.01)
    cache.put(gen, 0, 2, 50, 3, np.zeros((2, 50, 3)))
    cache.put(gen, 0, 2, 50, 3, np.ones((2, 50, 3)))
    assert cache.stats()["count"] == 1
    retrieved = cache.get(gen, 0, 2, 50, 3)
    assert (retrieved == 1.0).all()


# --- maintenance --------------------------------------------------------

def test_clear_removes_all_files(cache):
    gen = GaussianGenerator(mu=0.0, sigma=0.01)
    cache.put(gen, 0, 2, 50, 3, np.zeros((2, 50, 3)))
    cache.put(gen, 1, 2, 50, 3, np.zeros((2, 50, 3)))
    assert cache.stats()["count"] == 2
    cache.clear()
    assert cache.stats()["count"] == 0


def test_stats_reports_size(cache):
    gen = GaussianGenerator(mu=0.0, sigma=0.01)
    tensor = np.zeros((10, 100, 5), dtype=np.float64)
    cache.put(gen, 0, 10, 100, 5, tensor)
    s = cache.stats()
    assert s["count"] == 1
    # 10 * 100 * 5 * 8 bytes = 40000, plus numpy header overhead.
    assert s["total_bytes"] >= 40000

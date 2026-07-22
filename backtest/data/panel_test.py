import warnings

import numpy as np
import pandas as pd
import pytest

from backtest.data import DataPanel, DataView, FieldSpec


@pytest.fixture
def prices():
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    return pd.DataFrame(
        np.arange(30, dtype=float).reshape(10, 3) + 100,
        index=idx,
        columns=["AAPL", "MSFT", "GOOG"],
    )


@pytest.fixture
def volume(prices):
    return pd.DataFrame(
        np.full(prices.shape, 1_000_000.0),
        index=prices.index,
        columns=prices.columns,
    )


def test_construct_prices_only(prices):
    p = DataPanel(prices)
    assert p.dates.equals(prices.index)
    assert p.assets_all.equals(prices.columns)


def test_validate_index_type():
    bad = pd.DataFrame(np.zeros((3, 2)), index=[0, 1, 2], columns=["A", "B"])
    with pytest.raises(TypeError):
        DataPanel(bad)


def test_validate_monotonic(prices):
    shuffled = prices.iloc[[2, 0, 1, 3, 4, 5, 6, 7, 8, 9]]
    with pytest.raises(ValueError):
        DataPanel(shuffled)


def test_validate_duplicate_index(prices):
    dup = prices.copy()
    dup.index = pd.DatetimeIndex([prices.index[0]] * len(prices))
    with pytest.raises(ValueError):
        DataPanel(dup)


def test_validate_alignment(prices):
    bad_vol = pd.DataFrame(
        np.zeros((10, 3)),
        index=pd.date_range("2025-01-01", periods=10, freq="B"),
        columns=prices.columns,
    )
    with pytest.raises(ValueError):
        DataPanel(prices, volume=bad_vol)

    bad_cols = pd.DataFrame(
        np.zeros(prices.shape),
        index=prices.index,
        columns=["X", "Y", "Z"],
    )
    with pytest.raises(ValueError):
        DataPanel(prices, volume=bad_cols)


def test_as_of_exact_date(prices):
    p = DataPanel(prices)
    v = p.as_of(prices.index[5])
    assert v.t == prices.index[5]
    assert len(v.prices) == 6


def test_as_of_snaps_to_prior(prices):
    p = DataPanel(prices)
    between = prices.index[5] + pd.Timedelta(hours=12)
    v = p.as_of(between)
    assert v.t == prices.index[5]


def test_as_of_before_first_raises(prices):
    p = DataPanel(prices)
    with pytest.raises(ValueError):
        p.as_of(prices.index[0] - pd.Timedelta(days=1))


def test_pit_no_future_leakage(prices):
    p = DataPanel(prices)
    v = p.as_of(prices.index[3])
    assert v.prices.index.max() == prices.index[3]
    assert prices.index[4] not in v.prices.index


def test_lag_shifts_forward(prices):
    p = DataPanel(prices, specs={"prices": FieldSpec(lag=2)})
    v = p.as_of(prices.index[5])
    assert v.prices.iloc[-1].equals(prices.iloc[3].rename(prices.index[5]))
    assert v.prices.iloc[:2].isna().all().all()


def test_missing_ffill(prices):
    p_with_nan = prices.copy()
    p_with_nan.iloc[3, 0] = np.nan
    p_with_nan.iloc[4, 0] = np.nan
    p = DataPanel(p_with_nan, specs={"prices": FieldSpec(missing="ffill")})
    v = p.as_of(prices.index[5])
    assert v.prices.iloc[3, 0] == prices.iloc[2, 0]
    assert v.prices.iloc[4, 0] == prices.iloc[2, 0]


def test_missing_ffill_max_staleness(prices):
    p_with_nan = prices.copy()
    p_with_nan.iloc[3:7, 0] = np.nan
    p = DataPanel(
        p_with_nan,
        specs={"prices": FieldSpec(missing="ffill", max_staleness=2)},
    )
    v = p.as_of(prices.index[7])
    assert v.prices.iloc[3, 0] == prices.iloc[2, 0]
    assert v.prices.iloc[4, 0] == prices.iloc[2, 0]
    assert np.isnan(v.prices.iloc[5, 0])
    assert np.isnan(v.prices.iloc[6, 0])


def test_missing_value(prices):
    p_with_nan = prices.copy()
    p_with_nan.iloc[3, 1] = np.nan
    p = DataPanel(
        p_with_nan,
        specs={"prices": FieldSpec(missing="value", fill_value=-1.0)},
    )
    v = p.as_of(prices.index[5])
    assert v.prices.iloc[3, 1] == -1.0


def test_missing_unknown_raises(prices):
    with pytest.raises(ValueError):
        DataPanel(prices, specs={"prices": FieldSpec(missing="bogus")})


def test_assets_default_excludes_nan(prices):
    p_with_nan = prices.copy()
    p_with_nan.iloc[5, 1] = np.nan
    p = DataPanel(p_with_nan)
    v = p.as_of(prices.index[5])
    assert "MSFT" not in v.assets
    assert "AAPL" in v.assets
    assert "GOOG" in v.assets


def test_assets_with_universe(prices):
    u = pd.DataFrame(True, index=prices.index, columns=prices.columns)
    u.iloc[5, 0] = False
    p = DataPanel(prices, universe=u)
    v = p.as_of(prices.index[5])
    assert "AAPL" not in v.assets
    assert "MSFT" in v.assets


def test_features(prices):
    f = pd.DataFrame(
        np.arange(30, dtype=float).reshape(10, 3),
        index=prices.index,
        columns=prices.columns,
    )
    p = DataPanel(
        prices,
        features={"signal": f},
        specs={"signal": FieldSpec(lag=1)},
    )
    v = p.as_of(prices.index[5])
    assert v.has_feature("signal")
    assert v.feature("signal").iloc[-1].equals(f.iloc[4].rename(prices.index[5]))


def test_volume_passthrough(prices, volume):
    p = DataPanel(prices, volume=volume)
    v = p.as_of(prices.index[5])
    assert v.volume.shape == (6, 3)


def _random_walk_panel(seed: int = 0, n: int = 200, k: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.01, (n, k))
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    cols = [f"A{i}" for i in range(k)]
    return pd.DataFrame(np.cumprod(1 + rets, axis=0) * 100, index=idx, columns=cols)


def test_outlier_clean_data_no_warning():
    clean = _random_walk_panel()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        DataPanel(clean)


def test_outlier_spike_warns():
    dirty = _random_walk_panel()
    dirty.iloc[100, 0] *= 1.5  # ~50% jump
    with pytest.warns(UserWarning, match="DataPanel:"):
        DataPanel(dirty)


def test_outlier_report_shape():
    dirty = _random_walk_panel()
    dirty.iloc[100, 0] *= 1.5
    p = DataPanel(dirty, check_outliers=False)
    rep = p.outlier_report()
    assert {"date", "asset", "return", "mad_score"} <= set(rep.columns)
    assert len(rep) >= 1
    assert rep.iloc[0]["asset"] == "A0"
    assert rep["mad_score"].is_monotonic_decreasing


def test_outlier_opt_out():
    dirty = _random_walk_panel()
    dirty.iloc[100, 0] *= 1.5
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        DataPanel(dirty, check_outliers=False)


def test_outlier_short_history_skipped(prices):
    spiked = prices.copy()
    spiked.iloc[5, 0] *= 2.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        DataPanel(spiked)


def test_outlier_threshold_configurable():
    dirty = _random_walk_panel()
    dirty.iloc[100, 0] *= 1.5
    p_loose = DataPanel(dirty, check_outliers=False)
    rep_loose = p_loose.outlier_report(mad_threshold=100.0)
    rep_tight = p_loose.outlier_report(mad_threshold=5.0)
    assert len(rep_loose) <= len(rep_tight)

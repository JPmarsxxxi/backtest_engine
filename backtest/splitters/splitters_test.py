from math import comb

import numpy as np
import pandas as pd
import pytest

from backtest.splitters import CombinatorialPurgedCV, WalkForward


@pytest.fixture
def dates():
    return pd.date_range("2024-01-01", periods=120, freq="B")


def test_walk_forward_train_pct(dates):
    splits = list(WalkForward(train_pct=0.7).split(dates))
    assert len(splits) == 1
    train, test = splits[0]
    assert len(train) == 84
    assert len(test) == 36
    assert train[-1] < test[0]


def test_walk_forward_train_end(dates):
    end = dates[60]
    train, test = next(WalkForward(train_end=end).split(dates))
    assert train[-1] == end
    assert test[0] == dates[61]
    assert len(train) == 61
    assert len(test) == 59


def test_walk_forward_embargo(dates):
    train, test = next(WalkForward(train_pct=0.7, embargo_bars=5).split(dates))
    assert len(train) == 84 - 5
    assert len(test) == 36
    assert (test[0] - train[-1]).days >= 1


def test_walk_forward_n_splits(dates):
    assert WalkForward(train_pct=0.5).n_splits() == 1


def test_walk_forward_validation():
    with pytest.raises(ValueError):
        WalkForward()
    with pytest.raises(ValueError):
        WalkForward(train_pct=0.7, train_end="2024-01-01")
    with pytest.raises(ValueError):
        WalkForward(train_pct=0)
    with pytest.raises(ValueError):
        WalkForward(train_pct=1)
    with pytest.raises(ValueError):
        WalkForward(train_pct=-0.1)
    with pytest.raises(ValueError):
        WalkForward(train_pct=0.7, embargo_bars=-1)


def test_cpcv_n_splits():
    assert CombinatorialPurgedCV(n_splits=6, n_test_groups=2).n_splits() == 15
    assert CombinatorialPurgedCV(n_splits=10, n_test_groups=3).n_splits() == comb(10, 3)
    assert CombinatorialPurgedCV(n_splits=4, n_test_groups=1).n_splits() == 4


def test_cpcv_yields_correct_count(dates):
    cv = CombinatorialPurgedCV(n_splits=6, n_test_groups=2)
    splits = list(cv.split(dates))
    assert len(splits) == cv.n_splits()


def test_cpcv_test_groups_disjoint(dates):
    cv = CombinatorialPurgedCV(n_splits=6, n_test_groups=2)
    for train, test in cv.split(dates):
        assert len(train.intersection(test)) == 0


def test_cpcv_no_purge_no_embargo_full_coverage(dates):
    cv = CombinatorialPurgedCV(
        n_splits=6, n_test_groups=2, purge_bars=0, embargo_pct=0
    )
    for train, test in cv.split(dates):
        union = train.union(test)
        assert len(union) == len(dates)


def test_cpcv_purge_drops_neighbors():
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    purge = 2
    cv = CombinatorialPurgedCV(
        n_splits=6, n_test_groups=1, purge_bars=purge, embargo_pct=0
    )
    pos = {d: i for i, d in enumerate(dates)}
    for train, test in cv.split(dates):
        train_idx = {pos[t] for t in train}
        for tp in (pos[t] for t in test):
            for d in range(-purge, purge + 1):
                assert tp + d not in train_idx, (
                    f"train at {tp + d} within purge of test at {tp}"
                )


def test_cpcv_embargo_drops_after():
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    embargo_pct = 0.05
    embargo_bars = int(100 * embargo_pct)
    cv = CombinatorialPurgedCV(
        n_splits=10, n_test_groups=1, purge_bars=0, embargo_pct=embargo_pct
    )
    pos = {d: i for i, d in enumerate(dates)}
    for train, test in cv.split(dates):
        train_idx = {pos[t] for t in train}
        last_test = max(pos[t] for t in test)
        for i in range(last_test + 1, min(last_test + 1 + embargo_bars, len(dates))):
            assert i not in train_idx


def test_cpcv_coverage_symmetry(dates):
    n_groups = 6
    k = 2
    cv = CombinatorialPurgedCV(
        n_splits=n_groups, n_test_groups=k, purge_bars=0, embargo_pct=0
    )
    test_count = pd.Series(0, index=dates, dtype=int)
    for _, test in cv.split(dates):
        test_count.loc[test] += 1
    expected = comb(n_groups - 1, k - 1)
    assert test_count.min() == expected
    assert test_count.max() == expected


def test_cpcv_validation():
    with pytest.raises(ValueError):
        CombinatorialPurgedCV(n_splits=1)
    with pytest.raises(ValueError):
        CombinatorialPurgedCV(n_splits=5, n_test_groups=5)
    with pytest.raises(ValueError):
        CombinatorialPurgedCV(n_splits=5, n_test_groups=6)
    with pytest.raises(ValueError):
        CombinatorialPurgedCV(n_splits=5, n_test_groups=0)
    with pytest.raises(ValueError):
        CombinatorialPurgedCV(n_splits=5, n_test_groups=2, purge_bars=-1)
    with pytest.raises(ValueError):
        CombinatorialPurgedCV(n_splits=5, n_test_groups=2, embargo_pct=-0.1)
    with pytest.raises(ValueError):
        CombinatorialPurgedCV(n_splits=5, n_test_groups=2, embargo_pct=1.0)


def test_cpcv_empty_dates():
    cv = CombinatorialPurgedCV(n_splits=6, n_test_groups=2)
    splits = list(cv.split(pd.DatetimeIndex([])))
    assert splits == []


def test_cpcv_test_count_per_split(dates):
    cv = CombinatorialPurgedCV(n_splits=6, n_test_groups=2, embargo_pct=0)
    for _, test in cv.split(dates):
        # 2 groups * 20 bars/group = 40 bars
        assert len(test) == 40


def test_walk_forward_n_paths():
    assert WalkForward(train_pct=0.5).n_paths() == 1


def test_cpcv_n_paths():
    cv = CombinatorialPurgedCV(n_splits=6, n_test_groups=2)
    assert cv.n_paths() == comb(5, 1)
    cv = CombinatorialPurgedCV(n_splits=10, n_test_groups=3)
    assert cv.n_paths() == comb(9, 2)


def test_walk_forward_assemble_paths(dates):
    train, test = next(WalkForward(train_pct=0.7).split(dates))
    fake_returns = pd.Series(np.linspace(0.001, 0.002, len(test)), index=test)
    paths = WalkForward(train_pct=0.7).assemble_paths(dates, [fake_returns])
    assert paths.columns.tolist() == ["path_0"]
    pd.testing.assert_series_equal(
        paths["path_0"].dropna(), fake_returns, check_names=False
    )


def test_walk_forward_assemble_paths_validation(dates):
    wf = WalkForward(train_pct=0.7)
    with pytest.raises(ValueError):
        wf.assemble_paths(dates, [pd.Series(), pd.Series()])


def test_cpcv_assemble_paths_full_coverage(dates):
    cv = CombinatorialPurgedCV(n_splits=6, n_test_groups=2, embargo_pct=0)
    splits = list(cv.split(dates))
    fake_returns = [
        pd.Series(np.full(len(test), float(i + 1)) * 0.001, index=test)
        for i, (_, test) in enumerate(splits)
    ]
    paths = cv.assemble_paths(dates, fake_returns)
    assert paths.shape == (len(dates), cv.n_paths())
    # No NaN gaps after assembly
    assert not paths.isna().any().any()


def test_cpcv_assemble_paths_uses_distinct_splits(dates):
    cv = CombinatorialPurgedCV(n_splits=6, n_test_groups=2, embargo_pct=0)
    splits = list(cv.split(dates))
    fake_returns = [
        pd.Series(np.full(len(test), float(i + 1)) * 0.001, index=test)
        for i, (_, test) in enumerate(splits)
    ]
    paths = cv.assemble_paths(dates, fake_returns)
    # Different paths should differ (each uses a different split per group)
    p0 = paths["path_0"]
    p1 = paths["path_1"]
    assert not p0.equals(p1)


def test_cpcv_assemble_paths_wrong_count_raises(dates):
    cv = CombinatorialPurgedCV(n_splits=6, n_test_groups=2)
    with pytest.raises(ValueError):
        cv.assemble_paths(dates, [pd.Series()])


def test_cpcv_purge_and_embargo_stack_on_trailing_side():
    # Lopez de Prado: leading buffer = purge; trailing buffer = purge + embargo.
    dates = pd.date_range("2024-01-01", periods=200, freq="B")
    purge = 3
    embargo_pct = 0.05
    embargo_bars = int(200 * embargo_pct)  # 10
    cv = CombinatorialPurgedCV(
        n_splits=10, n_test_groups=1, purge_bars=purge, embargo_pct=embargo_pct
    )
    pos = {d: i for i, d in enumerate(dates)}
    for train, test in cv.split(dates):
        train_idx = {pos[t] for t in train}
        first_test = min(pos[t] for t in test)
        last_test = max(pos[t] for t in test)
        # Leading: bars [first_test - purge, first_test) excluded from train.
        for i in range(max(0, first_test - purge), first_test):
            assert i not in train_idx, f"leading purge missed at {i}"
        # Trailing: bars [last_test + 1, last_test + 1 + purge + embargo) excluded.
        trailing = purge + embargo_bars
        for i in range(last_test + 1, min(last_test + 1 + trailing, len(dates))):
            assert i not in train_idx, f"trailing purge+embargo missed at {i}"
        # And the bar just past the trailing buffer IS in train (unless it
        # belongs to a neighboring test run via a different group boundary).
        # We don't check that here to avoid coupling to group geometry.

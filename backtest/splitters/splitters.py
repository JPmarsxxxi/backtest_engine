from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from math import comb
from typing import Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

SplitTuple = Tuple[pd.DatetimeIndex, pd.DatetimeIndex]


class Splitter(ABC):
    """Yields (train_dates, test_dates) tuples."""

    @abstractmethod
    def split(self, dates: pd.DatetimeIndex) -> Iterator[SplitTuple]: ...

    @abstractmethod
    def n_splits(self) -> int: ...

    @abstractmethod
    def n_paths(self) -> int: ...

    @abstractmethod
    def assemble_paths(
        self,
        all_dates: pd.DatetimeIndex,
        split_returns: List[pd.Series],
    ) -> pd.DataFrame: ...


class WalkForward(Splitter):
    """Single-path walk-forward: train on the past, test on the future.

    Specify exactly one of train_end (inclusive cutoff date) or train_pct
    (fraction of bars in train). embargo_bars drops the last N bars of train
    to prevent boundary leakage.
    """

    def __init__(
        self,
        train_end: Optional[pd.Timestamp] = None,
        train_pct: Optional[float] = None,
        embargo_bars: int = 0,
    ):
        if (train_end is None) == (train_pct is None):
            raise ValueError("specify exactly one of train_end or train_pct")
        if train_pct is not None and not 0 < train_pct < 1:
            raise ValueError("train_pct must be in (0, 1)")
        if embargo_bars < 0:
            raise ValueError("embargo_bars must be >= 0")
        self.train_end = pd.Timestamp(train_end) if train_end is not None else None
        self.train_pct = train_pct
        self.embargo_bars = embargo_bars

    def split(self, dates: pd.DatetimeIndex) -> Iterator[SplitTuple]:
        dates = pd.DatetimeIndex(dates)
        if self.train_end is not None:
            split_idx = int(dates.searchsorted(self.train_end, side="right"))
        else:
            split_idx = int(len(dates) * self.train_pct)
        train_end_idx = max(0, split_idx - self.embargo_bars)
        train = dates[:train_end_idx]
        test = dates[split_idx:]
        yield train, test

    def n_splits(self) -> int:
        return 1

    def n_paths(self) -> int:
        return 1

    def assemble_paths(
        self,
        all_dates: pd.DatetimeIndex,
        split_returns: List[pd.Series],
    ) -> pd.DataFrame:
        if len(split_returns) != 1:
            raise ValueError(f"WalkForward expects 1 split, got {len(split_returns)}")
        return pd.DataFrame({"path_0": split_returns[0]})


class CombinatorialPurgedCV(Splitter):
    """Combinatorial Purged Cross-Validation (Lopez de Prado 2018).

    The timeline is split into n_splits contiguous groups. For each combination
    of n_test_groups groups as test, the rest are train. Yields C(n_splits,
    n_test_groups) splits.

    Purging: drops train bars within purge_bars of either side of any test run.
    Embargo: drops train bars within (embargo_pct * total_bars) AFTER any test run.
    """

    def __init__(
        self,
        n_splits: int = 6,
        n_test_groups: int = 2,
        purge_bars: int = 0,
        embargo_pct: float = 0.01,
    ):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if n_test_groups < 1:
            raise ValueError("n_test_groups must be >= 1")
        if n_test_groups >= n_splits:
            raise ValueError("n_test_groups must be < n_splits")
        if purge_bars < 0:
            raise ValueError("purge_bars must be >= 0")
        if not 0 <= embargo_pct < 1:
            raise ValueError("embargo_pct must be in [0, 1)")
        self.n_groups = n_splits
        self.n_test_groups = n_test_groups
        self.purge_bars = purge_bars
        self.embargo_pct = embargo_pct

    def split(self, dates: pd.DatetimeIndex) -> Iterator[SplitTuple]:
        dates = pd.DatetimeIndex(dates)
        n = len(dates)
        if n == 0:
            return
        embargo_bars = int(n * self.embargo_pct)

        bounds = np.linspace(0, n, self.n_groups + 1, dtype=int)
        groups = [(bounds[i], bounds[i + 1]) for i in range(self.n_groups)]

        for combo in itertools.combinations(range(self.n_groups), self.n_test_groups):
            test_mask = np.zeros(n, dtype=bool)
            for g_idx in combo:
                start, end = groups[g_idx]
                test_mask[start:end] = True
            train_mask = self._build_train_mask(test_mask, self.purge_bars, embargo_bars)
            yield dates[train_mask], dates[test_mask]

    def n_splits(self) -> int:
        return comb(self.n_groups, self.n_test_groups)

    def n_paths(self) -> int:
        return comb(self.n_groups - 1, self.n_test_groups - 1)

    def assemble_paths(
        self,
        all_dates: pd.DatetimeIndex,
        split_returns: List[pd.Series],
    ) -> pd.DataFrame:
        all_dates = pd.DatetimeIndex(all_dates)
        n = len(all_dates)
        bounds = np.linspace(0, n, self.n_groups + 1, dtype=int)
        group_dates = [all_dates[bounds[g]:bounds[g + 1]] for g in range(self.n_groups)]

        all_combos = list(
            itertools.combinations(range(self.n_groups), self.n_test_groups)
        )
        if len(split_returns) != len(all_combos):
            raise ValueError(
                f"expected {len(all_combos)} split returns, got {len(split_returns)}"
            )

        group_to_splits: dict[int, list[int]] = {g: [] for g in range(self.n_groups)}
        for split_idx, combo in enumerate(all_combos):
            for g in combo:
                group_to_splits[g].append(split_idx)

        n_paths = self.n_paths()
        path_data: dict[str, pd.Series] = {}
        for p in range(n_paths):
            segments = []
            for g in range(self.n_groups):
                split_idx = group_to_splits[g][p]
                seg = split_returns[split_idx].reindex(group_dates[g])
                segments.append(seg)
            path = pd.concat(segments).sort_index()
            path_data[f"path_{p}"] = path
        return pd.DataFrame(path_data)

    @staticmethod
    def _build_train_mask(
        test_mask: np.ndarray, purge: int, embargo: int
    ) -> np.ndarray:
        n = len(test_mask)
        train = ~test_mask
        if purge == 0 and embargo == 0:
            return train
        diff = np.diff(np.r_[False, test_mask, False].astype(np.int8))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        for s, e in zip(starts, ends):
            lo = max(0, s - purge)
            hi = min(n, e + purge + embargo)
            train[lo:hi] = False
        return train

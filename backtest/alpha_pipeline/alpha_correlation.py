"""Skill 9 - Alpha correlation.

Measure a new alpha's uniqueness against a pool of existing ("winner") alphas
(Finding Alphas, Ch. 8). Lower correlation = more unique = more valuable
(Ch. 4: max-corr > 0.7 too high, < 0.3 good).

Representation depends on the method:
    pearson / temporal / weekly / sign : 1-D PnL series per alpha (profit line).
    position / trading                 : 2-D position frame (date x asset).

Pairwise correlation math (Ch. 8):
    pearson  (Eq 3): standard Pearson of two PnL series.
    temporal (Eq 6): weighted cosine, w_t = 1 - t/n with t=1 most recent:
                     r = sum_t w_t p_t q_t / sqrt(sum_t w_t p_t^2 * sum_t w_t q_t^2)
    weekly   (Eq 9): aggregate into 5-bar (weekly) means, then Pearson.
    sign     (Eq 11): Pearson of sign(p), sign(q).
    position (Eq 13): stack last d days of positions -> one vector, Pearson.
    trading  (Eq 14): stack last d days of position *differences*, Pearson.
                      Different universes use the column intersection.

Pool aggregates (Ch. 8):
    max_corr  = max of pairwise correlations.
    t_corr    = sum of pairwise correlations.
    avg_score = sum_{j=-10..9} c_j * (j/10), c_j = fraction of pairwise corrs in
                histogram bin j (20 bins over [-1, 1], Table 8.1).

The pool is the curated `winners/` library: load_pool() reads every winner's
saved pnl.parquet (or positions.parquet) so each new candidate is checked
against the accumulated keepers.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd

_PNL_METHODS = ("pearson", "temporal", "weekly", "sign")
_POS_METHODS = ("position", "trading")
_METHODS = _PNL_METHODS + _POS_METHODS


def _pearson_1d(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _align(p: pd.Series, q: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    df = pd.concat([p, q], axis=1).dropna()
    return df.iloc[:, 0].to_numpy(float), df.iloc[:, 1].to_numpy(float)


def _temporal(a: np.ndarray, b: np.ndarray) -> float:
    n = a.size
    if n < 2:
        return float("nan")
    w = np.arange(n) / n            # w_t = 1 - t/n, t=1 most recent (oldest -> 0)
    num = np.sum(w * a * b)
    den = np.sqrt(np.sum(w * a * a) * np.sum(w * b * b))
    return float(num / den) if den > 0 else float("nan")


def _weekly(a: np.ndarray, b: np.ndarray) -> float:
    k = a.size // 5
    if k < 2:
        return float("nan")
    wa = a[: k * 5].reshape(k, 5).mean(axis=1)
    wb = b[: k * 5].reshape(k, 5).mean(axis=1)
    return _pearson_1d(wa, wb)


def _pnl_corr(new: pd.Series, member: pd.Series, method: str) -> float:
    a, b = _align(new, member)
    if method == "pearson":
        return _pearson_1d(a, b)
    if method == "temporal":
        return _temporal(a, b)
    if method == "weekly":
        return _weekly(a, b)
    return _pearson_1d(np.sign(a), np.sign(b))  # sign


def _pos_corr(new: pd.DataFrame, member: pd.DataFrame, method: str, d: int) -> float:
    cols = new.columns.intersection(member.columns)
    if len(cols) == 0:
        return float("nan")
    P, Q = new[cols], member[cols]
    if method == "trading":
        P, Q = P.diff(), Q.diff()
    P, Q = P.iloc[-d:], Q.iloc[-d:]
    a, b = P.to_numpy(float).ravel(), Q.to_numpy(float).ravel()
    mask = ~(np.isnan(a) | np.isnan(b))
    return _pearson_1d(a[mask], b[mask])


def run(
    pool: List[Union[pd.Series, pd.DataFrame]],
    new: Union[pd.Series, pd.DataFrame],
    method: str = "pearson",
    d: int = 20,
) -> dict:
    """Correlate a new alpha against a pool of existing alphas.

    Parameters
    ----------
    pool : list
        Existing alphas. PnL Series for pearson/temporal/weekly/sign; position
        DataFrames (date x asset) for position/trading. Empty pool is allowed
        (first alpha) -> max_corr/avg_score are NaN.
    new : pd.Series | pd.DataFrame
        Candidate alpha, same representation as the pool members.
    method : {'pearson', 'temporal', 'weekly', 'sign', 'position', 'trading'}
    d : int, default 20
        Lookback (bars) for position/trading correlation (Ch. 8).

    Returns
    -------
    dict
        max_corr, t_corr, avg_score, pairwise (np.ndarray), histogram
        (counts, np.ndarray), bin_edges (np.ndarray).
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")

    pair = np.array(
        [
            _pnl_corr(new, m, method) if method in _PNL_METHODS
            else _pos_corr(new, m, method, d)
            for m in pool
        ],
        dtype=float,
    )
    valid = pair[~np.isnan(pair)]

    edges = np.round(np.arange(-1.0, 1.0001, 0.1), 2)  # 21 edges -> 20 bins
    counts, _ = np.histogram(valid, bins=edges)
    total = counts.sum()
    levels = edges[:-1]  # bin lower edges = j/10 for j=-10..9

    return {
        "max_corr": float(valid.max()) if valid.size else float("nan"),
        "t_corr": float(np.nansum(pair)),
        "avg_score": float((counts / total * levels).sum()) if total else float("nan"),
        "pairwise": pair,
        "histogram": counts,
        "bin_edges": edges,
    }


def load_pool(winners_dir: Union[str, Path], kind: str = "pnl") -> List:
    """Load the correlation pool from the winners/ library.

    Each winner is a subdirectory containing pnl.parquet (and optionally
    positions.parquet). Returns a list ready to pass to run().

    Parameters
    ----------
    winners_dir : str | Path
        Path to the winners/ directory.
    kind : {'pnl', 'positions'}
        'pnl' -> list of PnL Series (pearson/temporal/weekly/sign);
        'positions' -> list of position DataFrames (position/trading).
    """
    if kind not in ("pnl", "positions"):
        raise ValueError("kind must be 'pnl' or 'positions'")
    fname = "pnl.parquet" if kind == "pnl" else "positions.parquet"
    root = Path(winners_dir)
    pool: List = []
    if not root.exists():
        return pool
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        f = sub / fname
        if f.exists():
            obj = pd.read_parquet(f)
            pool.append(obj.squeeze("columns") if kind == "pnl" else obj)
    return pool


def quick_test() -> None:
    """Synthetic smoke test: PnL methods, position methods, load_pool."""
    import tempfile

    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=20, freq="D")
    base = pd.Series(rng.normal(0, 1, 20), index=idx, name="pnl")

    # PnL pool: one identical to `new`, one its negative
    out = run([base.copy(), -base], base, method="pearson")
    assert np.isclose(out["pairwise"][0], 1.0)    # identical -> +1
    assert np.isclose(out["pairwise"][1], -1.0)   # opposite  -> -1
    assert np.isclose(out["max_corr"], 1.0)
    assert np.isclose(out["t_corr"], 0.0)
    assert out["histogram"].sum() == 2
    # -1 in bin level -1.0, +1 in last bin level 0.9 -> mean -0.05
    assert np.isclose(out["avg_score"], -0.05)

    for meth in ("temporal", "weekly", "sign"):
        assert np.isclose(run([base.copy()], base, method=meth)["pairwise"][0], 1.0), meth

    P = pd.DataFrame(rng.normal(0, 1, (20, 2)), index=idx, columns=["A", "B"])
    for meth in ("position", "trading"):
        assert np.isclose(run([P.copy()], P, method=meth, d=20)["pairwise"][0], 1.0), meth

    # empty pool (first alpha)
    r0 = run([], base, method="pearson")
    assert np.isnan(r0["max_corr"]) and r0["t_corr"] == 0.0

    # load_pool from a temp winners/ dir
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name in ("a", "b"):
            (root / name).mkdir()
            base.to_frame().to_parquet(root / name / "pnl.parquet")
        pool = load_pool(root, kind="pnl")
        assert len(pool) == 2 and all(isinstance(s, pd.Series) for s in pool)
        assert np.isclose(run(pool, base, "pearson")["max_corr"], 1.0)

    print("alpha_correlation.quick_test passed")
    print("\npearson vs pool [identical, negated]:")
    print(f"  pairwise  {out['pairwise']}")
    print(f"  max_corr {out['max_corr']:.3f}  t_corr {out['t_corr']:.3f}  avg_score {out['avg_score']:.3f}")


if __name__ == "__main__":
    quick_test()

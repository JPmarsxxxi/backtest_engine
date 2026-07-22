from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from backtest.data.panel import DataPanel
from backtest.engine.engine import BacktestResult, Engine
from backtest.metrics import compute_metrics
from backtest.registry.hashing import dataset_id_for, strategy_hash
from backtest.selection import dsr, effective_k
from backtest.strategy.base import Strategy

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    strategy_class TEXT NOT NULL,
    strategy_hash TEXT NOT NULL,
    strategy_repr TEXT,
    family TEXT,
    causal_graph_path TEXT,
    exploratory INTEGER NOT NULL,
    dataset_id TEXT NOT NULL,
    train_start TEXT,
    train_end TEXT,
    test_start TEXT,
    test_end TEXT,
    n_obs INTEGER,
    sharpe REAL,
    psr REAL,
    max_drawdown REAL,
    returns_path TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_dataset ON trials(dataset_id);
CREATE INDEX IF NOT EXISTS idx_family ON trials(family);
"""


class TrialRegistry:
    """Persistent log of backtest trials.

    Stores trial metadata in SQLite (registry.sqlite) and per-trial returns as
    parquet files under returns/. Provides queries for K (raw count or
    effective via clustering) used by selection-bias corrections.
    """

    def __init__(self, directory: Union[str, Path]):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "returns").mkdir(exist_ok=True)
        self.db_path = self.directory / "registry.sqlite"
        self._init_schema()

    def _init_schema(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.executescript(_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def log(
        self,
        result: BacktestResult,
        strategy: Strategy,
        data: DataPanel,
        *,
        causal_graph_path: Optional[str] = None,
        family: Optional[str] = None,
        dataset_id: Optional[str] = None,
        skip_causal_graph: bool = False,
        verify_path: bool = True,
    ) -> int:
        if not skip_causal_graph:
            if causal_graph_path is None:
                raise ValueError(
                    "causal_graph_path is required. Pass skip_causal_graph=True "
                    "to log as exploratory (excluded from K)."
                )
            if verify_path and not Path(causal_graph_path).exists():
                raise FileNotFoundError(
                    f"causal graph file not found: {causal_graph_path}"
                )

        metrics = compute_metrics(result.returns, result.equity_curve)

        if dataset_id is None:
            dataset_id = dataset_id_for(data)

        sh = strategy_hash(strategy)
        srepr = repr(strategy)[:500]
        scls = f"{type(strategy).__module__}.{type(strategy).__name__}"

        idx = result.returns.index
        test_start = pd.Timestamp(idx[0]).isoformat() if len(idx) else None
        test_end = pd.Timestamp(idx[-1]).isoformat() if len(idx) else None
        timestamp = datetime.now(timezone.utc).isoformat()

        conn = self._connect()
        returns_path: Optional[str] = None
        try:
            with conn:
                cur = conn.execute(
                    "INSERT INTO trials (timestamp, strategy_class, strategy_hash, "
                    "strategy_repr, family, causal_graph_path, exploratory, "
                    "dataset_id, train_start, train_end, test_start, test_end, "
                    "n_obs, sharpe, psr, max_drawdown, returns_path) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        timestamp, scls, sh, srepr, family,
                        causal_graph_path, int(skip_causal_graph), dataset_id,
                        None, None, test_start, test_end,
                        metrics.n_obs,
                        _safe_float(metrics.sharpe),
                        _safe_float(metrics.psr),
                        _safe_float(metrics.max_drawdown),
                        "",
                    ),
                )
                trial_id = cur.lastrowid
                returns_path = str(self.directory / "returns" / f"{trial_id:08d}.parquet")
                conn.execute(
                    "UPDATE trials SET returns_path = ? WHERE id = ?",
                    (returns_path, trial_id),
                )
                result.returns.to_frame("ret").to_parquet(
                    returns_path, engine="pyarrow"
                )
            return int(trial_id)
        except Exception:
            if returns_path and os.path.exists(returns_path):
                try:
                    os.remove(returns_path)
                except OSError:
                    pass
            raise
        finally:
            conn.close()

    def run(
        self,
        engine: Engine,
        *,
        causal_graph_path: Optional[str] = None,
        family: Optional[str] = None,
        dataset_id: Optional[str] = None,
        skip_causal_graph: bool = False,
        verify_path: bool = True,
        start=None,
        end=None,
        train_dates=None,
        test_dates=None,
    ) -> BacktestResult:
        result = engine.run(
            start=start, end=end,
            train_dates=train_dates, test_dates=test_dates,
        )
        self.log(
            result, engine.strategy, engine.data,
            causal_graph_path=causal_graph_path,
            family=family,
            dataset_id=dataset_id,
            skip_causal_graph=skip_causal_graph,
            verify_path=verify_path,
        )
        return result

    def list(
        self,
        *,
        family: Optional[str] = None,
        dataset_id: Optional[str] = None,
        include_exploratory: bool = True,
    ) -> List[dict]:
        sql = "SELECT * FROM trials WHERE 1=1"
        params: list = []
        if family is not None:
            sql += " AND family = ?"
            params.append(family)
        if dataset_id is not None:
            sql += " AND dataset_id = ?"
            params.append(dataset_id)
        if not include_exploratory:
            sql += " AND exploratory = 0"
        sql += " ORDER BY id"
        conn = self._connect()
        try:
            rows = conn.execute(sql, params).fetchall()
        finally:
            conn.close()
        return [dict(r) for r in rows]

    def get(self, trial_id: int) -> dict:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM trials WHERE id = ?", (trial_id,)
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            raise KeyError(f"trial {trial_id} not found")
        return dict(row)

    def returns(self, trial_id: int) -> pd.Series:
        row = self.get(trial_id)
        df = pd.read_parquet(row["returns_path"], engine="pyarrow")
        s = df["ret"]
        s.name = "returns"
        return s

    def returns_df(
        self,
        *,
        family: Optional[str] = None,
        dataset_id: Optional[str] = None,
        include_exploratory: bool = False,
    ) -> pd.DataFrame:
        trials = self.list(
            family=family,
            dataset_id=dataset_id,
            include_exploratory=include_exploratory,
        )
        if not trials:
            return pd.DataFrame()
        cols = {f"trial_{t['id']}": self.returns(t["id"]) for t in trials}
        return pd.DataFrame(cols)

    def k(
        self,
        *,
        family: Optional[str] = None,
        dataset_id: Optional[str] = None,
        method: str = "count",
        threshold: float = 0.5,
    ) -> int:
        if method == "count":
            return len(
                self.list(
                    family=family,
                    dataset_id=dataset_id,
                    include_exploratory=False,
                )
            )
        if method == "effective":
            df = self.returns_df(
                family=family,
                dataset_id=dataset_id,
                include_exploratory=False,
            )
            if df.shape[1] == 0:
                return 0
            return effective_k(df, threshold=threshold)
        raise ValueError(f"unknown k method: {method!r}")

    def dsr_for(
        self,
        trial_id: int,
        *,
        family: Optional[str] = None,
        dataset_id: Optional[str] = None,
        method: str = "count",
        threshold: float = 0.5,
    ) -> float:
        ret = self.returns(trial_id)
        K = self.k(
            family=family,
            dataset_id=dataset_id,
            method=method,
            threshold=threshold,
        )
        if K < 1:
            K = 1
        return dsr(ret, K=K)


def _safe_float(x) -> Optional[float]:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if pd.notna(f) else None

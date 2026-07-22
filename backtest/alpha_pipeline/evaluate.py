"""Skill 8 - Evaluate.

Convert a final alpha vector into positions, simulate net PnL after costs, and
score it. Two modes:
    mode="engine" : wrap backtest.metrics.compute_metrics (rich report -
                    Sharpe/IR, ann_return, drawdown, turnover, margin, plus
                    Sharpe CI / PSR; DSR is a separate Stage-9 step).
    mode="pdf"    : the Finding Alphas Ch. 5 / WebSim metrics, exact.

Adapter (shared; "raw vector -> positions" conversion happens here):
    positions  pos_t   = signal_t / sum_i|signal_{t,i}| * book_size
                         (gross = book; dollar-neutral if signal was neutralised)
    returns    ret_t   = prices_t / prices_{t-1} - 1
    gross pnl  pnl_t   = sum_i pos_{t-1,i} * ret_{t,i}        (no lookahead)
    traded     d_t     = sum_i |pos_{t,i} - pos_{t-1,i}|      (entry from flat)
    cost       c_t     = sum_i (halfspread_i / 1e4) * |dpos_{t,i}|
                         Half the bid-ask spread per one-way trade (Ch. 7).
                         cost_bps may be scalar (flat), per-instrument Series,
                         or per-bar DataFrame. The per-instrument form is the
                         accurate, liquidity-aware PDF model.
    net pnl    npnl_t  = pnl_t - c_t
    equity     eq_t    = book_size + cumsum(npnl_t)

PDF-exact metrics (mode="pdf"; Ch. 5):
    IR             = mean(npnl) / std(npnl) * sqrt(ann_factor)
    annual_return  = mean(npnl) * ann_factor / (book_size / 2)
    daily_turnover = mean_t(d_t) / book_size
    margin         = sum(npnl) / sum(d_t)                 (profit per $ traded)
    max_drawdown   = (peak - trough of cum npnl) / (book_size / 2)
    pct_profitable_days = mean(npnl > 0)

The most realistic costing (time-varying spreads + size-based market impact)
lives in the full engine via CompositeCostModel (MarketImpact, LiquidityCap);
use the engine on-ramp when an alpha is promising and you want the truth.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

from backtest.metrics import compute_metrics

CostBps = Union[float, pd.Series, pd.DataFrame]


def _positions(signal: pd.DataFrame, book_size: float) -> pd.DataFrame:
    """Normalise signal to dollar positions: gross exposure = book_size."""
    gross = signal.abs().sum(axis=1).replace(0.0, np.nan)
    pos = signal.div(gross, axis=0) * book_size
    return pos.fillna(0.0)


def _cost_series(trades_abs: pd.DataFrame, cost_bps: CostBps) -> pd.Series:
    """Per-bar cost = sum_i (bps_i / 1e4) * |dpos_i|."""
    if isinstance(cost_bps, pd.DataFrame):
        per = cost_bps.reindex(index=trades_abs.index, columns=trades_abs.columns)
        return (trades_abs * per).sum(axis=1) / 1e4
    if isinstance(cost_bps, pd.Series):
        per = cost_bps.reindex(trades_abs.columns)
        return trades_abs.mul(per, axis=1).sum(axis=1) / 1e4
    return trades_abs.sum(axis=1) * float(cost_bps) / 1e4


def _simulate(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    book_size: float,
    cost_bps: CostBps,
) -> dict:
    """Run the position/PnL simulation; return series trimmed to the active period."""
    prices = prices.reindex(index=signal.index, columns=signal.columns)
    pos = _positions(signal, book_size)
    ret = prices.pct_change()

    gross_pnl = (pos.shift(1) * ret).sum(axis=1)          # skipna -> first bar 0
    delta = pos.diff()
    delta.iloc[0] = pos.iloc[0]                            # enter from flat
    trades_abs = delta.abs()
    traded = trades_abs.sum(axis=1)
    cost = _cost_series(trades_abs, cost_bps)
    net_pnl = gross_pnl - cost
    equity = book_size + net_pnl.cumsum()

    # trim leading warmup (no position taken yet)
    active = pos.abs().sum(axis=1) > 0
    if active.any():
        start = active.idxmax()
        net_pnl = net_pnl.loc[start:]
        traded = traded.loc[start:]
        cost = cost.loc[start:]
        trades_abs = trades_abs.loc[start:]
        equity = equity.loc[start:]
        ret = ret.loc[start:]

    return {
        "positions": pos,
        "net_pnl": net_pnl,
        "traded": traded,
        "cost": cost,
        "trades_abs": trades_abs,
        "equity": equity,
        "returns": net_pnl / book_size,
    }


def _pdf_metrics(sim: dict, book_size: float, ann_factor: int) -> dict:
    npnl = sim["net_pnl"]
    traded = sim["traded"]
    half_book = book_size / 2.0

    mean, std = npnl.mean(), npnl.std(ddof=1)
    ir = float(mean / std * np.sqrt(ann_factor)) if std and np.isfinite(std) and std > 0 else float("nan")

    cum = npnl.cumsum()
    drawdown = (cum.cummax() - cum).max()
    total_traded = traded.sum()

    return {
        "IR": ir,
        "annual_return": float(mean * ann_factor / half_book),
        "daily_turnover": float(traded.mean() / book_size),
        "margin": float(npnl.sum() / total_traded) if total_traded > 0 else float("nan"),
        "max_drawdown": float(drawdown / half_book),
        "pct_profitable_days": float((npnl > 0).mean()),
    }


def run(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    book_size: float,
    cost_bps: CostBps,
    mode: str = "engine",
    ann_factor: int = 252,
) -> dict:
    """Evaluate an alpha vector.

    Parameters
    ----------
    signal : pd.DataFrame
        Final signal vector, date x asset.
    prices : pd.DataFrame
        Price frame (same frequency as the signal), date x asset.
    book_size : float
        Total gross capital. Positions are normalised so gross = book_size.
    cost_bps : float | pd.Series | pd.DataFrame
        Half-spread per one-way trade, in basis points. Scalar = flat; Series =
        per-instrument (accurate PDF model); DataFrame = per-bar x instrument.
    mode : {'engine', 'pdf'}
        'engine' -> compute_metrics report dict; 'pdf' -> Ch. 5/WebSim metrics.
    ann_factor : int, default 252
        Annualisation factor (PDF Ch. 5 used 256; 252 is the engine default).

    Returns
    -------
    dict
        mode='pdf'    -> {IR, annual_return, daily_turnover, margin,
                          max_drawdown, pct_profitable_days}
        mode='engine' -> MetricsReport.to_dict() (Sharpe, ann_return, turnover,
                          margin, drawdown, ...).
    """
    if mode not in ("engine", "pdf"):
        raise ValueError(f"mode must be 'engine' or 'pdf', got {mode!r}")

    sim = _simulate(signal, prices, book_size, cost_bps)

    if mode == "pdf":
        return _pdf_metrics(sim, book_size, ann_factor)

    report = compute_metrics(
        sim["returns"],
        sim["equity"],
        costs=sim["cost"],
        trades=sim["trades_abs"],
        ann_factor=ann_factor,
    )
    return report.to_dict()


def quick_test() -> None:
    """Deterministic adapter check + both modes run."""
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    prices = pd.DataFrame(
        {"A": [100.0, 110.0, 121.0], "B": [100.0, 90.0, 81.0]}, index=idx
    )
    # constant long-A / short-B signal
    signal = pd.DataFrame({"A": [1.0, 1.0, 1.0], "B": [-1.0, -1.0, -1.0]}, index=idx)

    m = run(signal, prices, book_size=1000.0, cost_bps=10.0, mode="pdf")

    # hand-computed: pos = +500/-500; pnl t1=t2=100; entry traded=1000, cost=1
    # net_pnl = [-1, 100, 100]; traded = [1000, 0, 0]
    assert np.isclose(m["margin"], 199 / 1000)          # 0.199
    assert np.isclose(m["daily_turnover"], (1000 / 3) / 1000)  # 0.3333
    assert np.isclose(m["max_drawdown"], 0.0)            # monotone up after entry
    assert np.isclose(m["pct_profitable_days"], 2 / 3)

    # engine mode returns the rich report dict
    rep = run(signal, prices, book_size=1000.0, cost_bps=10.0, mode="engine")
    for key in ("sharpe", "margin", "turnover", "max_drawdown"):
        assert key in rep

    # per-instrument cost (PDF model): B wider spread than A
    spread = pd.Series({"A": 2.0, "B": 20.0})
    m2 = run(signal, prices, book_size=1000.0, cost_bps=spread, mode="pdf")
    # entry cost = 500*2/1e4 + 500*20/1e4 = 0.1 + 1.0 = 1.1; net sum = 200 - 1.1
    assert np.isclose(m2["margin"], 198.9 / 1000)

    print("evaluate.quick_test passed")
    print("\npdf metrics (flat 10bps):")
    for k, v in m.items():
        print(f"  {k:22s} {v:.4f}")


if __name__ == "__main__":
    quick_test()

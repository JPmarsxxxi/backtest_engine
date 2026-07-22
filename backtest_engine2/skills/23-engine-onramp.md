# Skill 23 — Engine On-Ramp (the real verdict)

> **Read `PROTOCOL.md`, `00-overview.md`, `12-alpha-overview.md` first.** Costs **mandatory**, DSR **mandatory**.

> Skill 8 gives a quick single-path score. This stage wraps the finished alpha as a real `Strategy`, runs it through the full `Engine` (or `MultiPathEngine` for CPCV) with a real cost model, and deflates the Sharpe (DSR). **This is the verdict** that decides whether an alpha is real or overfit luck.

---

## When to load

- An alpha passed Skills 1–9 and you need the honest, purged, multiple-testing-aware result before promoting (Skill 22).

---

## The bridge: raw signal → engine weights

The pipeline carries a raw date × asset signal. The engine's `Strategy.generate_weights(data, t)` returns a **`pd.Series` of portfolio weights (fractions of equity)** for bar `t`. `source: backtest/strategy/base.py:33` Convert once, gross-normalised:

```python
# final_signal is the delay-aligned output of Skills 1–7 (already PIT-safe).
gross = final_signal.abs().sum(axis=1).replace(0, pd.NA)
weights = final_signal.div(gross, axis=0).fillna(0.0)   # gross leverage = 1 per bar
```

Because the signal was already delayed (Skill 2), row `t` uses only `≤ t-1` info — **set the panel `FieldSpec` lag to 0 so PIT is not applied twice** (Skill 2 note).

---

## Precomputed-weights Strategy

```python
import pandas as pd
from backtest.strategy.base import Strategy

class PrecomputedAlpha(Strategy):
    rebalance_frequency = "daily"          # source: backtest/strategy/base.py:26
    def __init__(self, weights: pd.DataFrame):
        self._w = weights
    def generate_weights(self, data, t):   # source: backtest/strategy/base.py:33
        if t not in self._w.index:
            return pd.Series(0.0, index=data.assets)
        return self._w.loc[t].reindex(data.assets).fillna(0.0)
```

---

## Wire up panel + cost + engine

```python
from backtest.data.panel import DataPanel, FieldSpec
from backtest.costs import CompositeCostModel, Spread, MarketImpact, Commission
from backtest.engine.engine import Engine

panel = DataPanel(
    prices, volume=volume, universe=mask,
    specs={"prices": FieldSpec(lag=0)},        # delay already applied in Skill 2
)                                              # source: backtest/data/panel.py:73
costs = CompositeCostModel([                   # mandatory — never zero-cost
    Spread(half_spread_bps=spread_bps),        # per-instrument half-spread (Ch.7)
    MarketImpact(...), Commission(...),
])                                             # source: backtest/costs/__init__.py:1
result = Engine(panel, PrecomputedAlpha(weights), costs=costs,
                initial_capital=1_000_000).run()   # source: backtest/engine/engine.py:154
rep = result.summary()                         # MetricsReport (IR, turnover, margin, maxdd)
```

---

## CPCV + DSR (the multiple-testing deflation)

A single path can still be lucky. Use the combinatorial purged splitter + `MultiPathEngine`, then deflate by the number of trials you actually ran:

```python
from backtest.selection.dsr import dsr
deflated = dsr(result.returns, K=n_trials_tested)   # source: backtest/selection/dsr.py:32
print("DSR:", round(deflated, 3))   # P(true Sharpe > luck-max); want it high
```
`K` = how many alpha variants you tested this session (lookbacks, methods…). DSR near 1 → robust; near 0.5 or below → likely overfit. (`MultiPathEngine` in `backtest/engine/multi_path.py` runs the purged combinatorial paths.)

---

## Validation checklist

- [ ] `FieldSpec` lag = 0 (delay already applied upstream) — no double PIT.
- [ ] Weights gross-normalised; `generate_weights` returns a Series indexed by asset.
- [ ] A real `CompositeCostModel` is attached (per-instrument spread), **not** zero cost.
- [ ] DSR computed with the honest `K` (every variant counts as a trial).
- [ ] Engine metrics (post-cost IR, turnover, margin, max_drawdown) reported, not the Skill 8 quick score.

---

## What NOT to do

- **Do not** present the Skill 8 quick score as the verdict — it has no purging or DSR.
- **Do not** run zero-cost; costs are mandatory (PROTOCOL).
- **Do not** under-count `K` — every lookback/method you tried inflates luck-max; honest `K` is the whole point of DSR.
- **Do not** re-apply PIT in the panel after the pipeline already delayed the signal.
- **Do not** feed un-normalised raw signal as weights — gross-normalise first.

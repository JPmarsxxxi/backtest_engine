# Skill 03 — Risk

> **Read `PROTOCOL.md` and `skills/00-overview.md` first.** Cell-loop and announcement rules in PROTOCOL apply.

---

## When to load

- **Stage 5** (risk overlay) — when the user wants per-asset / gross / net caps or vol targeting beyond the engine defaults.
- Any cell that sets the strategy's `risk = RiskConfig(...)` class attribute.
- Any cell that overrides `apply_risk` for custom logic.
- Any later cell that changes risk parameters (each change is a new strategy hash → a new trial; see `11-registry.md`).

Skip this skill **only** when the user has explicitly accepted the engine defaults and no risk parameter appears in the spec. The defaults are not zero — they cap per-asset at ±20% and gross/net at 1.0 (§3 below). If the spec implies anything beyond that, this skill is required.

---

## What you'll find here

1. `RiskConfig` — every field, default, type, source.
2. The default config (`DEFAULT_RISK_CONFIG`) — what it actually enforces.
3. The application order — why ordering matters and how constraints interact.
4. Each constraint in detail with the formula from source.
5. The drawdown sign convention (positive-magnitude inside the engine loop, negative in `BacktestResult.summary`).
6. Custom `apply_risk` overrides — when to write one.
7. Common mismatches and the questions to ask.
8. Minimal valid cell shape.
9. Validation checklist.
10. Anti-patterns.

---

## 1. `RiskConfig` — the contract

Frozen dataclass with six fields:

```python
@dataclass(frozen=True)
class RiskConfig:
    max_position:  Optional[float] = 0.20    # per-asset weight cap (both signs)
    max_gross:     float           = 1.0     # cap on sum(|weights|)
    max_net:       float           = 1.0     # cap on |sum(weights)|
    max_leverage:  Optional[float] = None    # alias for max_gross; if set, overrides
    target_vol:    Optional[float] = None    # annualized vol target
    vol_lookback:  int             = 60      # bars used for cov estimate
```
`source: backtest/risk/config.py`

| Field | Type | Default | Set to `None` to disable? |
|---|---|---|---|
| `max_position` | `float \| None` | `0.20` | Yes (`None` = no per-asset cap) |
| `max_gross` | `float` | `1.0` | **No** — float, can't be `None`; pick a large number to "disable" |
| `max_net` | `float` | `1.0` | **No** — same |
| `max_leverage` | `float \| None` | `None` | Yes (`None` = use `max_gross` instead) |
| `target_vol` | `float \| None` | `None` | Yes (default off) |
| `vol_lookback` | `int` | `60` | n/a (only used if `target_vol` is set) |

The dataclass is `frozen=True` — once constructed, values cannot be reassigned. To "tweak" a config, build a new one.

**There is intentionally no drawdown kill switch.** The engine used to have a `dd_breaker` field; it was removed because a kill switch hides exactly the recovery information a backtest exists to reveal. If a strategy needs DD-conditional sizing for live deployment, that belongs in a custom `Strategy.apply_risk` (§6) and stays out of the backtest config. The engine still populates `state["drawdown"]` for custom risk overrides to read (§5).

---

## 2. The default — what `DEFAULT_RISK_CONFIG` actually does

```python
DEFAULT_RISK_CONFIG = RiskConfig()
```
`source: backtest/risk/config.py:29`

That is **not the same as "no risk overlay"**. The default applies:

- `max_position = 0.20` — every weight is clipped to [-0.20, +0.20] per asset.
- `max_gross = 1.0` — gross exposure capped at 1.0 (no leverage).
- `max_net = 1.0` — net exposure capped at 1.0.
- `target_vol = None`, `max_leverage = None` — off.

A `Strategy` that doesn't set `risk` gets these. Surface this to the user explicitly the first time you construct a strategy:

> "Strategy will use `DEFAULT_RISK_CONFIG`: ±20% per asset, gross ≤ 1.0, net ≤ 1.0. The spec doesn't mention risk — confirm this is acceptable, or set a custom `RiskConfig`."

If the spec says "no constraints, just raw weights", you must explicitly set:
```python
risk = RiskConfig(max_position=None, max_gross=1e9, max_net=1e9)
```
There is no "off" preset.

---

## 3. Application order

```
1. target_vol     (best-effort multiplicative scale)
2. max_position   (per-asset clip)
3. max_net        (proportional rescale if violated)
4. max_gross      (proportional rescale if violated; max_leverage overrides)
```
`source: backtest/risk/config.py` (docstring), `backtest/risk/manager.py` (implementation)

The docstring says "later constraints win when they conflict." That means:

- **`target_vol` may scale weights *up*; `max_position` may then clip them back down.** A portfolio with realized vol of 5% and `target_vol=0.10` doubles all weights, then `max_position=0.20` clips. The realized vol of the final weights can be **less than the target**. This is a known asymmetry — surface it if the spec asks for hard vol targeting.
- **`max_net` and `max_gross` only rescale *down*.** They never widen weights. So if `target_vol` already scaled up beyond gross, the gross cap restores. But they can't enforce a *minimum* exposure.
- **`max_position` happens before `max_net` / `max_gross`.** A symmetric proportional rescale by `max_net / |net|` (factor < 1) can push some weights below their per-asset clip, which is fine. But if the rescale factor were > 1 (impossible here), it would push some above the cap. The order is intentionally chosen so the rescales only shrink.

If two constraints are mutually unsatisfiable (e.g. `max_position=0.05` × 3 assets = 0.15 max gross, but `max_gross=1.0` requires more), the binding constraint is the per-asset clip — gross is never reached.

---

## 4. Each constraint in detail

### 4.1 `target_vol` — annualized vol scale

```python
rets = data.prices.pct_change().tail(lookback)
if len(rets) < 2:
    return w                                           # no-op early in history
cov  = rets.cov().to_numpy() * _TRADING_DAYS           # annualized via × 252
w_vec   = w.reindex(cols).fillna(0.0).to_numpy()
port_var = float(w_vec @ cov @ w_vec)                  # w' Σ w
if not np.isfinite(port_var) or port_var <= 0:
    return w                                           # degenerate → no-op
port_vol = sqrt(port_var)
scale = target_vol / port_vol
return w * scale
```
`source: backtest/risk/manager.py` (`_scale_to_target_vol`). `_TRADING_DAYS = 252` at top of file.

Key facts:

- **Annualization is hard-coded to 252 days.** If the user is on weekly or monthly data, this is wrong — surface as a mismatch.
- **Sample covariance** over the trailing `vol_lookback` bars (default 60) of `pct_change(prices)`. No EWMA, no shrinkage, no mean-correction beyond what `pd.DataFrame.cov` does.
- **Bilateral scaling.** If realized vol < target, weights scale **up**. If > target, they scale **down**. There is no clipping inside `_scale_to_target_vol` — that's `max_position`'s job afterwards.
- **Early-history no-op.** Until there are at least 2 return rows, the function returns `w` unchanged. Combined with `vol_lookback=60`, that means target-vol scaling effectively kicks in once ~60 bars of history exist.
- **Degenerate no-op.** All-zero weights → port_var = 0 → no-op.
- **Cov uses `data.prices`, the strategy's `DataView`.** It is point-in-time correct (no lookahead) — the engine snapped the view at `t` before calling `apply_risk`. `source: backtest/engine/engine.py:208–216`.

### 4.2 `max_position` — per-asset clip

```python
if cfg.max_position is not None:
    w = w.clip(lower=-cfg.max_position, upper=cfg.max_position)
```
`source: backtest/risk/manager.py`

- Symmetric clip on both signs.
- Single scalar for **all assets** — there is no `max_position` dict for per-asset limits. If the spec needs `{"AAPL": 0.10, "ZM": 0.05}`, that requires a custom `apply_risk` (§6).
- `None` disables.

### 4.3 `max_net` — net exposure cap

```python
net = float(w.sum())
if cfg.max_net is not None and abs(net) > cfg.max_net and abs(net) > 0:
    w = w * (cfg.max_net / abs(net))
```
`source: backtest/risk/manager.py`

- Proportional rescale that brings `|sum(w)|` down to `max_net`.
- `max_net` is a `float` (not `Optional`) — but the code handles `None` (the typed default of `1.0` won't trigger that path; only a custom config with `max_net=None` would). If a user explicitly wants no net cap, set `max_net = 1e9`.
- `abs(net) > 0` guard prevents divide-by-zero on a flat portfolio.

### 4.4 `max_gross` / `max_leverage` — gross exposure cap

```python
gross_cap = cfg.max_leverage if cfg.max_leverage is not None else cfg.max_gross
gross = float(w.abs().sum())
if gross_cap is not None and gross > gross_cap and gross > 0:
    w = w * (gross_cap / gross)
```
`source: backtest/risk/manager.py`

- `max_leverage` overrides `max_gross` if set. They mean the same thing (sum of |weights| cap); `max_leverage` is just a more familiar alias.
- Proportional rescale, same shape as `max_net`.
- Final step in the pipeline.

---

## 5. Drawdown sign convention — for custom `apply_risk`

The engine still populates `state["drawdown"]` on every rebalance, even though no built-in constraint reads it. This is so a custom `apply_risk` override (§6) can implement DD-aware logic (e.g. continuous exposure scaling) without re-computing.

The convention inside the engine loop is **non-negative magnitude** — positive when in drawdown, 0 at peak:

```python
dd = (peak - equity_pre) / peak if peak > 0 else 0.0
```
`source: backtest/engine/engine.py:188–189`

So a custom override checking `state["drawdown"] >= 0.20` triggers at 20% drawdown or worse.

⚠️ Different convention used elsewhere: `BacktestResult.summary()` plots drawdown as **negative** (`(equity - peak) / peak`) — `source: backtest/engine/engine.py:71`. That's for visual display, not for risk decisions. The two conventions don't interact, but be aware when reading source.

---

## 6. Custom `apply_risk` — when to override

Use the default (set `risk = RiskConfig(...)`) **whenever the user's risk needs map onto the seven fields above.** If something doesn't fit, override `apply_risk` on the strategy class. See `02-strategy.md §7` for the override pattern.

Cases that **require an override** (cannot be expressed in `RiskConfig`):

- Per-asset limits varying by asset (`{"AAPL": 0.10, ...}`).
- Sector / industry / region exposure caps.
- Factor / beta neutralization.
- Volatility targeting on a different annualization (non-252).
- Volatility targeting using EWMA, shrinkage, or non-sample covariance.
- Per-position stop-losses or trailing stops.
- Risk parity / inverse-vol weighting on top of strategy weights.
- Hedging overlays (e.g. always hold a -X% futures position).
- Drawdown logic with hysteresis or partial flattening.

Pattern for "use default + add custom on top":

```python
def apply_risk(self, proposed, state, data):
    base = super().apply_risk(proposed, state, data)
    return my_extra_logic(base, state, data)
```

Pattern for "fully custom, no declarative":

```python
def apply_risk(self, proposed, state, data):
    # don't call super(); read state and data; return a Series
    ...
```

The override receives the same arguments as the default and must return a `pd.Series`. Engine reindexes to `assets_all` and fills NaN with 0 afterwards (`02-strategy.md §3`).

---

## 7. Common mismatches — surface these by default

Use the question shape from `PROTOCOL.md §3`. Engine-faithful default first.

| Spec says | Engine has | Question to ask |
|---|---|---|
| "Cap each name at 5%" | `max_position=0.05` (single scalar). | Use `max_position=0.05`. |
| "Cap AAPL at 10% but other names at 5%" | Single scalar only. | Custom `apply_risk` — write it explicitly, or relax to a single uniform cap. |
| "Sector exposure ≤ 30%" | Not in `RiskConfig`. | Custom `apply_risk` reading sector membership from a feature, or drop the constraint. |
| "Net exposure 0 (market neutral)" | `max_net` is a *cap*, not a *target* — won't enforce neutrality, only prevents going above. | Custom `apply_risk` that subtracts the mean weight, **or** rely on the strategy already producing zero-net weights. |
| "Annualized vol target 12%" on weekly data | Vol target hard-codes 252-day annualization. Weekly data → wrong scale. | Custom `apply_risk` with the right annualization, **or** acknowledge the misannualized number. |
| "EWMA / RiskMetrics covariance for vol targeting" | Sample cov, no decay. | Custom `apply_risk`, **or** accept sample cov. |
| "20% trailing stop per position" | Not in `RiskConfig`. | Custom `apply_risk` tracking entry prices on `self`, **or** drop. |
| "Halve exposure at 10% DD" / any DD-conditional sizing | **No built-in DD logic.** Engine deliberately omits a kill switch (§1) but exposes `state["drawdown"]` (§5) for custom logic. | Custom `apply_risk` reading `state["drawdown"]`, **or** drop the DD overlay. Do not re-introduce a kill switch into the backtest. |
| "No risk constraints — raw weights" | `DEFAULT_RISK_CONFIG` enforces per-asset 20% / gross 1 / net 1. | Set `risk = RiskConfig(max_position=None, max_gross=1e9, max_net=1e9)` explicitly — there is no "off" preset. Confirm with user. |
| "Beta neutralize against SPY" | Not in `RiskConfig`. | Custom `apply_risk` computing rolling beta from a feature, **or** drop. |
| "Volatility scale, but never amplify (only shrink)" | `target_vol` is bilateral — can scale up. | Custom `apply_risk` with one-sided scaling, **or** accept bilateral. |
| Per-asset short-borrow limits | Not in `RiskConfig`. (`ShortBorrow` cost component is unrelated — see `04-costs.md`.) | Custom `apply_risk` checking per-asset borrow caps, **or** drop. |

---

## 8. Minimal valid cell — Stage 5

This is the shape of a Stage 5 cell *if the user wants to change risk*. If the user is fine with `DEFAULT_RISK_CONFIG`, **skip this stage entirely** and proceed to Stage 6 (multi-path).

```python
# Stage 5 — risk overlay. Tweak limits, then re-run the engine for comparison.
from backtest.risk import RiskConfig

class UserStrategyRisk(UserStrategy):
    risk = RiskConfig(
        max_position=0.20,        # ±20% per asset
        max_gross=1.0,            # gross ≤ 1.0
        max_net=1.0,              # |net| ≤ 1.0
        target_vol=0.10,          # 10% annualized vol target (252-day)
        vol_lookback=60,          # 60-bar covariance window
    )
    def __repr__(self):
        return f"UserStrategyRisk(lookback={self.lookback})"

result_risk = Engine(panel, UserStrategyRisk(60), costs=costs, liquidity=liq).run(
    start=panel.dates[252],
)

ann_vol_default = float(result_costs.returns.std() * (252 ** 0.5))
ann_vol_risk    = float(result_risk.returns.std()  * (252 ** 0.5))
print(f"Realized ann. vol — default risk: {ann_vol_default:.2%}")
print(f"Realized ann. vol — overlay:      {ann_vol_risk:.2%}")
```

Note that subclassing the prior strategy preserves `generate_weights` and the rebalance frequency — only `risk` and `__repr__` need to change.

---

## 9. Validation checklist (after the cell runs)

After the user pastes the output:

- [ ] **Realized ann. vol with `target_vol=X`** lands close to `X` (within ~1–2 percentage points). Wildly off → vol target can't be met because `max_position` is binding (§3 / §4.1). Surface it.
- [ ] **Per-asset weight magnitudes** in `result_risk.weights.abs().max().max()` ≤ `max_position` + 1e-9. If not, the engine's float coercion is suspect (it shouldn't happen).
- [ ] **Gross exposure** `result_risk.weights.abs().sum(axis=1).max()` ≤ `max_gross` (or `max_leverage`).
- [ ] **Net exposure** `result_risk.weights.sum(axis=1).abs().max()` ≤ `max_net`.
- [ ] **Max equity drawdown** in `result_risk` is **not bounded** by any RiskConfig field — that's by design. Compare it to the no-overlay run and report how the risk overlay changed the realized DD, but don't expect any specific cap.

If any fails: state which, propose a `RiskConfig` adjustment as a new cell. Do not silently retune.

---

## 10. What NOT to do

- **Do not assume "no risk = `DEFAULT_RISK_CONFIG`".** The default is restrictive (§2). Surface it.
- **Do not pass `max_gross=None` or `max_net=None`.** Their type is `float`, not `Optional[float]`. To "disable" them, use a large number like `1e9`.
- **Do not assume `target_vol` will hit its target.** It can be capped by `max_position` (§3, §4.1).
- **Do not assume drawdown sign convention** without re-reading §5. It's positive inside the loop and negative in `summary()`. Different code paths, different signs.
- **Do not try to add a `dd_breaker` argument to `RiskConfig`.** It was deliberately removed (§1, §7). If a strategy needs DD-conditional sizing, it goes in a custom `apply_risk` override and the user accepts that this changes what the backtest measures.
- **Do not implement vol targeting in `generate_weights`** by hand if `RiskConfig.target_vol` would suffice. Use the declarative form.
- **Do not mutate the `RiskConfig`** after construction — it's `frozen`. Build a new one.
- **Do not put per-asset caps as a dict in `max_position`.** It's a single scalar. Per-asset limits require a custom `apply_risk` (§6).
- **Do not change `_TRADING_DAYS`** in `manager.py`. If you need a different annualization, override `apply_risk` and call your own scaler — don't patch the engine to suit one strategy.
- **Do not forget that risk runs *after* eligibility filtering.** Ineligible assets are zeroed before `apply_risk`. Custom risk overrides should expect the proposed Series may already have zeros for delisted names.

# Skill 04 — Costs and Liquidity

> **Read `PROTOCOL.md` and `skills/00-overview.md` first.** Cell-loop and announcement rules in PROTOCOL apply.

> Reminder from PROTOCOL §6: **costs are mandatory**. Stage 3 (configuring costs + liquidity) must happen before Stage 4 (single-path engine run). No first-look "gross" run unless the user explicitly opts in.

---

## When to load

- **Stage 3** of every backtest (configuring `CompositeCostModel` + `LiquidityCap`).
- Any later cell that adds, removes, or retunes a cost component.
- Any cell that introduces a strategy needing market impact / borrow / liquidity throttling.

---

## What you'll find here

1. The cost interface (`CostModel` ABC).
2. When the engine charges trade-cost vs holding-cost.
3. `Commission` — bps or per-share, with formulas.
4. `Spread` — scalar / dict / Series of per-asset half-bps (constant in time).
4b. `RealizedSpread` + `spread_source` — TIME-VARYING per-bar half-spread from real bid/ask or an OHLC estimate. **Mandatory user choice: measured vs estimated.**
5. `MarketImpact` — sqrt or linear, ADV-based.
6. `ShortBorrow` — annualized rate on shorts, charged every bar.
7. `CompositeCostModel` — additive composition.
8. `LiquidityCap` — the orthogonal trade throttle (not a cost).
9. Volume requirements — which components need it, how each behaves when it's missing.
10. Common mismatches and the questions to ask.
11. Minimal valid cell.
12. Validation checklist.
13. Anti-patterns.

---

## 1. The cost interface

```python
class CostModel(ABC):
    @abstractmethod
    def trade_cost(self, trades: pd.Series, view: DataView) -> float: ...

    def holding_cost(self, positions: pd.Series, view: DataView) -> float:
        return 0.0
```
`source: backtest/costs/base.py:11–18`

| Argument | What it is |
|---|---|
| `trades` | **Signed dollar** amounts. Positive = buy (long open or short cover). Negative = sell (long close or short open). Indexed by asset. |
| `positions` | **Signed dollar** exposures. Positive = net long. Negative = net short. Indexed by asset. |
| `view` | `DataView` at the current bar — used for the last price (`view.prices.iloc[-1]`) and recent volume (`view.volume.tail(...)`). |

Return value: a single **float** (total dollar cost for this bar across all assets — not per-asset).

Two constants used throughout `backtest/costs/components.py`:
- `_BPS = 10_000.0` — `bps=2` means `2 / 10_000 = 0.0002` of notional.
- `_TRADING_DAYS = 252` — used by `ShortBorrow` for the daily-rate conversion.

---

## 2. When the engine charges what

From `Engine.run` (`source: backtest/engine/engine.py:191–252`):

| Cost type | Charged on | Inputs |
|---|---|---|
| `holding_cost` | **Every bar** (not just rebalance bars) | current `positions` series, current `view` |
| `trade_cost`   | Only on rebalance bars, **after** liquidity throttling | the *executed* trades (not what the strategy proposed), current `view` |

Cash bookkeeping (per bar, in order):
1. `cash -= holding_cost` (every bar)
2. If rebalance bar:
   - `executed` = strategy weights → target dollars → trades, throttled by `LiquidityCap`
   - `cash -= executed.sum() + trade_cost` (line 245)

This means **`trade_cost` sees `executed`, not `proposed_trades`** — if liquidity capped your trade to 10% of what you wanted, you pay commission/spread/impact on 10%, not 100%. Don't model partial-fill rebates.

---

## 3. `Commission`

```python
Commission(bps=None, per_share=None)
```
**Pass exactly one.** Both or neither → `ValueError`. `source: backtest/costs/components.py:22–26` and `costs_test.py:56–60`.

### 3.1 Bps mode

```python
cost = |trades|.sum() * bps / 10_000
```
`source: backtest/costs/components.py:28–30`

- Flat across assets.
- No price lookup — depends only on dollar notional.
- `bps=2` means 2 basis points = 0.0002.

### 3.2 Per-share mode

```python
prices = view.prices.iloc[-1].reindex(trades.index)
shares = (|trades| / prices).replace([±inf], NaN).fillna(0)
cost   = shares.sum() * per_share
```
`source: backtest/costs/components.py:31–33`

- Uses the **last available price in the view** (i.e. at `view.t`).
- Asset with NaN price → 0 shares → no cost (silent zero, see §10).
- `per_share=0.005` means half a cent per share.

---

## 4. `Spread`

```python
Spread(half_bps)
```
where `half_bps` is one of:
- `float` / `int` — uniform across all assets,
- `dict[str, float]` — per-asset; missing assets → 0,
- `pd.Series` — per-asset; missing assets → 0.

`source: backtest/costs/components.py:36–46`

### 4.1 Scalar

```python
cost = |trades|.sum() * half_bps / 10_000
```

### 4.2 Per-asset (dict or Series)

```python
rates = Series(half_bps).reindex(trades.index).fillna(0.0) / 10_000
cost  = (|trades| * rates).sum()
```

- "Half-spread" because the convention is that the full bid-ask spread is `2 × half_bps`, and a market order eats half of it.
- A name not in the dict pays **zero spread** — flag this if it's a surprise (§10).

---

## 4b. `RealizedSpread` + `spread_source` — time-varying spread

`Spread` (§4) is **constant** per asset. `RealizedSpread` charges a **per-bar, per-asset** half-spread — the point-in-time analogue — sourced from real quotes or an OHLC estimate. Build the input frame with `spread_source`, then feed it to the model.

```python
from backtest.costs import RealizedSpread, spread_source as ss

# 1) source a per-bar half-spread frame (date x asset, in bps):
half_bp = ss.from_bidask(bid, ask)                                    # MEASURED (real quotes)
half_bp = ss.estimate(high, low, close, open_, method="abdi_ranaldo", window=21)  # ESTIMATED (OHLC)

# 2) consume it EITHER as a panel feature named 'spread' (attach in Stage 1) ...
rs = RealizedSpread(field="spread")
# ... OR pass the frame directly:
rs = RealizedSpread(half_bps=half_bp)
```
`cost = |trades| * half_bp(t) / 1e4` per bar (half-spread convention); reads PIT (last row ≤ `view.t`). `source: backtest/costs/components.py:RealizedSpread`, `source: backtest/costs/spread_source.py:from_bidask`, `…/spread_source.py:estimate`

### ⚠️ MANDATORY DECISION — always ask which spread source, and state pros/cons

Whenever spread matters, **stop and ask the user which of the two sources to use** in the cell announcement (per PROTOCOL §3–4). Never silently default. Present both with their trade-offs:

| Source | Pros | Cons |
|---|---|---|
| **1. Measured** — `from_bidask(bid, ask)` | Exact, true point-in-time spread; captures real session / rollover / vol flares (this is why #016's flat assumption was 2–7× too low). | Needs real bid/ask: **have it for FX (Dukascopy / MT5 ticks) and crypto**; for equities only via a **paid** quote feed (Polygon / Databento / TAQ). |
| **2. Estimated** — `estimate(OHLC, method=…)` | Works from OHLC you already have (e.g. yfinance equities); no quote feed. | An *estimate*: noisy per name/day → use a rolling `window`; and **daily only** — there is no accurate intraday equity spread estimate. |

If **Estimated**, ask a **second** question — which estimator (accuracy vs dependency):
- `edge` — Ardia, Guidotti & Kroencke (2024). Most accurate from OHLC. **Requires `pip install bidask`** (lazy-imported; only needed for this method).
- `abdi_ranaldo` — robust, few negatives, **zero-dependency default**.
- `corwin_schultz` — classic two-day high-low.
- `roll` — close-only, noisy; fallback.

Do not choose for the user. Surface the fork, state pros/cons, wait for go-ahead.

### Attaching as a panel feature
For `RealizedSpread(field="spread")`, attach the frame in Stage 1: `DataPanel(prices, volume=vol, features={"spread": half_bp})`. Or skip the feature and pass `RealizedSpread(half_bps=half_bp)` directly.

---

## 5. `MarketImpact`

```python
MarketImpact(k, kind="sqrt", adv_lookback=20)
```

- `kind` must be `"sqrt"` or `"linear"` — else `ValueError`. `source: backtest/costs/components.py:59–60`
- **Requires `view.volume`** — raises `ValueError("MarketImpact requires volume on the data panel")` if absent. `source: backtest/costs/components.py:66–67`

### 5.1 Formulas

```python
adv_shares    = view.volume.tail(adv_lookback).median()
adv_notional  = (adv_shares * view.prices.iloc[-1]).reindex(trades.index)
ratio         = (|trades| / adv_notional).replace([±inf], NaN).fillna(0)

if kind == "sqrt":
    impact_bps = k * sqrt(ratio)
else:  # linear
    impact_bps = k * ratio

cost = (|trades| * impact_bps / 10_000).sum()
```
`source: backtest/costs/components.py:65–76`

### 5.2 Worked example

From the test (`costs_test.py:99–106`): `MarketImpact(k=10, kind="sqrt")`, asset A with 1M shares ADV at $100 → ADV_notional = $100M. A $1M trade → ratio = 0.01 → impact_bps = `10 * sqrt(0.01) = 1.0` bps → cost = `1_000_000 * 1.0 / 10_000 = $100`.

### 5.3 Semantics to flag

- **Volume is interpreted as shares**, not dollars. If the user has dollar-volume data, they must convert it before constructing the panel.
- **ADV uses the trailing `adv_lookback` bars and takes the median**, not the mean — robust to spikes. Default 20 bars.
- **NaN ADV (e.g. asset never traded) → ratio = 0 → cost = 0** for that asset. Silent zero — surface if the universe includes thin names.
- **sqrt impact scales sub-linearly** (the typical Almgren-style assumption). Doubling trade size → impact grows by `sqrt(2) ≈ 1.41×`, but total cost ≈ `2 × 1.41 = 2.83×`. The test at `costs_test.py:116–120` confirms this scaling.

---

## 6. `ShortBorrow`

```python
ShortBorrow(annual_bps, trading_days=252)
```

```python
self.daily_rate = annual_bps / 10_000 / trading_days

def trade_cost(self, trades, view):
    return 0.0                               # no per-trade borrow

def holding_cost(self, positions, view):
    shorts = positions.clip(upper=0.0).abs()  # extract only the negative side
    return float(shorts.sum() * self.daily_rate)
```
`source: backtest/costs/components.py:79–90`

- **Charged every bar** the position is held (it's a `holding_cost`), not just rebalance bars. `source: backtest/engine/engine.py:194–198`.
- **Only on shorts** (`positions.clip(upper=0.0)`). Long-only portfolios pay zero. Tested at `costs_test.py:128–139`.
- `trading_days=252` is the conversion factor. If the user is on weekly or monthly bars, this annualization is wrong — surface as a mismatch and pass a custom `trading_days` (`trading_days=52` for weekly, `12` for monthly). `source: backtest/costs/components.py:82`
- No per-asset rate. If the user needs differentiated borrow (hot stocks 50 bps, locates 800 bps), compose two `ShortBorrow` instances inside a `CompositeCostModel`, each with a different `annual_bps`, and split the positions across them — or write a custom `CostModel`.
- **Negative `annual_bps` is allowed and models a net rebate.** Passing `annual_bps=-50` makes `holding_cost` negative, so the engine's `cash -= hc` adds cash on shorts (roughly +0.5% per year). This approximates the real-world scenario where the rebate on the short-sale collateral exceeds the borrow fee — common for easy-to-borrow names. Use sparingly and only as a single-number net approximation: it ignores rate-environment dependence, mixes borrow fee and rebate into one parameter, and is wrong for hard-to-borrow names. Surface the assumption to the user before applying.

---

## 7. `CompositeCostModel`

```python
CompositeCostModel([model_a, model_b, ...])
```

```python
def trade_cost(self, trades, view):
    return sum(m.trade_cost(trades, view) for m in self.models)

def holding_cost(self, positions, view):
    return sum(m.holding_cost(positions, view) for m in self.models)
```
`source: backtest/costs/base.py:21–31`

- Simple additive sum — no interaction terms, no precedence.
- All component types may be mixed (commission + spread + impact + borrow).
- This is the canonical way to combine cost components; do not write a strategy that calls each cost model directly.

---

## 8. `LiquidityCap` — the orthogonal throttle

```python
LiquidityCap(cap_pct, adv_lookback=20)
```

**Important: this is *not* a `CostModel`.** It's a separate object the engine consumes as a distinct argument (`Engine(panel, strat, costs=..., liquidity=...)`).

```python
adv_shares    = view.volume.tail(adv_lookback).median()
adv_notional  = (adv_shares * view.prices.iloc[-1]).reindex(trades.index)
cap           = (cap_pct * adv_notional).fillna(inf)

over           = |trades| > cap
executed       = trades.copy()
executed[over] = sign(trades[over]) * cap[over]
unfilled       = trades - executed
return executed, unfilled
```
`source: backtest/costs/liquidity.py:22–39`

- `cap_pct` must be in `(0, 1]`. Else `ValueError`. `source: backtest/costs/liquidity.py:17–18`
- Returns `(executed, unfilled)` — both `pd.Series` of dollars. Engine uses `executed`; `unfilled` ends up in `BacktestResult.unfilled`.
- **Sign preserved** — large short orders get capped to the negative side, not flipped (`costs_test.py:176–180`).
- **NaN cap (no ADV for an asset) → inf cap → no throttling.** That's a deliberate passthrough but can hide a thin-name problem.
- **Volume requirement enforced at `Engine.__init__`.** If you pass `LiquidityCap` to an `Engine` whose panel lacks volume, construction raises `ValueError("LiquidityCap was supplied but the panel has no volume. ...")` immediately — you don't reach `.run()`. `source: backtest/engine/engine.py` (Engine constructor, after the `required_data` check).
- The `LiquidityCap.apply` method itself still does silent passthrough when handed a no-volume `view`, for direct/manual use. The engine just prevents the combination ever being built.

---

## 9. Volume requirements — at a glance

| Component | Needs `view.volume`? | What happens without it |
|---|---|---|
| `Commission(bps=...)` | No | n/a |
| `Commission(per_share=...)` | No (uses prices only) | n/a |
| `Spread` | No | n/a |
| `MarketImpact` | **Yes** | **Raises `ValueError`** at first `trade_cost` call (mid-run) |
| `ShortBorrow` | No | n/a |
| `LiquidityCap` | **Yes** | **Raises `ValueError`** at `Engine.__init__` (before any run) |

If the user has no volume DataFrame: either drop `MarketImpact` and `LiquidityCap`, or insist they provide volume. The engine catches `LiquidityCap` immediately; `MarketImpact` fails on the first rebalance. Either way, surface the choice up front — don't quietly drop components.

---

## 10. Common mismatches — surface these by default

Use the question shape from `PROTOCOL.md §3`. Engine-faithful default first.

| Spec says | Engine has | Question to ask |
|---|---|---|
| "Realistic costs" with no numbers | No defaults to fall back on. | Propose specific numbers (e.g. `Commission(bps=2) + Spread(half_bps=2) + MarketImpact(k=10, kind="sqrt") + ShortBorrow(annual_bps=300)`) and ask user to confirm. |
| "Use IB / brokerage commission schedule" | Only flat bps or flat per-share. | Approximate with one of the two; ask which closer matches the schedule, and surface the deviation. |
| Tiered commissions (cheaper above N shares) | Not supported. | Single rate average, **or** custom `CostModel` (subclass; ask before writing). |
| Per-asset borrow tiers | `ShortBorrow` is uniform. | One blended rate, **or** compose multiple `ShortBorrow` instances at the cost of complexity, **or** custom `CostModel`. |
| Bid-ask in absolute cents / dollars | `Spread.half_bps` is bps only. | Convert to bps at the typical price (caveat: bps drifts with price), **or** custom `CostModel`. |
| Per-asset **time-varying** half-spread (real bid-ask feed, or per-bar) | **`RealizedSpread` + `spread_source`** (§4b) — per-bar, PIT. Supersedes the old "constant per asset" limitation. | Ask MEASURED (`from_bidask`) vs ESTIMATED (`estimate`) per §4b's mandatory decision. |
| Spread for equities with no quotes (yfinance) | `spread_source.estimate(OHLC, method=…)` — EDGE / Abdi-Ranaldo / Corwin-Schultz / Roll. | Which estimator? (accuracy vs `bidask` dependency, §4b). Note: daily only. |
| Linear impact (no sqrt) | `MarketImpact(kind="linear")`. | Use `kind="linear"`. |
| Almgren-Chriss impact with permanent + temporary components | Single-term sqrt or linear only. | Single-term approximation, **or** custom `CostModel`. |
| Stock-borrow rebate (positive cash on hard-to-borrow shorts) | Not modeled. | Drop, **or** approximate as negative `annual_bps` (which becomes cash income — verify this is what the user wants). |
| Margin interest on leverage | Not modeled. | Add a custom `CostModel.holding_cost` that charges on `positions.sum() - equity` when negative, **or** acknowledge gross leverage cost is missing. |
| "Costs only on entry, not exit" | `trade_cost` runs on every executed dollar regardless of direction. | Cannot do without custom `CostModel`. Surface as out-of-scope. |
| `LiquidityCap` set but no volume DataFrame on panel | `Engine.__init__` raises `ValueError` immediately (§8). | Provide volume, **or** drop `LiquidityCap`. There is no in-between. |
| MarketImpact with weekly / monthly data | `_TRADING_DAYS=252` is in `ShortBorrow`, not `MarketImpact` — impact itself doesn't annualize. But `adv_lookback=20` bars means ~5 months on weekly, ~20 months on monthly. | Reduce `adv_lookback` to a sensible window for the data frequency. |
| Want to disable `LiquidityCap` for a comparison | Pass `liquidity=None` to `Engine`. | Confirm comparison is intentional. |

---

## 11. Minimal valid cell — Stage 3

This is the shape of Stage 3 (configure costs + liquidity). Adapt the numbers to the user's spec.

```python
from backtest.costs import (
    Commission, Spread, MarketImpact, ShortBorrow,
    CompositeCostModel, LiquidityCap,
)

costs = CompositeCostModel([
    Commission(bps=2),                                      # 2 bps commission
    Spread(half_bps=2),                                     # 2 bps half-spread
    MarketImpact(k=10, kind="sqrt", adv_lookback=20),       # sqrt impact, 20-bar ADV
    ShortBorrow(annual_bps=300, trading_days=252),          # 3% annual borrow on shorts
])

liq = LiquidityCap(cap_pct=0.10, adv_lookback=20)           # 10% of ADV cap

# Quick sanity: print what we configured.
print("Cost components:", [type(m).__name__ for m in costs.models])
print(f"LiquidityCap: cap_pct={liq.cap_pct}, adv_lookback={liq.adv_lookback}")
print(f"Panel has volume: {panel.has_field('volume')}")  # MUST be True if MarketImpact or LiquidityCap is used
```

Two important constructions to flag for the user before writing the cell:

1. **`MarketImpact` requires volume** — `Engine.run()` will raise mid-loop if the panel doesn't have it.
2. **`LiquidityCap` requires volume** — `Engine.__init__` will raise immediately if the panel doesn't have it.

The `has_field('volume')` print at the bottom is a pre-flight check: it catches both issues *before* you try to construct the engine in Stage 4, so the user can fix Stage 1 without seeing an error stack first.

If `panel.has_field('volume')` is `False` and the user wants to keep costs as designed, this is a hard stop — Stage 1 (data) needs to be redone with volume.

---

## 12. Validation checklist (after the cell runs)

After the user pastes the output:

- [ ] **`Cost components` print** lists exactly what you expected. No surprise additions, no missing components.
- [ ] **`Panel has volume`** is `True` if `MarketImpact` or `LiquidityCap` is in the config. If `False`, **stop** — Stage 1 needs volume.
- [ ] **No `ValueError`** raised. The two common ones: `MarketImpact` without volume, `LiquidityCap` with `cap_pct ∉ (0, 1]`.

The real validation of cost magnitude happens after Stage 4 (the engine run). At that point, check:

- [ ] `result.costs.sum() / result.equity_curve.iloc[0]` is in a plausible range (rough order: 5–50 bps/year for a low-turnover strategy with these settings, more for high-turnover).
- [ ] `result.costs > 0` on rebalance bars (assuming the strategy actually trades).
- [ ] `result.unfilled.abs().sum().sum() > 0` only if `LiquidityCap` actually capped something. If it's zero with `LiquidityCap` enabled, either (a) trades were always small enough — fine, or (b) the cap is silently inert (no volume — §8) — bad.

---

## 13. What NOT to do

- **Do not skip Stage 3.** Costs are mandatory (`PROTOCOL.md §6`). If the user hasn't given numbers, propose specific values and ask.
- **Do not pass `costs=None` to `Engine.run()`.** The engine accepts it; PROTOCOL forbids it.
- **Do not invent a "default" cost preset.** There isn't one. Each spec needs explicit numbers from the user, or you propose and confirm.
- **Do not combine costs by calling them directly from the strategy.** Compose them with `CompositeCostModel` and let the engine call them.
- **Do not use `MarketImpact` without verifying the panel has volume.** It will raise mid-run.
- **Do not use `LiquidityCap` without verifying the panel has volume.** The engine refuses to build — fix Stage 1 first.
- **Do not treat `LiquidityCap` as a `CostModel`.** It's a separate parameter to `Engine`, not part of `CompositeCostModel`.
- **Do not assume `_TRADING_DAYS=252` is right** for non-daily data. Adjust `trading_days` in `ShortBorrow`.
- **Do not pass `LiquidityCap(cap_pct=0)` or `cap_pct > 1`.** Both raise `ValueError`.
- **Do not interpret cost components per-asset.** They return a single float — total dollar cost for the bar.
- **Do not double-count trades.** `trade_cost` is called on the *executed* trades after liquidity throttling, so partial fills are already accounted for. Don't model an extra "missed-trade slippage" cost unless explicitly asked.
- **Do not subclass `CostModel` without asking.** New cost types are an engine extension; propose via the `PROTOCOL §7` three-option pattern.
- **Do not silently pick a spread source.** If using `RealizedSpread`, always ask the user MEASURED vs ESTIMATED (and which estimator) with pros/cons in the announcement — §4b's mandatory decision.
- **Do not use `RealizedSpread` with an OHLC estimate at intraday frequency for equities.** The estimators are daily; intraday needs real quotes. Surface this.

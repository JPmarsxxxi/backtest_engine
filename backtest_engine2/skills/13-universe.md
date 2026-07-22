# Skill 13 — Universe (alpha pipeline Stage 1)

> **Read `PROTOCOL.md`, `00-overview.md`, and `12-alpha-overview.md` first.** Cell-loop and announcement rules apply.

> Stage 1 of the alpha pipeline. This cell fetches price/volume and builds the tradable universe mask. Everything downstream acts only on the in-universe instruments.

---

## When to load

- **Stage 1** of an alpha hunt: building the universe before any signal exists.
- Any cell that constructs or restricts the tradable set by liquidity or category.

---

## API

```python
from backtest.alpha_pipeline import universe

mask = universe.run(
    prices,             # pd.DataFrame, date × asset
    volume,             # pd.DataFrame, same index & columns (coin/share volume)
    top_n,              # int — keep the N most liquid per bar
    metadata=None,      # pd.DataFrame indexed by asset; cols asset_class/region/sector
    asset_class=None,   # str | sequence — categorical pre-filter
    region=None,
    sector=None,
    adv_window=30,      # rolling window (bars) for average dollar volume
) -> pd.DataFrame       # boolean date × asset mask; True == in-universe
```
`source: backtest/alpha_pipeline/universe.py:run`

---

## Math (PDF)

- **Liquidity** = rolling `adv_window`-bar average **dollar** volume: `adv[t,i] = mean(price · volume over t-29..t)`; keep the top-N per bar. PDF Ch. 31 defines the universe as *"the top most-liquid stocks… determined by the highest average daily dollar volume traded"* — dollar volume, not share volume.
- **Categorical filters** (asset class / region / sector) restrict the eligible set *before* the liquidity rank, so "top N" is taken within the chosen slice. PDF Ch. 4 "Alpha Universe" lists exactly these restriction dimensions.
- `min_periods=1`: early bars rank on a partial window (no all-False gap at the start). `source: backtest/alpha_pipeline/universe.py:run`
- Ties at the N-th rank are **kept** (`method="min"`), so the mask may exceed `top_n` on tie bars.

---

## Engine fit

The boolean mask is the exact type `DataPanel(universe=...)` expects (`source: backtest/data/panel.py:78`), consumed by `DataView.assets` (`source: backtest/data/panel.py:47–56`). **No adapter needed** — Stage 1 output plugs straight into the panel, and also gates which columns Stages 2–9 act on.

---

## Mismatches with the spec — surface these (PROTOCOL §3 shape)

| Idea says | Pipeline has | Question to ask |
|---|---|---|
| "Top 50 coins" | top-N by 30-bar ADV. | Confirm N and that liquidity = dollar volume (price×volume), not share volume. |
| "Only L1 coins" / "US stocks" | categorical pre-filter needs a static `metadata` table (asset → sector/region/class). | Do you have metadata? Time-varying membership is deferred (survivorship; Ch. 10). |
| "Survivorship-free universe" | universe is point-in-time by construction (mask per bar), but **delisting/terminal-knowledge is the panel's job**, not this skill. | Surface; do not drop NaN columns manually. |
| Fundamental/illiquid universe | thin coins have huge spreads → cost destroys the alpha (Ch. 7). | Flag that a wide, illiquid universe needs the per-instrument cost in Skill 8. |

Always offer the closest-existing default first.

---

## Minimal valid cell

```python
from backtest.alpha_pipeline import universe

# prices, volume already fetched (date × asset, identical shape)
mask = universe.run(prices, volume, top_n=50)

print("universe shape:", mask.shape)
print("avg names in-universe per bar:", mask.sum(axis=1).mean().round(1))
print(mask.iloc[-1][mask.iloc[-1]].index.tolist()[:10], "...")
```

---

## Validation checklist (after the cell runs)

- [ ] `mask.shape == prices.shape` and dtype is bool.
- [ ] `mask.sum(axis=1)` ≈ `top_n` after warmup (a little higher on tie bars is fine).
- [ ] No all-False bars after the first `adv_window` rows (would mean no tradable universe).
- [ ] If a categorical filter was used, no out-of-group asset is ever True.

---

## What NOT to do

- **Do not** rank by share volume — use **dollar** volume (price × volume). A cheap coin with huge unit volume is not necessarily liquid in dollars.
- **Do not** hand-drop "dead" coins or late-NaN columns — survivorship/delisting is the panel's responsibility, not the universe filter.
- **Do not** apply a categorical filter without `metadata` — it raises by design.
- **Do not** carry the universe forward silently into a fundamental/illiquid set without flagging the cost implication for Skill 8.

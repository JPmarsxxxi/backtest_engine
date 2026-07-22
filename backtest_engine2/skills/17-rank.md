# Skill 17 — Rank / winsor+scale (alpha pipeline Stage 5)

> **Read `PROTOCOL.md`, `00-overview.md`, `12-alpha-overview.md` first.** Cell-loop and announcement rules apply.

> Stage 5. Cross-sectionally **winsorise** each bar's signal to the `[low_pct, high_pct]` band, then **linearly rescale** the band to `[-1, +1]`. Magnitudes inside the band survive; only the tails are clipped. **Execution order: this runs *before* neutralisation (Skill 4), per Ch. 5.**

> Replaces the prior pure-ordinal rank (which destroyed all magnitude). Defaults reproduce a standard 5/95 winsorisation; set `low_pct=0.0, high_pct=1.0` to disable winsorisation entirely (pure min-max rescale).

---

## When to load

- **Stage 5 (runtime: before neutralise)**: bound the signal to `[-1, +1]` cross-sectionally while preserving relative magnitudes inside the central band. If you do *not* want magnitudes flattened to ordinal positions, this is the stage that respects that.

---

## API

```python
from backtest.alpha_pipeline import rank

ranked = rank.run(sig)                                 # default 5/95 winsor
ranked = rank.run(sig, low_pct=0.01, high_pct=0.99)     # tighter band (less clipping)
ranked = rank.run(sig, low_pct=0.0, high_pct=1.0)       # no winsorisation (pure min-max)
```
`run(signal, low_pct=0.05, high_pct=0.95) -> pd.DataFrame` (values in `[-1, 1]`). `source: backtest/alpha_pipeline/rank.py:run`

---

## Math (PDF Ch. 5 `Alpha3 = rank(Alpha1)`; Ch. 12; standard winsorisation)

For each bar `t` with `N_t` valid (non-NaN) values:

```
P_low(t)  = quantile_{low_pct}(signal_t)
P_high(t) = quantile_{high_pct}(signal_t)
clipped(i, t) = clip(signal(i, t), P_low(t), P_high(t))
out(i, t) = 2 * (clipped(i, t) - P_low(t)) / (P_high(t) - P_low(t)) - 1
```

- Smallest valid signal in the band → **−1**; largest in the band → **+1**; magnitudes **inside** the band are linearly preserved.
- Values **outside** the band are pinned to `±1` (that's the winsorisation tail). Their original magnitude beyond the band is *not* encoded.
- Edge cases: `N_t == 0` → all NaN; `N_t == 1` → 0.0 (no relative magnitude); degenerate bar with `P_low == P_high` → 0.0 for all valid entries. NaN preserved. `source: backtest/alpha_pipeline/rank.py:run`
- Ch. 5 ranks *then* neutralises — that's why this precedes Skill 4.

---

## Why winsor+scale instead of ordinal rank

| | Ordinal rank | Winsor + scale |
|---|---|---|
| Input `[1, 2, 3]` | `[-1, 0, +1]` | `[-1, 0, +1]` |
| Input `[1, 100, 1000]` | `[-1, 0, +1]` | `[-1, 0, +1]` (default 5/95 → endpoints clipped) |
| Input `[1, 100, 1000]` w/ `0/1` pct | `[-1, 0, +1]` | `[-1, -0.802, +1]` (raw magnitudes preserved) |
| Input `[1, 2, 3, 4, 5]` evenly spaced | `[-1, -0.5, 0, 0.5, 1]` | evenly spaced in `[-1, +1]` (interior preserved) |
| Input `[1, 5, 5, 5, 9]` (clustered) | `[-1, 0, 0, 0, +1]` | most around 0, smallest/largest at ±1 |

Ordinal rank discards distance information. Winsor + scale keeps it inside the band; only the extreme tails get bucketed to `±1`.

---

## Engine fit

Bounded `[-1, +1]` date × asset vector — the well-behaved signal that feeds neutralisation → turnover control / decay. No engine API.

---

## Mismatches with the spec

| Idea says | Pipeline has | Question to ask |
|---|---|---|
| "Rank to [0,1]" (WebSim default) | this rescales to `[-1, +1]` (centred). | Confirm `[-1, +1]`; it's zero-meaned for symmetric distributions, which suits long-short. |
| "Preserve raw magnitudes / don't winsorise" | call with `low_pct=0.0, high_pct=1.0` for pure min-max. | Confirm whether you want the tails clipped — outliers in any one bar would dominate the rescale otherwise. |
| "Winsorise 1/99 (tighter)" | `low_pct=0.01, high_pct=0.99`. | Confirm the winsor band; the smaller the tails, the more outliers count in the rescale denominator. |
| "Keep ordinal rank semantics (old behaviour)" | not exposed any more; use `scipy.stats.rankdata` or pandas `rank` directly if needed. | Surface if the strategy specifically relies on ordinal positions. |

---

## Minimal valid cell

```python
from backtest.alpha_pipeline import rank

ranked = rank.run(sig)  # default 5/95 winsor + rescale to [-1, +1]
print("range:", float(ranked.min().min()), "→", float(ranked.max().max()))
print("per-bar mean (≈0 for symmetric input):", ranked.mean(axis=1).abs().max().round(4))
print(ranked.tail(3).round(3))
```

---

## Validation checklist

- [ ] All values in `[-1, +1]`; NaNs only where the signal was NaN.
- [ ] On a strict-order bar with N ≥ 3, smallest unclipped value > −1 (interior), largest unclipped < +1 (interior); only the tails pin to ±1.
- [ ] Single-asset bars are 0.
- [ ] All-tied bars are 0.
- [ ] `low_pct=0.0, high_pct=1.0` yields pure min-max scaling (no clipping; evenly-spaced inputs → evenly-spaced outputs).

---

## What NOT to do

- **Do not** run this stage *after* neutralise — winsor+scale is not group-aware and would undo group-neutrality. Order is rank → neutralise.
- **Do not** assume "rank" means ordinal positions any more — this module preserves interior magnitudes. If you truly need ordinal rank, compute it explicitly.
- **Do not** set `low_pct` very high or `high_pct` very low — too-tight a band crushes most of the signal to ±1 and loses IR.
- **Do not** call with `low_pct >= high_pct` — raises by design.

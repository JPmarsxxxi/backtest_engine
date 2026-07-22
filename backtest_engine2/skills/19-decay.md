# Skill 19 — Decay (alpha pipeline Stage 7)

> **Read `PROTOCOL.md`, `00-overview.md`, `12-alpha-overview.md` first.** Cell-loop and announcement rules apply.

> Stage 7. Blend each bar's signal with its recent history so the alpha persists over its horizon and trades less. Applied **after** turnover control (Skill 6). Pick **one** method.

---

## When to load

- **Stage 7**: the alpha's edge lasts several bars and you want to hold the view (smoother, lower turnover) rather than re-decide every bar.

---

## API

```python
from backtest.alpha_pipeline import decay

out = decay.run(sig, "sma", window=5)       # rolling mean
out = decay.run(sig, "linear", window=5)    # linear-decay weighted avg
out = decay.run(sig, "ema", window=5)       # ema, λ = 2/(window+1)
```
`run(signal, method, window) -> pd.DataFrame`. `source: backtest/alpha_pipeline/decay.py:run`

---

## Math (PDF Ch. 7 "Decay")

- **sma**: equal-weight rolling mean over `window` bars.
- **linear**: `Σ w_k·sig_{t−k} / Σ w_k`, `w_k = window−k` (newest heaviest) — reuses `_linear` from turnover_control. `source: backtest/alpha_pipeline/decay.py:run`
- **ema**: exponential with span bridge `λ = 2/(window+1)` so one `window` knob maps across all three methods — reuses `_ema`. `source: backtest/alpha_pipeline/decay.py:run`

A single `window` parameter drives all three (the ema span bridge keeps them comparable).

---

## Engine fit

Same date × asset shape; lengthens the alpha's effective holding period before it becomes a position. No engine API.

---

## Mismatches with the spec

| Idea says | Pipeline has | Question to ask |
|---|---|---|
| "Decay 5 / decay over a week" | `window=5`, default `linear` flavour. | Which weighting: equal (sma), linear fade, or exponential (ema)? |
| "Make it stickier" | larger `window` = longer memory, lower turnover, more lag. | Confirm the horizon vs lag trade-off. |
| "Decay + turnover control" | both exist; decay is the *separate stage after* clamp/hump (Skill 6). | Confirm you want both stages, and the order. |

---

## Minimal valid cell

```python
from backtest.alpha_pipeline import decay

out = decay.run(sig, "linear", window=5)
print("turnover before:", sig.diff().abs().sum(axis=1).mean().round(4),
      "| after:", out.diff().abs().sum(axis=1).mean().round(4))
print(out.tail(3).round(4))
```

---

## Validation checklist

- [ ] Per-bar turnover lower than the input.
- [ ] Leading `window−1` rows are NaN (warmup), rest finite.
- [ ] Signal sign/shape preserved (decay smooths, doesn't invert).

---

## What NOT to do

- **Do not** apply decay *before* turnover control — it's the later stage (Ch. 7 order).
- **Do not** use a `window` longer than the alpha's real horizon — you'll decay away the edge.
- **Do not** mix decay flavours across compared alphas without noting it.

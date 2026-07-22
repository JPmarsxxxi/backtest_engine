# Skill 18 — Turnover Control (alpha pipeline Stage 6)

> **Read `PROTOCOL.md`, `00-overview.md`, `12-alpha-overview.md` first.** Cell-loop and announcement rules apply.

> Stage 6. Damp bar-to-bar position changes so trading cost doesn't eat the alpha. Pick **one** method per run.

---

## When to load

- **Stage 6**: the signal is correct but trades too much (high turnover / cost drag, Ch. 7).

---

## API

```python
from backtest.alpha_pipeline import turnover_control as tc

out = tc.run(neut, "clamp", k=3, window=20)     # clip to rolling mean ± k·std
out = tc.run(neut, "hump", threshold=0.05)      # hold prev if |Δ| < threshold
out = tc.run(neut, "ema", lam=0.2)              # λ·sig + (1−λ)·prev
out = tc.run(neut, "linear", d=5)               # linear-decay weighted avg
out = tc.run(neut, "trade_when", trigger=trig, exit=ex)  # event-gated: update only on trigger, flatten on exit
```
`run(signal, method, **params) -> pd.DataFrame`, `method ∈ {clamp, hump, ema, linear, trade_when}`. `source: backtest/alpha_pipeline/turnover_control.py:run`

---

## Math (PDF Ch. 7 "Turnover")

- **clamp** `_clamp(k, window)`: clip each asset to `rolling_mean ± k·rolling_std` (window bars) — caps outlier jumps. `source: …/turnover_control.py:_clamp`
- **hump** `_hump(threshold)`: if `|sig_t − held_{t−1}| < threshold`, **carry the previous value forward**; else update. Kills small churn. `source: …:_hump`
- **ema** `_ema(lam)`: `out_t = λ·sig_t + (1−λ)·out_{t−1}` via `ewm(adjust=False)`. Smaller `λ` = smoother/less turnover. `source: …:_ema`
- **linear** `_linear(d)`: `Σ w_k·sig_{t−k} / Σ w_k`, `w_k = d−k`, denom `d(d+1)/2` — newest bar weighted most. `source: …:_linear`
- **trade_when** `_trade_when(trigger, exit_cond)`: hold the previous value forward; adopt the current signal only on bars where `trigger` is nonzero; set NaN (flatten) where `exit` is nonzero. **Exit takes precedence** over trigger on the same bar. `trigger`/`exit` are date × asset frames (nonzero = true). This is a *gate*, not a smoother — it cuts turnover by trading only on chosen events. `source: …:_trade_when`

---

## Engine fit

Reduces turnover *inside the signal* before it ever becomes a position; the engine's cost model (Skill 8 / 23) then charges whatever turnover remains. Same date × asset shape.

---

## Mismatches with the spec

| Idea says | Pipeline has | Question to ask |
|---|---|---|
| "Smooth it" | three smoothers (clamp/ema/linear) + one gate (hump). | Which behaviour: cap spikes (clamp), exponential memory (ema), windowed fade (linear), or ignore tiny moves (hump)? |
| "Only trade on an event / signal / regime; hold otherwise" | `trade_when(trigger, exit)` — event-gated, not a smoother. | What is the trigger condition, and what closes the position (exit)? |
| "Apply all" | one method per call (compose by chaining cells). | Confirm the order if chaining. |
| "Decay the alpha" | that's Skill 7 (decay) — same math family, applied *after* clamp/hump. | Clarify: turnover-control vs the separate decay stage. |

---

## Minimal valid cell

```python
from backtest.alpha_pipeline import turnover_control as tc

out = tc.run(neut, "ema", lam=0.2)
print("turnover before:", neut.diff().abs().sum(axis=1).mean().round(4),
      "| after:", out.diff().abs().sum(axis=1).mean().round(4))
print(out.tail(3).round(4))
```

---

## Validation checklist

- [ ] Mean per-bar `|Δposition|` is lower after than before.
- [ ] `clamp`: no value exceeds its rolling band (compare only on non-NaN bars).
- [ ] `hump`: bars with sub-threshold moves exactly equal the prior bar.
- [ ] `trade_when`: value only changes on trigger bars; is NaN on/after exit until the next trigger.
- [ ] Warmup NaNs from rolling windows are expected; downstream handles them.

---

## What NOT to do

- **Do not** stack every method blindly — each adds lag; justify the choice by the turnover/IR trade-off.
- **Do not** confuse this with Skill 7 decay (decay runs *after*, as its own stage).
- **Do not** clamp with too-tight `k` — you'll flatten the signal and lose IR.

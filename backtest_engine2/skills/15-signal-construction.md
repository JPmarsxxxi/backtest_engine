# Skill 15 — Signal Construction (alpha pipeline Stage 3)

> **Read `PROTOCOL.md`, `00-overview.md`, `12-alpha-overview.md` first.** Cell-loop and announcement rules apply.

> Stage 3. Turn the resampled price (or fundamental field) into a raw signal vector — one value per instrument per bar, `+` = long, `−` = short.

---

## When to load

- **Stage 3**: encoding the *idea* as a signal. This is where the user's hypothesis becomes math.

---

## API

```python
from backtest.alpha_pipeline import signal_construction as sc

sig = sc.run(px, "mean_reversion", lookback=5)
sig = sc.run(px, "momentum", lookback=20)
sig = sc.run(field, "fundamental", lookback=4)          # field passed as data
sig = sc.run(px, "custom", fn=lambda d, n: -d.pct_change(n), n=5)   # any idea
```
`run(data, signal_type, lookback=None, fn=None, **fn_kwargs) -> pd.DataFrame`. `source: backtest/alpha_pipeline/signal_construction.py:run`

---

## Math (PDF)

- **mean_reversion** (Ch. 5 Alpha1): `−Σ(log returns over lookback) = −log(close_t/close_{t−L})`. Log returns make "sum of returns" exactly equal the book's total-window-return, no approximation. `+` when price fell (expect bounce).
- **momentum** (Ch. 21): `+Σ(log returns)` — the sign flip of mean-reversion.
- **fundamental** (Ch. 19): `field_t − field_{t−L}` (rate of change). Pass the fundamental field *as `data`*.
- **custom**: `fn(data, **fn_kwargs)` — arbitrary expression returning a date × asset frame. **This is the "state any signal" path**: when the user describes a novel idea, write `fn` here.

---

## Engine fit

Output is a raw date × asset vector — same shape the rest of the pipeline passes. No engine API. Leading `lookback` rows are NaN (warmup): you need `L+1` prices for `L` returns.

---

## Mismatches with the spec

| Idea says | Pipeline has | Question to ask |
|---|---|---|
| "Reversion / contrarian" | `mean_reversion` (log returns). | Confirm lookback and that log-return summing is acceptable (it equals the PDF total-window return). |
| "Trend / momentum" | `momentum` (sign flip). | Confirm lookback/horizon. |
| Anything else ("volume-weighted reversion", "breakout"…) | `custom` with a written `fn`. | Confirm the exact expression before writing the `fn`. |
| Fundamental factor | `fundamental` = change over lookback; needs the field resampled (Skill 2) and passed as `data`. | Do you have the fundamental field? Raw level vs change? |

---

## Minimal valid cell

```python
from backtest.alpha_pipeline import signal_construction as sc

sig = sc.run(px, "mean_reversion", lookback=5)
print("signal shape:", sig.shape, "| warmup NaN rows:", sig.isna().all(axis=1).sum())
print(sig.tail(3).round(4))
```

---

## Validation checklist

- [ ] Leading `lookback` rows are NaN; the rest are finite.
- [ ] Sign convention is as intended (`+` long): for `mean_reversion`, a recent price rise → negative signal.
- [ ] For `custom`, the returned frame matches `data.shape` (the module raises otherwise).

---

## What NOT to do

- **Do not** hard-code a new `signal_type` for every idea — use `custom` with an `fn`.
- **Do not** mix simple and log returns across runs when comparing alphas — keep it log (the module's choice).
- **Do not** use `pct_change()` on the full series in a way that peeks ahead; build the signal only from past bars.

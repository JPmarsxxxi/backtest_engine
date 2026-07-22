# Skill 14 — Frequency / PIT (alpha pipeline Stage 2)

> **Read `PROTOCOL.md`, `00-overview.md`, `12-alpha-overview.md` first.** Cell-loop and announcement rules apply.

> Stage 2. Resample the in-universe price/volume to the prediction frequency and apply the delay (point-in-time) shift so bar `t` uses only data it could have known.

---

## When to load

- **Stage 2**: choosing the prediction frequency (tick/intraday/daily/weekly/monthly) and the delay convention (0 or 1) before building a signal.

---

## API

```python
from backtest.alpha_pipeline import frequency

px = frequency.run(prices, freq="daily", delay=1)            # close per bar
vol = frequency.run(volume, freq="daily", delay=1, agg="sum")  # summed volume
```
`run(data, freq, delay, agg="last") -> pd.DataFrame`. `source: backtest/alpha_pipeline/frequency.py:run`

---

## Math (PDF Ch. 4 "Alpha Prediction Frequency")

- Frequencies: `tick`, `intraday` (passthrough — native resolution); `daily`/`weekly`/`monthly` (calendar resample, `agg="last"`=close, `agg="sum"`=volume).
- **delay 1** = *"only data available before the current trading day"* → `out.shift(1)` (bar `t` sees only `≤ t-1`); introduces a leading NaN. **delay 0** = snapshot (bar `t` sees its own period). No lookahead.

---

## Engine fit — overlap to flag

The engine enforces PIT independently: `FieldSpec(lag=...)` shifts a field forward (`source: backtest/data/panel.py:13–25`) and `DataView.prices` only exposes `.loc[:self.t]` (`source: backtest/data/panel.py:37–39`). This module applies its **own** delay, so the raw-vector pipeline is self-contained. **When you later feed the engine (Skill 23), set the panel's `FieldSpec` lag to 0** so PIT is not applied twice.

---

## Mismatches with the spec

| Idea says | Pipeline has | Question to ask |
|---|---|---|
| "Intraday/tick alpha" | passthrough only — you can't manufacture finer bars than the data holds. | Is the fetched data already sub-daily? If not, intraday/tick is meaningless. |
| "Use delay-0 / today's close" | `delay=0` snapshot. | Confirm: delay-0 risks same-bar lookahead unless the close is genuinely tradable at decision time. |
| "Weekly signal" | `freq="weekly"` (calendar resample). | Confirm resample aggregation (`last` for price, `sum` for volume). |

---

## Minimal valid cell

```python
from backtest.alpha_pipeline import frequency

px = frequency.run(prices[mask.columns], "daily", delay=1)
print("resampled price shape:", px.shape)
print(px.tail(3).round(2))
```

---

## Validation checklist

- [ ] First row(s) are NaN when `delay=1` (the PIT shift).
- [ ] Row count matches the calendar resample (≈ days/weeks/months in range).
- [ ] No future bar leaks into bar `t` (spot-check that `t`'s value equals the prior period when `delay=1`).

---

## What NOT to do

- **Do not** request `intraday`/`tick` on daily data and expect detail — it just passes through.
- **Do not** double-apply PIT: if this cell delays, the engine panel's `FieldSpec` lag must be 0.
- **Do not** resample price with `sum` or volume with `last` — `last`=close (price), `sum`=volume.

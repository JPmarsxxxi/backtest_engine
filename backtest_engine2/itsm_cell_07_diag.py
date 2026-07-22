# Cell 7 — Diagnostic (read-only): unintended overnight inventory + unfilled by year.
pos = result.positions["BTCUSDT"]
idx = pos.index
hm = idx.hour * 60 + idx.minute

# Intended exposure: only the 23:30->00:00 bar. Position recorded at the 23:30 stamp
# (post-trade) is carried into the midnight bar; position at any OTHER stamp should be 0.
intended_carry = (hm == 23 * 60 + 30)
held = pos.abs() > 1.0  # >$1 = held
overnight_bars = held & ~intended_carry & ~(hm == 0)  # held outside window (00:00 stamp itself is exit-bar, post-trade should be 0 too)
held_at_exit = held & (hm == 0)

pnl = result.pnl
pnl_intended = pnl[(hm == 0)].sum()              # last-half-hour P&L lands on the midnight bar
pnl_unintended = pnl[overnight_bars.shift(1, fill_value=False).values].sum()

print(f"Bars with intended position (23:30 stamps held): {int((held & intended_carry).sum())}")
print(f"Bars STILL holding at exit stamp (00:00, post-trade): {int(held_at_exit.sum())}")
print(f"Bars holding outside the window entirely: {int(overnight_bars.sum())}")
print(f"P&L on intended last-half-hour bars: ${pnl_intended:,.0f}")
print(f"P&L on bars following unintended holds: ${pnl_unintended:,.0f}")
print(f"Total costs: ${result.costs.sum():,.0f}")

print("\nBy year: unfilled $, year-end equity, bars-held-outside-window")
unf = result.unfilled.abs().sum(axis=1)
eq = result.equity_curve
for y in range(2017, 2027):
    m = idx.year == y
    if not m.any():
        continue
    print(f"  {y}: unfilled ${unf[m].sum():>13,.0f} | eq end ${eq[m].iloc[-1]:>11,.0f} "
          f"| overnight-bars {int(overnight_bars[m].sum()):>6}")

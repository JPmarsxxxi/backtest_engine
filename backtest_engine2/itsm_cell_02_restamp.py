# Cell 2 — Stage 1 amendment: re-stamp bars to close-time (open_time + 30min).
# Bar labeled t now means "all information in this bar existed by t" — removes
# the 30-minute lookahead DataView would otherwise grant on every bar.
# Outliers were reviewed and accepted in Cell 1 (real moves), so check_outliers=False here.
prices = prices.copy()
volume = volume.copy()
prices.index = prices.index + pd.Timedelta("30min")
volume.index = volume.index + pd.Timedelta("30min")

panel = DataPanel(prices, volume=volume, check_outliers=False)

print(f"Bars: {len(panel.dates)} (unchanged from Cell 1: 154273)")
print(f"Range: {panel.dates[0]} -> {panel.dates[-1]}")
print("Midnight-stamped bars (= 23:30->24:00 bars, the last-half-hour P&L bar):",
      int(((panel.dates.hour == 0) & (panel.dates.minute == 0)).sum()))
print("00:30-stamped bars (= first-half-hour signal bars):",
      int(((panel.dates.hour == 0) & (panel.dates.minute == 30)).sum()))
prices.tail(2)

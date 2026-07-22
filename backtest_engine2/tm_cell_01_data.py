# Cell 1 — #014 setup + data: FTMO macro universe daily close panel.
# Source: data/ftmo_daily.parquet (Dukascopy daily BID closes 2000-2025 + 2026 H1 from
# hourly; 45 instruments: 28 FX, 12 indices, 2 metals, 3 energy; crypto EXCLUDED — swap
# -30%/yr per side). Instruments start at different dates (GER40/US30 2013, NATGAS 2012,
# UKOIL 2006) -> ragged head is expected; ffill capped at 5 days so dead/stale series
# don't ghost-trade; leading NaNs stay NaN until an instrument is born.
import numpy as np
import pandas as pd
from backtest.data import DataPanel

RAW = pd.read_parquet(r"C:\Users\User\backtest_engine\backtest_engine2\data\ftmo_daily.parquet")
RAW["date"] = pd.to_datetime(RAW["date"]).dt.tz_localize(None)

px = RAW.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
px = px[sorted(px.columns)]
# Sunday rows: FX/crypto venues print a thin Sunday candle; indices don't. Collapse to a
# weekday grid (Mon-Fri), Sunday's close is superseded by Monday anyway at daily horizon.
px = px[px.index.dayofweek < 5]
grid = pd.bdate_range(px.index[0], px.index[-1])
prices = px.reindex(grid).ffill(limit=5)

panel = DataPanel(prices, check_outliers=False)  # known real jumps: CHF depeg 2015, COVID, oil 2020

print(f"Panel: {len(panel.dates)} days x {len(panel.assets_all)} instruments | "
      f"{panel.dates[0].date()} -> {panel.dates[-1].date()}")
born = prices.notna().idxmax()
late = born[born > prices.index[0] + pd.Timedelta(days=30)]
print(f"Late starters: {dict((k, str(v.date())) for k, v in late.items())}")
alive = prices.notna().sum(axis=1)
print(f"Instruments alive: start {alive.iloc[0]}, 2010 {alive[alive.index.year == 2010].iloc[0]}, "
      f"end {alive.iloc[-1]}")
print(f"Last dates with data per instrument: min {prices.apply(lambda c: c.last_valid_index()).min().date()}")

# Sanity: known levels at known dates (rough).
checks = [("EURUSD", "2008-07-15", 1.55, 1.62), ("US500", "2020-03-23", 2100, 2400),
          ("XAUUSD", "2011-09-05", 1850, 1950), ("USOIL", "2008-07-03", 135, 150),
          ("JP225", "2024-07-11", 41000, 43000)]
for sym, d, lo, hi in checks:
    v = prices.loc[d, sym]
    print(f"  {sym} @ {d}: {v:,.2f} {'OK' if lo <= v <= hi else f'!! expected [{lo},{hi}]'}")
prices.tail(2).iloc[:, :6]

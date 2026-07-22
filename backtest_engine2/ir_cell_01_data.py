# Cell 1 — #019 index reversal: daily US500 panel from ftmo_daily (2000-2026, CFD-faithful).
# Full sample; volume retained (patchy ~53%, UNUSED for costs — index CFD has no ADV impact).
# check_outliers=True to SEE the 2008/2020 crash bars (real moves, accept like #013).
import numpy as np
import pandas as pd
from backtest.data import DataPanel

ENG = r"C:\Users\User\backtest_engine\backtest_engine2"
fd = pd.read_parquet(ENG + r"\data\ftmo_daily.parquet")
u = fd[fd.symbol == "US500"].copy()
u["date"] = pd.to_datetime(u["date"])
u = u.set_index("date").sort_index()
u = u[~u.index.duplicated(keep="last")]

prices = u[["close"]].rename(columns={"close": "US500"})
volume = u[["volume"]].rename(columns={"volume": "US500"})
panel = DataPanel(prices, volume=volume, check_outliers=True)

print(f"Bars {len(panel.dates)} | {panel.dates[0].date()} -> {panel.dates[-1].date()}")
print(f"NaN frac prices: {prices.isna().mean().mean():.3%} | volume>0: {(volume['US500'] > 0).mean():.1%}")
yrs = pd.Series(panel.dates.year).value_counts().sort_index()
print("bars/yr:", {int(k): int(v) for k, v in yrs.items()})

rep = panel.outlier_report(mad_threshold=10.0)
print(f"\noutlier bars (MAD>10): {len(rep)}")
if len(rep):
    print(rep.head(5).to_string())
print("\ntail:")
print(prices.tail(3).to_string())

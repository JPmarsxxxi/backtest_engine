# Cell 1 — Setup + data load (#027 intraday FX stat-arb)
import pandas as pd
from backtest.data import DataPanel

prices = pd.read_parquet("data/fx_intraday_m15.parquet").sort_index()
prices = prices[~prices.index.duplicated(keep="first")]   # engine requires unique, monotonic index

panel = DataPanel(prices, check_outliers=True)

print(f"Bars:   {len(panel.dates):,}")
print(f"Assets: {len(panel.assets_all)} -> {list(panel.assets_all)}")
print(f"Range:  {panel.dates[0]} -> {panel.dates[-1]}   tz={prices.index.tz}")
print(f"NaN fraction in prices: {prices.isna().mean().mean():.4%}")
print(f"index monotonic increasing: {prices.index.is_monotonic_increasing}, unique: {prices.index.is_unique}")
print("\nprices.tail(3):")
print(prices.tail(3).to_string())

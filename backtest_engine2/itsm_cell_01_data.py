# Cell 1 — Setup + data load: BTCUSDT 30m panel (full history, tz-naive UTC)
import pandas as pd
from backtest.data import DataPanel

RAW = pd.read_parquet(r"C:\Users\User\backtest_engine\backtest_engine2\data\btc_30m.parquet")

prices = RAW.pivot(index="open_time", columns="symbol", values="close")
volume = RAW.pivot(index="open_time", columns="symbol", values="quote_volume")
prices.index = prices.index.tz_localize(None)   # tz-naive UTC (values unchanged)
volume.index = volume.index.tz_localize(None)

panel = DataPanel(prices, volume=volume, check_outliers=True)

n_days = len(pd.unique(prices.index.date))
expected = (prices.index[-1] - prices.index[0]) / pd.Timedelta("30min") + 1
print(f"Bars: {len(panel.dates)}  (expected on full 30m grid: {int(expected)}, gap {1 - len(panel.dates)/int(expected):.3%})")
print(f"Assets: {len(panel.assets_all)}  -> {list(panel.assets_all)}")
print(f"Range: {panel.dates[0]} -> {panel.dates[-1]}  ({n_days} UTC days)")
print(f"NaN fraction in prices: {prices.isna().mean().mean():.3%}")
prices.tail(3)

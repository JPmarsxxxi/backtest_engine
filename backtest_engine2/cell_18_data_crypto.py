# Cell 18 — Crypto daily panel from Binance hourly top-50 pool
import pandas as pd
from backtest.data import DataPanel

p = r"C:\Users\User\backtest_engine\backtest_engine2\data\binance_hourly_top50_pool.parquet"
df = pd.read_parquet(p)

# drop stablecoins / metal-pegged (no reversal signal, pollute the cross-section)
EXCLUDE = {"BFUSDUSDT","RLUSDUSDT","USD1USDT","USDEUSDT","USDSUSDT","XAUTUSDT","PAXGUSDT"}
df = df[~df["symbol"].isin(EXCLUDE)].copy()
df["coin_vol"] = df["quote_volume"] / df["close"]          # quote($)->coin count for engine

close = df.pivot_table(index="open_time", columns="symbol", values="close")
cvol  = df.pivot_table(index="open_time", columns="symbol", values="coin_vol")
close.index = pd.to_datetime(close.index).tz_localize(None)
cvol.index  = pd.to_datetime(cvol.index).tz_localize(None)

prices_crypto = close.resample("1D").last().sort_index()
volume_crypto = cvol.resample("1D").sum().reindex(index=prices_crypto.index, columns=prices_crypto.columns)
ren = {c: c[:-4] if c.endswith("USDT") else c for c in prices_crypto.columns}
prices_crypto = prices_crypto.rename(columns=ren); volume_crypto = volume_crypto.rename(columns=ren)
prices_crypto.index.name = volume_crypto.index.name = "Date"

panel_crypto = DataPanel(prices_crypto, volume=volume_crypto, check_outliers=True)
print(f"Bars:   {len(panel_crypto.dates)}")
print(f"Assets: {len(panel_crypto.assets_all)}")
print(f"Range:  {panel_crypto.dates[0].date()} -> {panel_crypto.dates[-1].date()}")
print(f"NaN fraction: {prices_crypto.isna().mean().mean():.1%}")
live = prices_crypto.notna().sum(axis=1)
print(f"Coins live: {live.iloc[0]} at start -> {live.iloc[-1]} at end (breadth grows over time)")
dvol = (prices_crypto * volume_crypto).median()
q = dvol.quantile([0.05,0.5,0.95]) / 1e6
print(f"Median daily $-vol per coin ($M): p5={q.iloc[0]:.1f} median={q.iloc[1]:.1f} p95={q.iloc[2]:.0f}")

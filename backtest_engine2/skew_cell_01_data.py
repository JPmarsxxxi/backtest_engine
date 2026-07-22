# Cell 1 — FTMO-confirmed crypto daily panel from Binance hourly pool
import pandas as pd
from backtest.data import DataPanel

P = r"C:\Users\User\backtest_engine\backtest_engine2\data\binance_hourly_top50_pool.parquet"
df = pd.read_parquet(P)

# FTMO-confirmed tradables (intersection with pool); stables/pegs excluded by construction
FTMO14 = ["BTC","ETH","ADA","DOT","DASH","LTC","XRP","DOGE","SOL","BNB","XLM","AAVE","LINK","AVAX"]
KEEP = {c + "USDT" for c in FTMO14}
df = df[df["symbol"].isin(KEEP)].copy()
df["coin_vol"] = df["quote_volume"] / df["close"]          # quote($) -> coin count for ADV

close = df.pivot_table(index="open_time", columns="symbol", values="close")
cvol  = df.pivot_table(index="open_time", columns="symbol", values="coin_vol")
close.index = pd.to_datetime(close.index).tz_localize(None)
cvol.index  = pd.to_datetime(cvol.index).tz_localize(None)

prices_skew = close.resample("1D").last().sort_index()
volume_skew = cvol.resample("1D").sum().reindex(index=prices_skew.index, columns=prices_skew.columns)
ren = {c: c[:-4] for c in prices_skew.columns}
prices_skew = prices_skew.rename(columns=ren); volume_skew = volume_skew.rename(columns=ren)
prices_skew.index.name = volume_skew.index.name = "Date"

# truncate ragged tail: 11/14 coins' data collection ends 2026-05-19
CUTOFF = "2026-05-19"
prices_skew = prices_skew.loc[:CUTOFF]
volume_skew = volume_skew.loc[:CUTOFF]

missing = sorted(set(FTMO14) - set(prices_skew.columns))
panel_skew = DataPanel(prices_skew, volume=volume_skew, check_outliers=True)
print(f"Bars:   {len(panel_skew.dates)}")
print(f"Assets: {len(panel_skew.assets_all)}  (missing from pool: {missing or 'none'})")
print(f"Range:  {panel_skew.dates[0].date()} -> {panel_skew.dates[-1].date()}")
print(f"NaN fraction: {prices_skew.isna().mean().mean():.1%}")
live = prices_skew.notna().sum(axis=1)
print(f"Coins live: {live.iloc[0]} at start -> {live.iloc[-1]} at end")
dvol = (prices_skew * volume_skew).median()
q = dvol.quantile([0.05, 0.5, 0.95]) / 1e6
print(f"Median daily $-vol per coin ($M): p5={q.iloc[0]:.1f} median={q.iloc[1]:.1f} p95={q.iloc[2]:.0f}")

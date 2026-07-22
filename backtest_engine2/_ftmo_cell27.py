# Cell 27 — FTMO-restricted universe (confirmed-13): build panel_crypto_ftmo
from backtest.data import DataPanel

FTMO13 = ["BTC","ETH","ADA","DOT","DASH","LTC","XRP","DOGE","SOL","BNB","XLM","AAVE","LINK"]
keep    = [c for c in FTMO13 if c in prices_crypto.columns]
missing = [c for c in FTMO13 if c not in prices_crypto.columns]

prices_ftmo = prices_crypto[keep].copy()
volume_ftmo = volume_crypto[keep].copy()
panel_crypto_ftmo = DataPanel(prices_ftmo, volume=volume_ftmo, check_outliers=True)

print(f"FTMO-13 kept ({len(keep)}): {keep}")
print(f"missing from pool: {missing}")
print(f"Assets: {len(panel_crypto_ftmo.assets_all)}  (full universe was {len(panel_crypto.assets_all)})")
print(f"Range:  {panel_crypto_ftmo.dates[0].date()} -> {panel_crypto_ftmo.dates[-1].date()}")
# breadth over time — a cross-sectional strat needs enough live names to rank
live = prices_ftmo.notna().sum(axis=1)
print("live coins per year (min/mean/max):")
for yr in range(2019, 2027):
    sub = live[live.index.year == yr]
    if len(sub):
        print(f"  {yr}: min={sub.min()}  mean={sub.mean():4.1f}  max={sub.max()}")

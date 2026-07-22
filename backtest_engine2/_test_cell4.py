import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, r'C:\Users\User\backtest_engine\backtest_engine2')

from backtest.costs import Commission, Spread, MarketImpact, CompositeCostModel, LiquidityCap
import pandas as pd, yfinance as yf
from backtest.data import DataPanel

costs = CompositeCostModel([
    Commission(bps=1),
    Spread(half_bps={'SPX': 1, 'CASH': 0}),
    MarketImpact(k=10, kind='sqrt', adv_lookback=20),
])
liq = LiquidityCap(cap_pct=0.10, adv_lookback=20)

print('Cost components :', [type(m).__name__ for m in costs.models])
print(f'LiquidityCap    : cap_pct={liq.cap_pct}, adv_lookback={liq.adv_lookback}')

raw = yf.download(['^GSPC','^VIX'], start='1990-01-01', auto_adjust=False, progress=False)['Close']
raw.columns = ['SPX','VIX']; raw = raw.dropna()
vol_raw = yf.download('^GSPC', start='1990-01-01', auto_adjust=False, progress=False)['Volume'].squeeze()
vol_raw.name = 'SPX'
idx = raw.index.intersection(vol_raw.index); raw = raw.loc[idx]; vol_raw = vol_raw.loc[idx]
prices_df = pd.DataFrame({'SPX': raw['SPX'], 'CASH': 1.0}, index=raw.index)
volume_df = pd.DataFrame({'SPX': vol_raw, 'CASH': float('nan')}, index=raw.index)
vix_df    = pd.DataFrame({'SPX': raw['VIX'], 'CASH': raw['VIX']}, index=raw.index)
panel = DataPanel(prices_df, volume=volume_df, features={'VIX': vix_df}, check_outliers=True)

print(f'Panel has volume: {panel.has_field("volume")}')
print("OK")

# Cell 1 — Setup + data load (S&P 100, 2015->now)
import pandas as pd, yfinance as yf
from pathlib import Path
from backtest.data import DataPanel

DATA_DIR = Path(r"C:\Users\User\backtest_engine\backtest_engine2\data")
PX_PATH, VOL_PATH = DATA_DIR / "sp100_prices.parquet", DATA_DIR / "sp100_volume.parquet"
START = "2015-01-01"

# Current S&P 100 (OEX) members. Survivorship-biased by construction (logged in registry).
SP100 = ["AAPL","ABBV","ABT","ACN","ADBE","AIG","AMD","AMGN","AMT","AMZN","AVGO","AXP",
 "BA","BAC","BK","BKNG","BLK","BMY","BRK-B","C","CAT","CHTR","CL","CMCSA","COF","COP",
 "COST","CRM","CSCO","CVS","CVX","DHR","DIS","DOW","DUK","EMR","FDX","GD","GE","GILD",
 "GM","GOOG","GOOGL","GS","HD","HON","IBM","INTC","INTU","JNJ","JPM","KO","LIN","LLY",
 "LMT","LOW","MA","MCD","MDLZ","MDT","MET","META","MMM","MO","MRK","MS","MSFT","NEE",
 "NFLX","NKE","NVDA","ORCL","PEP","PFE","PG","PM","PYPL","QCOM","RTX","SBUX","SCHW",
 "SO","SPG","T","TGT","TMO","TMUS","TSLA","TXN","UNH","UNP","UPS","USB","V","VZ","WFC",
 "WMT","XOM"]

if PX_PATH.exists() and VOL_PATH.exists():
    prices = pd.read_parquet(PX_PATH); volume = pd.read_parquet(VOL_PATH)
    print(f"Loaded cached parquet: {prices.shape[1]} tickers.")
else:
    raw = yf.download(SP100, start=START, auto_adjust=True, progress=False, threads=True)
    prices = raw["Close"].copy(); volume = raw["Volume"].copy()
    # identical index & columns required by DataPanel; drop tickers that returned nothing
    good = [c for c in prices.columns if prices[c].notna().any()]
    prices = prices[good].sort_index(); volume = volume.reindex(index=prices.index, columns=good)
    prices.index.name = volume.index.name = "Date"
    DATA_DIR.mkdir(exist_ok=True)
    prices.to_parquet(PX_PATH); volume.to_parquet(VOL_PATH)
    print(f"Downloaded {len(good)}/{len(SP100)} tickers -> cached parquet.")

panel = DataPanel(prices, volume=volume, check_outliers=True)
print(f"Bars:   {len(panel.dates)}")
print(f"Assets: {len(panel.assets_all)}")
print(f"Range:  {panel.dates[0].date()} -> {panel.dates[-1].date()}")
print(f"NaN fraction in prices: {prices.isna().mean().mean():.3%}")
prices.tail(3).iloc[:, :6]

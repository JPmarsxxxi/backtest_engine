# Cell 7 — Data load: S&P 400 MidCap (illiquidity-premium test universe)
import pandas as pd, requests, io, yfinance as yf
from pathlib import Path
from backtest.data import DataPanel

DATA_DIR = Path(r"C:\Users\User\backtest_engine\backtest_engine2\data")
PX_PATH, VOL_PATH = DATA_DIR / "sp400_prices.parquet", DATA_DIR / "sp400_volume.parquet"
START = "2015-01-01"

if PX_PATH.exists() and VOL_PATH.exists():
    prices_mid = pd.read_parquet(PX_PATH); volume_mid = pd.read_parquet(VOL_PATH)
    print(f"Loaded cached parquet: {prices_mid.shape[1]} tickers.")
else:
    h = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    r = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", headers=h, timeout=30)
    r.raise_for_status()
    syms = pd.read_html(io.StringIO(r.text))[0]["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
    raw = yf.download(syms, start=START, auto_adjust=True, progress=False, threads=True)
    prices_mid = raw["Close"].copy(); volume_mid = raw["Volume"].copy()
    good = [c for c in prices_mid.columns if prices_mid[c].notna().any()]
    prices_mid = prices_mid[good].sort_index(); volume_mid = volume_mid.reindex(index=prices_mid.index, columns=good)
    prices_mid.index.name = volume_mid.index.name = "Date"
    DATA_DIR.mkdir(exist_ok=True)
    prices_mid.to_parquet(PX_PATH); volume_mid.to_parquet(VOL_PATH)
    print(f"Downloaded {len(good)}/{len(syms)} tickers -> cached parquet.")

panel_mid = DataPanel(prices_mid, volume=volume_mid, check_outliers=True)
print(f"Bars:   {len(panel_mid.dates)}")
print(f"Assets: {len(panel_mid.assets_all)}")
print(f"Range:  {panel_mid.dates[0].date()} -> {panel_mid.dates[-1].date()}")
print(f"NaN fraction in prices: {prices_mid.isna().mean().mean():.3%}")

# Liquidity check: median daily $-volume per name, full-sample (diagnostic only, not a filter)
dollar_vol = (prices_mid * volume_mid).median()
q = dollar_vol.quantile([0.05, 0.25, 0.5, 0.75, 0.95]) / 1e6
print("Median daily $-volume per name ($M):")
print(f"  p5={q.iloc[0]:.1f}  p25={q.iloc[1]:.1f}  median={q.iloc[2]:.1f}  p75={q.iloc[3]:.1f}  p95={q.iloc[4]:.1f}")
print(f"  names under $5M/day: {(dollar_vol < 5e6).sum()} of {len(dollar_vol)}")

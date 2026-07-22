# Cell 1 — Setup + data load: GC=F daily OHLCV + spot-gold benchmark (spec.json).
# Trading: yfinance GC=F (continuous front-month COMEX gold futures).
# Benchmark: ftmo_daily XAUUSD spot proxy — yfinance XAUUSD=X is currently unavailable.
# Calendar: NYSE business days; missing bars ffill last price (spec filter).
import pandas as pd
import yfinance as yf
from pathlib import Path
from backtest.data import DataPanel, FieldSpec

DATA_DIR = Path(r"C:\Users\User\backtest_engine\backtest_engine2\data")
GC_PATH = DATA_DIR / "tfg_gc_futures.parquet"
BM_PATH = DATA_DIR / "tfg_xauusd_benchmark.parquet"
START = "2005-01-01"
ASSET = "GC=F"
GC_CONTRACT_OZ = 100  # COMEX GC contract multiplier: 100 troy oz per contract

if GC_PATH.exists():
    gc = pd.read_parquet(GC_PATH)
    print(f"Loaded cached GC=F parquet: {gc.shape[0]} rows.")
else:
    raw = yf.download(ASSET, start=START, auto_adjust=True, progress=False)
    gc = pd.DataFrame({
        "open": raw["Open"][ASSET],
        "high": raw["High"][ASSET],
        "low": raw["Low"][ASSET],
        "close": raw["Close"][ASSET],
        "volume": raw["Volume"][ASSET],
    })
    gc.index = pd.to_datetime(gc.index).tz_localize(None)
    gc.index.name = "date"
    DATA_DIR.mkdir(exist_ok=True)
    gc.to_parquet(GC_PATH)
    print(f"Downloaded GC=F -> {GC_PATH.name}")

if BM_PATH.exists():
    benchmark = pd.read_parquet(BM_PATH).squeeze("columns")
    print(f"Loaded cached benchmark parquet: {len(benchmark)} rows.")
else:
    ftmo = pd.read_parquet(DATA_DIR / "ftmo_daily.parquet")
    ftmo["date"] = pd.to_datetime(ftmo["date"]).dt.tz_localize(None)
    benchmark = (
        ftmo.loc[ftmo["symbol"] == "XAUUSD", ["date", "close"]]
        .set_index("date")["close"]
        .sort_index()
    )
    benchmark = benchmark[benchmark.index >= pd.Timestamp(START)]
    benchmark.to_frame("XAUUSD").to_parquet(BM_PATH)
    print(f"Built XAUUSD spot benchmark from ftmo_daily -> {BM_PATH.name}")

# NYSE business-day grid; spec: carry forward last known price on gaps.
grid = pd.bdate_range(gc.index.min(), gc.index.max())
ohlcv = gc.reindex(grid).ffill()
prices = ohlcv[["close"]].rename(columns={"close": ASSET})
# yfinance GC=F volume is in CONTRACTS; engine ADV math (LiquidityCap, MarketImpact)
# computes volume × price, so scale to oz so that product = true notional.
# Feed still undercounts real CME activity (~150-250k contracts/day) — conservative.
volume = ohlcv[["volume"]].rename(columns={"volume": ASSET}) * GC_CONTRACT_OZ
features = {
    "open": ohlcv[["open"]].rename(columns={"open": ASSET}),
    "high": ohlcv[["high"]].rename(columns={"high": ASSET}),
    "low": ohlcv[["low"]].rename(columns={"low": ASSET}),
}
ffill_spec = FieldSpec(lag=0, missing="ffill")
specs = {name: ffill_spec for name in ["prices", "volume", "open", "high", "low"]}

panel = DataPanel(
    prices,
    volume=volume,
    features=features,
    specs=specs,
    check_outliers=True,
)
benchmark = benchmark.reindex(grid).ffill()

print(f"Bars:   {len(panel.dates)}")
print(f"Assets: {list(panel.assets_all)}")
print(f"Range:  {panel.dates[0].date()} -> {panel.dates[-1].date()}")
print(f"NaN fraction in prices: {prices.isna().mean().mean():.3%}")
print(f"Features: {panel.available_fields()}")
print(f"Benchmark (XAUUSD spot proxy) aligned: {benchmark.notna().mean():.1%} coverage")
print(f"Active sample window per spec: {START} -> {panel.dates[-1].date()} "
      f"(10y train precedes first 2015-01 OOS slice)")
prices.tail(3)

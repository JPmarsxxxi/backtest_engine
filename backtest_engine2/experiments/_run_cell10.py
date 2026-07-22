"""Run Cell 10 logic standalone: single-path engine sanity for HurstPairsStrategy."""
import sys, warnings, time
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

REPO_ROOT = Path(r"C:\Users\User\backtest_engine")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
NB_DIR = REPO_ROOT / "backtest_engine2" / "experiments"
if str(NB_DIR) not in sys.path:
    sys.path.insert(0, str(NB_DIR))

from backtest.alpha_pipeline import universe, frequency
from backtest.data.panel import DataPanel, FieldSpec
from backtest.costs import Commission, Spread, ShortBorrow, CompositeCostModel, LiquidityCap
from backtest.engine.engine import Engine
from hurst_pairs_strategy import HurstPairsStrategy

DATA_DIR = REPO_ROOT / "backtest_engine2" / "data"
CACHE_DIR = DATA_DIR / "_cache"

# --- Cells 1-3 setup ---
raw = pd.read_parquet(DATA_DIR / "binance_hourly_top50_pool.parquet")
EXCLUDE = {"USD1USDT","USDSUSDT","RLUSDUSDT","BFUSDUSDT","USDEUSDT","XAUTUSDT","PAXGUSDT"}
filtered = raw[~raw["symbol"].isin(EXCLUDE)].copy()
prices = filtered.pivot(index="open_time", columns="symbol", values="close")
dollar_volume = filtered.pivot(index="open_time", columns="symbol", values="quote_volume")
prices.index = pd.DatetimeIndex(prices.index).tz_convert(None)
dollar_volume.index = prices.index
prices = prices.sort_index(); dollar_volume = dollar_volume.sort_index()
mask = universe.run(prices, dollar_volume, top_n=50, adv_window=720)
prices_pit = frequency.run(prices, freq="intraday", delay=1)
prices_pit_clean = prices_pit

# --- Cell 7 cache load ---
coint_series_df = pd.read_parquet(CACHE_DIR / "hurst_pairs_coint_series.parquet")
hurst_series_df = pd.read_parquet(CACHE_DIR / "hurst_pairs_hurst_series.parquet")
print(f"loaded caches: coint {len(coint_series_df):,} rows, hurst {len(hurst_series_df):,} rows")

# --- Cell 10 ---
volume_shares = (dollar_volume / prices).replace([np.inf, -np.inf], np.nan)

panel = DataPanel(
    prices=prices,
    volume=volume_shares,
    universe=mask,
    specs={"prices": FieldSpec(lag=0)},
)
print(f"panel: dates={len(panel.dates):,}, assets={len(panel.assets_all)}, has_volume={panel.has_field('volume')}")

costs = CompositeCostModel([
    Commission(bps=10),
    Spread(half_bps=5),
    ShortBorrow(annual_bps=200, trading_days=24*365),
])
liq = LiquidityCap(cap_pct=0.10, adv_lookback=168)
print(f"costs: {[type(m).__name__ for m in costs.models]}")

print("instantiating strategy...")
t0 = time.time()
strategy_v = HurstPairsStrategy(
    prices_pit=prices_pit_clean,
    coint_series_df=coint_series_df,
    hurst_series_df=hurst_series_df,
    verbose=True,
    assert_pit=True,
)
print(f"  done in {time.time()-t0:.1f}s")
print(f"strategy: {strategy_v}")

engine = Engine(data=panel, strategy=strategy_v, costs=costs, liquidity=liq, initial_capital=1_000_000.0)
print(f"\nengine built. Starting run...")
sys.stdout.flush()

t0 = time.time()
result = engine.run()
print(f"\n>>> engine.run() done in {(time.time()-t0)/60:.1f} min")

# --- Result summary ---
print(f"\n=== BACKTEST RESULT ===")
print(f"  bars run:        {len(result.equity_curve):,}")
print(f"  start equity:    ${result.equity_curve.iloc[0]:,.0f}")
print(f"  end equity:      ${result.equity_curve.iloc[-1]:,.0f}")
print(f"  total return:    {(result.equity_curve.iloc[-1]/result.equity_curve.iloc[0]-1)*100:+.2f}%")
print(f"  total costs:     ${result.costs.sum():,.0f}")
print(f"  costs / initial: {result.costs.sum()/result.equity_curve.iloc[0]*100:.2f}%")

report = result.summary(ann_factor=24*365)
print(f"\n  --- MetricsReport (annualised 8760/year) ---")
for k in ("sharpe","ann_return","max_drawdown","turnover","margin","hit_rate","pct_profitable_days"):
    if hasattr(report, k):
        v = getattr(report, k)
        if v is not None:
            print(f"  {k:>22}: {v}")

print(f"\n  --- Strategy counters ---")
print(f"  total entries opened: {strategy_v._n_entries_total:,}")
print(f"  exits TP:             {strategy_v._n_exits_tp:,}")
print(f"  exits SL:             {strategy_v._n_exits_sl:,}")
print(f"  exits barrier:        {strategy_v._n_exits_barrier:,}")
print(f"  exits de-listed:      {strategy_v._n_exits_delisted:,}")
print(f"  active at end:        {len(strategy_v.active_trades):,}")

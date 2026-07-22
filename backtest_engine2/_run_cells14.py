import sys, time, warnings
sys.path.insert(0, r'C:\Users\User\backtest_engine')
warnings.filterwarnings('ignore')

# ── CELL 1 ────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_datareader.data as pdr
from coinmetrics.api_client import CoinMetricsClient
from backtest.data import DataPanel, FieldSpec

START = '2014-01-01'; END = '2026-05-19'; MIN_MCAP_USD = 10_000_000; LAG_BARS = 1

TICKERS = [
    'BTC-USD','ETH-USD','BNB-USD','SOL-USD','ADA-USD','XRP-USD','AVAX-USD','DOT-USD',
    'LINK-USD','UNI-USD','AAVE-USD','MKR-USD','LTC-USD','BCH-USD','FIL-USD','ALGO-USD',
    'MANA-USD','SAND-USD','ENJ-USD','BAT-USD','ZRX-USD','STORJ-USD',
    'XMR-USD','DOGE-USD','XLM-USD','EOS-USD','TRX-USD','ZEC-USD','DASH-USD','ATOM-USD',
]
SYMS = [t.replace('-USD','') for t in TICKERS]

print('Fetching prices...')
raw = yf.download(TICKERS, start=START, end=END, progress=False, auto_adjust=True)
prices = raw['Close'].rename(columns=lambda c: c.replace('-USD',''))
volume = raw['Volume'].rename(columns=lambda c: c.replace('-USD',''))
prices.index = pd.to_datetime(prices.index)
volume.index = pd.to_datetime(volume.index)
print(f'Prices: {prices.shape}  NaN: {prices.isna().mean().mean():.1%}')

cm = CoinMetricsClient(host='community-api.coinmetrics.io')
CM_METRICS = {'CapMrktCurUSD':'MCAP','AdrActCnt':'AA','TxCnt':'NT','CapMVRVCur':'MRR'}

def _fetch_metric(cm_client, assets, metric, start, end, price_index):
    frames = []
    for sym in assets:
        try:
            df = cm_client.get_asset_metrics(
                assets=[sym.lower()], metrics=[metric],
                start_time=start, end_time=end, frequency='1d',
            ).to_dataframe()
            df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
            df = df.set_index('time')[[metric]].rename(columns={metric: sym})
            frames.append(df)
            time.sleep(0.15)
        except Exception:
            pass
    if not frames:
        return pd.DataFrame(index=price_index)
    return pd.concat(frames, axis=1).reindex(index=price_index)

print('Fetching Coin Metrics...')
cm_data = {}
for metric, label in CM_METRICS.items():
    df = _fetch_metric(cm, SYMS, metric, START, END, prices.index)
    df = df.reindex(columns=prices.columns)
    print(f'  {label}: {df.notna().mean().mean():.0%} coverage  {df.notna().any().sum()} assets')
    cm_data[label] = df
mcap = cm_data.pop('MCAP', None)

print('Fetching macro...')
MACRO_YF = {'^GSPC':'SPX','^VIX':'VIX','DX-Y.NYB':'USDX','GLD':'GLD','CL=F':'WTI'}
macro = {}
for ticker, label in MACRO_YF.items():
    s = yf.download(ticker, start=START, end=END, progress=False, auto_adjust=True)['Close'].squeeze()
    s.index = pd.to_datetime(s.index)
    macro[label] = s.reindex(prices.index, method='ffill')
try:
    ff  = pdr.DataReader('F-F_Research_Data_Factors_daily','famafrench',START,END)[0]/100
    mom = pdr.DataReader('F-F_Momentum_Factor_daily','famafrench',START,END)[0]/100
    ff.index  = pd.to_datetime(ff.index)
    mom.index = pd.to_datetime(mom.index)
    macro['RMRF'] = ff['Mkt-RF'].reindex(prices.index, method='ffill')
    macro['SMB']  = ff['SMB'].reindex(prices.index, method='ffill')
    macro['UMD']  = mom.iloc[:,0].reindex(prices.index, method='ffill')
    print('  FF RMRF/SMB/UMD: OK')
except Exception as e:
    print(f'  FF FAIL: {e}')

def _broadcast(s, idx, cols):
    arr = np.tile(s.reindex(idx, method='ffill').values[:,None], (1, len(cols)))
    return pd.DataFrame(arr, index=idx, columns=cols)

features = {**cm_data, **{k: _broadcast(v, prices.index, prices.columns) for k,v in macro.items()}}
if mcap is not None:
    features['MCAP'] = mcap
volume = volume.reindex(index=prices.index, columns=prices.columns)
universe = (mcap >= MIN_MCAP_USD).fillna(False) if mcap is not None else None

specs = {
    **{k: FieldSpec(lag=LAG_BARS, missing='ffill', max_staleness=7) for k in cm_data},
    **{k: FieldSpec(lag=LAG_BARS, missing='ffill', max_staleness=3) for k in macro},
    'MCAP':   FieldSpec(lag=0, missing='ffill', max_staleness=7),
    'volume': FieldSpec(lag=0, missing='ffill', max_staleness=3),
}
panel = DataPanel(prices, volume=volume, features=features, universe=universe,
                  specs=specs, check_outliers=True)
print(f'Panel: {len(panel.dates)} bars  {len(panel.assets_all)} assets  '
      f'{panel.dates[0].date()} to {panel.dates[-1].date()}')
print(f'Fields: {panel.available_fields()}')

# ── CELL 2 ────────────────────────────────────────────────────────────────────
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from backtest.strategy import Strategy
from backtest.data import DataView

FACTOR_COLS    = ['AA','NT','MRR','SPX','VIX','USDX','GLD','WTI','RMRF','SMB','UMD']
RETURN_FACTORS = frozenset({'RMRF','SMB','UMD'})
OLS3_COLS      = ['MRR','AA','NT']
NN_LAYERS = {
    'nn1':(32,),'nn2':(32,16),'nn3':(32,16,8),'nn4':(32,16,8,4),'nn5':(32,16,8,4,2),
}
ALL_MODEL_TYPES = ['ols','ols3','pls','lasso','ridge','enet','rf','gbrt','knn',
                   'nn1','nn2','nn3','nn4','nn5']

def _make_model(model_type):
    mt = model_type.lower()
    if mt in ('ols','ols3'): return LinearRegression()
    if mt == 'pls':   return PLSRegression(n_components=3)
    if mt == 'lasso': return Lasso(alpha=1e-3, max_iter=5000)
    if mt == 'ridge': return Ridge(alpha=1.0)
    if mt == 'enet':  return ElasticNet(alpha=1e-3, l1_ratio=0.5, max_iter=5000)
    if mt == 'rf':
        return RandomForestRegressor(n_estimators=200, max_depth=6,
                                     min_samples_leaf=5, n_jobs=-1, random_state=42)
    if mt == 'gbrt':
        return GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                         learning_rate=0.05, subsample=0.8, random_state=42)
    if mt == 'knn':
        return KNeighborsRegressor(n_neighbors=5, weights='distance', n_jobs=-1)
    if mt in NN_LAYERS:
        return MLPRegressor(hidden_layer_sizes=NN_LAYERS[mt], max_iter=500,
                            early_stopping=True, validation_fraction=0.1,
                            random_state=42, learning_rate_init=1e-3)
    raise ValueError(f'Unknown: {model_type}')


class CryptoMLFactorStrategy(Strategy):
    rebalance_frequency = 'W-SUN'

    def __init__(self, model_type='rf', min_train_samples=100, n_deciles=10,
                 gross_exposure=0.5):
        self.model_type        = model_type.lower()
        self.min_train_samples = min_train_samples
        self.n_deciles         = n_deciles
        self.gross_exposure    = gross_exposure
        self.feature_cols_     = OLS3_COLS if self.model_type == 'ols3' else FACTOR_COLS
        self._pipeline         = None

    def __repr__(self):
        return (f'CryptoMLFactorStrategy(model_type={self.model_type!r}, '
                f'n_deciles={self.n_deciles}, gross_exposure={self.gross_exposure})')

    def required_data(self):
        return FACTOR_COLS + ['volume', 'MCAP']

    def _to_weekly(self, dm):
        out = {}
        for col, df in dm.items():
            if col in RETURN_FACTORS:
                out[col] = df.resample('W-SUN').sum()
            else:
                w = df.resample('W-SUN').last().pct_change()
                out[col] = w.replace([np.inf, -np.inf], np.nan)
        return out

    def _build_Xy(self, weekly, prices):
        fwdret = prices.resample('W-SUN').last().pct_change().shift(-1)
        ref    = weekly[self.feature_cols_[0]].index
        weeks  = sorted(set(fwdret.index[:-1]) & set(ref))
        Xs, ys = [], []
        for dt in weeks:
            y_row  = fwdret.loc[dt].dropna()
            if len(y_row) < 3:
                continue
            assets = y_row.index
            X_cs = pd.DataFrame({
                col: weekly[col].loc[dt] if dt in weekly[col].index
                     else pd.Series(np.nan, index=prices.columns)
                for col in self.feature_cols_
            }).loc[assets]
            valid = assets[X_cs.notna().any(axis=1) & y_row.notna()]
            if len(valid) < 3:
                continue
            Xs.append(X_cs.loc[valid].to_numpy(dtype=float, na_value=np.nan))
            ys.append(y_row.loc[valid].to_numpy(dtype=float, na_value=np.nan))
        if not Xs:
            return np.empty((0, len(self.feature_cols_))), np.empty(0)
        return np.vstack(Xs), np.concatenate(ys)

    def fit(self, data):
        daily  = {col: data.feature(col) for col in self.feature_cols_}
        weekly = self._to_weekly(daily)
        n_weeks = weekly[self.feature_cols_[0]].dropna(how='all').shape[0]
        if n_weeks < self.min_train_samples:
            self._pipeline = None; return
        X, y = self._build_Xy(weekly, data.prices)
        if len(X) == 0:
            self._pipeline = None; return
        base = _make_model(self.model_type)
        if isinstance(base, PLSRegression):
            nc   = min(3, len(self.feature_cols_), X.shape[1])
            base = PLSRegression(n_components=nc)
        self._pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler',  StandardScaler()),
            ('model',   base),
        ])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self._pipeline.fit(X, y)

    def generate_weights(self, data, t):
        if self._pipeline is None:
            return pd.Series(dtype=float)
        daily  = {col: data.feature(col) for col in self.feature_cols_}
        weekly = self._to_weekly(daily)
        snap = {
            col: weekly[col].iloc[-1] if not weekly[col].empty
                 else pd.Series(np.nan, index=data.prices.columns)
            for col in self.feature_cols_
        }
        X_now = pd.DataFrame(snap)
        valid  = X_now.index[X_now.notna().any(axis=1)]
        if len(valid) < self.n_deciles:
            return pd.Series(dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            preds = self._pipeline.predict(X_now.loc[valid].to_numpy(dtype=float, na_value=np.nan)).ravel()
        pred_s = pd.Series(preds, index=valid)
        ranks  = pred_s.rank(pct=True)
        lo     = 1.0 / self.n_deciles
        hi     = (self.n_deciles - 1) / self.n_deciles
        long_  = ranks[ranks >= hi].index
        short_ = ranks[ranks <= lo].index
        mcap   = data.feature('MCAP').iloc[-1]
        def _vw(idx):
            cap = mcap.loc[idx].clip(lower=0).fillna(0)
            tot = cap.sum()
            return cap / tot if tot > 0 else pd.Series(1.0/len(idx), index=idx)
        w = pd.Series(0.0, index=valid)
        if len(long_):  w[long_]  =  _vw(long_)  * self.gross_exposure
        if len(short_): w[short_] = -_vw(short_) * self.gross_exposure
        return w

    def apply_risk(self, proposed, state, data):
        return proposed

print('Strategy class defined OK')

# ── CELL 3 ────────────────────────────────────────────────────────────────────
from backtest.costs import (
    Commission, Spread, MarketImpact, ShortBorrow,
    CompositeCostModel, LiquidityCap,
)

costs = CompositeCostModel([
    Commission(bps=10),
    Spread(half_bps=15),
    MarketImpact(k=10, kind='sqrt', adv_lookback=20),
    ShortBorrow(annual_bps=300, trading_days=252),
])
liq = LiquidityCap(cap_pct=0.05, adv_lookback=20)
print(f'Costs: {[type(m).__name__ for m in costs.models]}')
print(f'Panel has volume: {panel.has_field("volume")}')

# ── CELL 4 ────────────────────────────────────────────────────────────────────
from backtest.engine import Engine

TRAIN_END  = '2017-12-31'
TEST_START = '2018-01-01'

train_dates = panel.dates[panel.dates <= TRAIN_END]
print(f'\nTrain: {train_dates[0].date()} to {train_dates[-1].date()}'
      f'  ({len(train_dates)} bars, ~{len(train_dates)//7} weeks)')

strat_rf = CryptoMLFactorStrategy(model_type='rf')
engine   = Engine(panel, strat_rf, costs=costs, liquidity=liq, initial_capital=1_000_000)

print('Running engine (RF fit + test loop)...')
result = engine.run(train_dates=train_dates, start=TEST_START)

ic  = result.metadata['initial_capital']
fin = result.equity_curve.iloc[-1]
print(f'\nBars         : {len(result.equity_curve)}')
print(f'Rebalances   : {result.metadata["n_rebalances"]}')
print(f'Date range   : {result.metadata["start"].date()} to {result.metadata["end"].date()}')
print(f'Initial      : ${ic:>14,.0f}')
print(f'Final equity : ${fin:>14,.0f}')
print(f'Total return : {(fin/ic - 1):+.2%}')
print(f'Total costs  : ${result.costs.sum():>14,.2f}')
print(f'Fit called   : {result.metadata["fit_called"]}')
print(f'Max |weight| : {result.weights.abs().max().max():.2%}')
print(f'Max gross exp: {result.weights.abs().sum(axis=1).max():.2%}')

# ── CELL 5 — CPCV + MultiPathEngine ──────────────────────────────────────────
from backtest.splitters import CombinatorialPurgedCV
from backtest.engine import MultiPathEngine

splitter = CombinatorialPurgedCV(
    n_splits=6,
    n_test_groups=2,
    purge_bars=7,
    embargo_pct=0.005,
)

n            = len(panel.dates)
embargo_bars = int(n * splitter.embargo_pct)
print(f'\n=== CPCV ===')
print(f'Timeline : {n} bars  ({panel.dates[0].date()} to {panel.dates[-1].date()})')
print(f'Groups   : {splitter.n_groups} x ~{n // splitter.n_groups} bars (~{n // splitter.n_groups / 365:.1f} yrs each)')
print(f'Splits   : {splitter.n_splits()}  |  Paths: {splitter.n_paths()}')
print(f'Purge    : {splitter.purge_bars} bars  |  Embargo: {embargo_bars} bars  |  Trailing: {splitter.purge_bars + embargo_bars} bars')

first_train, first_test = next(iter(splitter.split(panel.dates)))
print(f'First split train: {len(first_train)} bars ({first_train[0].date()} to {first_train[-1].date()})')
print(f'First split test : {len(first_test)} bars ({first_test[0].date()} to {first_test[-1].date()})')
print(f'Disjoint: {len(first_train.intersection(first_test)) == 0}')

print('\nRunning MultiPathEngine (15 splits x RF, n_jobs=1)...')
mp = MultiPathEngine(
    panel,
    lambda: CryptoMLFactorStrategy(model_type='rf'),
    splitter,
    costs=costs,
    liquidity=liq,
    initial_capital=1_000_000,
)
res = mp.run(n_jobs=1)

print(f'\nSplits run  : {res.n_splits}')
print(f'Paths       : {res.n_paths}')
print(f'path_returns: {res.path_returns.shape}')
print(f'Date range  : {res.path_returns.index[0].date()} to {res.path_returns.index[-1].date()}')
print(f'NaN per path: {res.path_returns.isna().sum().to_dict()}')
print()
print('Final equity per path:')
for col, val in res.path_equity.iloc[-1].items():
    tr = val / 1_000_000 - 1
    print(f'  {col}: ${val:>12,.0f}  ({tr:+.1%})')

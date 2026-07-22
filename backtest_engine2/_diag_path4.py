import sys, time, warnings
sys.path.insert(0, r'C:\Users\User\backtest_engine')
warnings.filterwarnings('ignore')

import numpy as np, pandas as pd, yfinance as yf
import pandas_datareader.data as pdr
from coinmetrics.api_client import CoinMetricsClient
from backtest.data import DataPanel, FieldSpec
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
from backtest.costs import Commission, Spread, MarketImpact, ShortBorrow, CompositeCostModel, LiquidityCap
from backtest.splitters import CombinatorialPurgedCV
from backtest.engine import MultiPathEngine

# ── rebuild panel (same as before) ──
START='2014-01-01'; END='2026-05-19'
TICKERS=['BTC-USD','ETH-USD','BNB-USD','SOL-USD','ADA-USD','XRP-USD','AVAX-USD','DOT-USD',
         'LINK-USD','UNI-USD','AAVE-USD','MKR-USD','LTC-USD','BCH-USD','FIL-USD','ALGO-USD',
         'MANA-USD','SAND-USD','ENJ-USD','BAT-USD','ZRX-USD','STORJ-USD',
         'XMR-USD','DOGE-USD','XLM-USD','EOS-USD','TRX-USD','ZEC-USD','DASH-USD','ATOM-USD']
SYMS=[t.replace('-USD','') for t in TICKERS]
print('Fetching data...')
raw=yf.download(TICKERS,start=START,end=END,progress=False,auto_adjust=True)
prices=raw['Close'].rename(columns=lambda c:c.replace('-USD',''))
volume=raw['Volume'].rename(columns=lambda c:c.replace('-USD',''))
prices.index=pd.to_datetime(prices.index); volume.index=pd.to_datetime(volume.index)
cm=CoinMetricsClient(host='community-api.coinmetrics.io')
CM_METRICS={'CapMrktCurUSD':'MCAP','AdrActCnt':'AA','TxCnt':'NT','CapMVRVCur':'MRR'}
def _fetch(cm_client,assets,metric,start,end,idx):
    frames=[]
    for sym in assets:
        try:
            df=cm_client.get_asset_metrics(assets=[sym.lower()],metrics=[metric],start_time=start,end_time=end,frequency='1d').to_dataframe()
            df['time']=pd.to_datetime(df['time']).dt.tz_localize(None)
            df=df.set_index('time')[[metric]].rename(columns={metric:sym}); frames.append(df); time.sleep(0.15)
        except: pass
    return pd.concat(frames,axis=1).reindex(index=idx) if frames else pd.DataFrame(index=idx)
cm_data={}
for metric,label in CM_METRICS.items():
    df=_fetch(cm,SYMS,metric,START,END,prices.index).reindex(columns=prices.columns)
    cm_data[label]=df
mcap=cm_data.pop('MCAP',None)
MACRO_YF={'^GSPC':'SPX','^VIX':'VIX','DX-Y.NYB':'USDX','GLD':'GLD','CL=F':'WTI'}
macro={}
for ticker,label in MACRO_YF.items():
    s=yf.download(ticker,start=START,end=END,progress=False,auto_adjust=True)['Close'].squeeze()
    s.index=pd.to_datetime(s.index); macro[label]=s.reindex(prices.index,method='ffill')
ff=pdr.DataReader('F-F_Research_Data_Factors_daily','famafrench',START,END)[0]/100
mom=pdr.DataReader('F-F_Momentum_Factor_daily','famafrench',START,END)[0]/100
ff.index=pd.to_datetime(ff.index); mom.index=pd.to_datetime(mom.index)
macro['RMRF']=ff['Mkt-RF'].reindex(prices.index,method='ffill')
macro['SMB']=ff['SMB'].reindex(prices.index,method='ffill')
macro['UMD']=mom.iloc[:,0].reindex(prices.index,method='ffill')
def _bc(s,idx,cols):
    return pd.DataFrame(np.tile(s.reindex(idx,method='ffill').values[:,None],(1,len(cols))),index=idx,columns=cols)
features={**cm_data,**{k:_bc(v,prices.index,prices.columns) for k,v in macro.items()}}
if mcap is not None: features['MCAP']=mcap
volume=volume.reindex(index=prices.index,columns=prices.columns)
universe=(mcap>=10_000_000).fillna(False) if mcap is not None else None
specs={**{k:FieldSpec(lag=1,missing='ffill',max_staleness=7) for k in cm_data},
       **{k:FieldSpec(lag=1,missing='ffill',max_staleness=3) for k in macro},
       'MCAP':FieldSpec(lag=0,missing='ffill',max_staleness=7),
       'volume':FieldSpec(lag=0,missing='ffill',max_staleness=3)}
panel=DataPanel(prices,volume=volume,features=features,universe=universe,specs=specs,check_outliers=True)
print(f'Panel: {len(panel.dates)} bars  {panel.dates[0].date()} to {panel.dates[-1].date()}')

FACTOR_COLS=['AA','NT','MRR','SPX','VIX','USDX','GLD','WTI','RMRF','SMB','UMD']
RETURN_FACTORS=frozenset({'RMRF','SMB','UMD'})
OLS3_COLS=['MRR','AA','NT']
NN_LAYERS={'nn1':(32,),'nn2':(32,16),'nn3':(32,16,8),'nn4':(32,16,8,4),'nn5':(32,16,8,4,2)}

def _make_model(mt):
    mt=mt.lower()
    if mt in ('ols','ols3'): return LinearRegression()
    if mt=='pls': return PLSRegression(n_components=3)
    if mt=='lasso': return Lasso(alpha=1e-3,max_iter=5000)
    if mt=='ridge': return Ridge(alpha=1.0)
    if mt=='enet': return ElasticNet(alpha=1e-3,l1_ratio=0.5,max_iter=5000)
    if mt=='rf': return RandomForestRegressor(n_estimators=200,max_depth=6,min_samples_leaf=5,n_jobs=-1,random_state=42)
    if mt=='gbrt': return GradientBoostingRegressor(n_estimators=200,max_depth=4,learning_rate=0.05,subsample=0.8,random_state=42)
    if mt=='knn': return KNeighborsRegressor(n_neighbors=5,weights='distance',n_jobs=-1)
    if mt in NN_LAYERS: return MLPRegressor(hidden_layer_sizes=NN_LAYERS[mt],max_iter=500,early_stopping=True,validation_fraction=0.1,random_state=42,learning_rate_init=1e-3)

class CryptoMLFactorStrategy(Strategy):
    rebalance_frequency='W-SUN'
    def __init__(self,model_type='rf',min_train_samples=100,n_deciles=10,gross_exposure=0.5):
        self.model_type=model_type.lower(); self.min_train_samples=min_train_samples
        self.n_deciles=n_deciles; self.gross_exposure=gross_exposure
        self.feature_cols_=OLS3_COLS if self.model_type=='ols3' else FACTOR_COLS
        self._pipeline=None
    def __repr__(self):
        return f'CryptoMLFactorStrategy(model_type={self.model_type!r},n_deciles={self.n_deciles},gross_exposure={self.gross_exposure})'
    def required_data(self): return FACTOR_COLS+['volume','MCAP']
    def _to_weekly(self,dm):
        out={}
        for col,df in dm.items():
            if col in RETURN_FACTORS: out[col]=df.resample('W-SUN').sum()
            else:
                w=df.resample('W-SUN').last().pct_change(); out[col]=w.replace([np.inf,-np.inf],np.nan)
        return out
    def _build_Xy(self,weekly,prices):
        fwdret=prices.resample('W-SUN').last().pct_change().shift(-1)
        ref=weekly[self.feature_cols_[0]].index; weeks=sorted(set(fwdret.index[:-1])&set(ref))
        Xs,ys=[],[]
        for dt in weeks:
            y_row=fwdret.loc[dt].dropna()
            if len(y_row)<3: continue
            assets=y_row.index
            X_cs=pd.DataFrame({col:weekly[col].loc[dt] if dt in weekly[col].index
                               else pd.Series(np.nan,index=prices.columns)
                               for col in self.feature_cols_}).loc[assets]
            valid=assets[X_cs.notna().any(axis=1)&y_row.notna()]
            if len(valid)<3: continue
            Xs.append(X_cs.loc[valid].to_numpy(dtype=float,na_value=np.nan))
            ys.append(y_row.loc[valid].to_numpy(dtype=float,na_value=np.nan))
        if not Xs: return np.empty((0,len(self.feature_cols_))),np.empty(0)
        return np.vstack(Xs),np.concatenate(ys)
    def fit(self,data):
        daily={col:data.feature(col) for col in self.feature_cols_}; weekly=self._to_weekly(daily)
        n_weeks=weekly[self.feature_cols_[0]].dropna(how='all').shape[0]
        if n_weeks<self.min_train_samples: self._pipeline=None; return
        X,y=self._build_Xy(weekly,data.prices)
        if len(X)==0: self._pipeline=None; return
        base=_make_model(self.model_type)
        if isinstance(base,PLSRegression):
            nc=min(3,len(self.feature_cols_),X.shape[1]); base=PLSRegression(n_components=nc)
        self._pipeline=Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler()),('model',base)])
        with warnings.catch_warnings(): warnings.simplefilter('ignore'); self._pipeline.fit(X,y)
    def generate_weights(self,data,t):
        if self._pipeline is None: return pd.Series(dtype=float)
        daily={col:data.feature(col) for col in self.feature_cols_}; weekly=self._to_weekly(daily)
        snap={col:weekly[col].iloc[-1] if not weekly[col].empty
              else pd.Series(np.nan,index=data.prices.columns) for col in self.feature_cols_}
        X_now=pd.DataFrame(snap); valid=X_now.index[X_now.notna().any(axis=1)]
        if len(valid)<self.n_deciles: return pd.Series(dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            preds=self._pipeline.predict(X_now.loc[valid].to_numpy(dtype=float,na_value=np.nan)).ravel()
        pred_s=pd.Series(preds,index=valid); ranks=pred_s.rank(pct=True)
        lo=1.0/self.n_deciles; hi=(self.n_deciles-1)/self.n_deciles
        long_=ranks[ranks>=hi].index; short_=ranks[ranks<=lo].index
        mcap=data.feature('MCAP').iloc[-1]
        def _vw(idx):
            cap=mcap.loc[idx].clip(lower=0).fillna(0); tot=cap.sum()
            return cap/tot if tot>0 else pd.Series(1.0/len(idx),index=idx)
        w=pd.Series(0.0,index=valid)
        if len(long_): w[long_]=_vw(long_)*self.gross_exposure
        if len(short_): w[short_]=-_vw(short_)*self.gross_exposure
        return w
    def apply_risk(self,proposed,state,data): return proposed

costs=CompositeCostModel([Commission(bps=10),Spread(half_bps=15),
                          MarketImpact(k=10,kind='sqrt',adv_lookback=20),
                          ShortBorrow(annual_bps=300,trading_days=252)])
liq=LiquidityCap(cap_pct=0.05,adv_lookback=20)
splitter=CombinatorialPurgedCV(n_splits=6,n_test_groups=2,purge_bars=7,embargo_pct=0.005)

print('Running MultiPathEngine...')
mp=MultiPathEngine(panel,lambda:CryptoMLFactorStrategy(model_type='rf'),
                   splitter,costs=costs,liquidity=liq,initial_capital=1_000_000)
res=mp.run(n_jobs=1)

# ── PATH 4 DIAGNOSTICS ────────────────────────────────────────────────────────
import itertools

n=len(panel.dates)
bounds=np.linspace(0,n,7,dtype=int)
group_edges=[(panel.dates[bounds[i]],panel.dates[bounds[i+1]-1]) for i in range(6)]
combos=list(itertools.combinations(range(6),2))

# Which splits feed path_4?
# path_4 index: for each group g, find the 5th (index=4) split that tests g
path4_splits={}
for g in range(6):
    splits_with_g=[i for i,c in enumerate(combos) if g in c]
    path4_splits[g]=splits_with_g[4] if len(splits_with_g)>4 else splits_with_g[-1]

print()
print('=== PATH 4 SPLIT STRUCTURE (temporal direction) ===')
for g,(gs,ge) in enumerate(group_edges):
    si=path4_splits[g]
    combo=combos[si]
    train_groups=[j for j in range(6) if j not in combo]
    train_start=group_edges[train_groups[0]][0]
    train_end=group_edges[train_groups[-1]][1]
    direction='FUTURE TRAIN' if train_start>gs else 'historical  '
    print(f'  Group {g} test={gs.date()} to {ge.date()} | split {si:2d} trained on groups {train_groups} '
          f'({train_start.date()} to {train_end.date()}) <- {direction}')

print()
p4_ret=res.path_returns['path_4'].dropna()
p4_eq =res.path_equity['path_4']
print('=== PATH 4 OVERALL ===')
ann_r=(1+p4_ret).prod()**(252/len(p4_ret))-1
ann_v=p4_ret.std()*252**0.5
print(f'Ann return : {ann_r:+.1%}')
print(f'Ann vol    : {ann_v:.1%}')
print(f'Sharpe     : {ann_r/ann_v:.2f}')
print(f'Max DD     : {(p4_eq/p4_eq.cummax()-1).min():.1%}')

print()
print('=== PER-GROUP RETURNS in path_4 ===')
for g,(gs,ge) in enumerate(group_edges):
    seg=p4_ret.loc[gs:ge]
    si=path4_splits[g]; combo=combos[si]
    train_groups=[j for j in range(6) if j not in combo]
    train_end=group_edges[train_groups[-1]][1]
    direction='[FUTURE TRAIN]' if train_end>ge else '[hist train] '
    if len(seg)==0:
        print(f'  Group {g}: no data'); continue
    cum=(1+seg).prod()-1
    ann=((1+seg).prod()**(252/max(len(seg),1))-1)
    sr=seg.mean()/seg.std()*252**0.5 if seg.std()>0 else 0
    print(f'  Group {g} ({gs.date()} to {ge.date()}): cum={cum:+.0%}  ann={ann:+.0%}  SR={sr:.1f}  {direction}')

print()
print('=== COMPARE: paths 0-3 (no future-train contamination) ===')
for col in ['path_0','path_1','path_2','path_3']:
    ret=res.path_returns[col].dropna()
    eq=res.path_equity[col]
    ann_r=(1+ret).prod()**(252/len(ret))-1
    ann_v=ret.std()*252**0.5
    sr=ann_r/ann_v if ann_v>0 else 0
    mdd=(eq/eq.cummax()-1).min()
    final=eq.iloc[-1]
    print(f'  {col}: ann={ann_r:+.1%}  vol={ann_v:.0%}  SR={sr:.2f}  maxDD={mdd:.0%}  final=${final:,.0f}')

print()
print('=== PATH 4 by sub-period (future-trained vs hist-trained) ===')
# groups 0-3 are future-trained; groups 4-5 use split 14 (trained on 0-3 = historical)
cutoff=group_edges[3][1]   # end of group 3 = 2022-06-27
p4_early=p4_ret.loc[:cutoff]
p4_late =p4_ret.loc[cutoff:]
for label,seg in [('Groups 0-3 (future-trained)', p4_early),
                  ('Groups 4-5 (hist-trained)  ', p4_late)]:
    if len(seg)==0: continue
    cum=(1+seg).prod()-1
    ann=((1+seg).prod()**(252/max(len(seg),1))-1)
    sr=seg.mean()/seg.std()*252**0.5 if seg.std()>0 else 0
    print(f'  {label}: cum={cum:+.0%}  ann={ann:+.0%}  SR={sr:.1f}  ({len(seg)} obs)')

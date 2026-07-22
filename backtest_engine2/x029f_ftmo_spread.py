"""#029f - pull REAL FTMO per-bar spread structure for all 28 pairs from the MT5 demo (read-only, ticks;
NO orders). Short tick history (~10 days) is enough to characterize each pair's half-spread DISTRIBUTION
and its HOUR-OF-DAY profile (FX spreads are tight in liquid hours, wide at rollover). Output: a
per-pair x UTC-hour median half-spread table -> map onto the 2019-2026 M15 bars to run the gate on
realistic FTMO spreads. Resumable: appends per pair, skips pairs already saved."""
import os, sys, time as _t
from datetime import datetime, timedelta
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

PAIRS = ['EURUSD','GBPUSD','AUDUSD','NZDUSD','USDCAD','USDCHF','USDJPY','EURGBP','EURAUD','EURJPY',
         'GBPJPY','AUDJPY','EURNZD','EURCAD','EURCHF','GBPAUD','GBPNZD','GBPCAD','GBPCHF','AUDNZD',
         'AUDCAD','AUDCHF','NZDCAD','NZDCHF','NZDJPY','CADCHF','CADJPY','CHFJPY']
DAYS = 10
OUT = r'C:\Users\User\backtest_engine\backtest_engine2\data\ftmo_spread_profile.parquet'

done = set()
if os.path.exists(OUT):
    done = set(pd.read_parquet(OUT)['pair'].unique())
    print(f'resuming, {len(done)} pairs already saved')

if not mt5.initialize():
    print('INIT FAIL:', mt5.last_error()); sys.exit(1)
print('connected:', mt5.account_info().server)

rows = []
for sym in PAIRS:
    if sym in done:
        continue
    if mt5.symbol_info(sym) is None:
        print(f'  {sym}: no symbol'); continue
    mt5.symbol_select(sym, True)
    _t.sleep(0.5)                                          # let the symbol sync before requesting ticks
    tick = mt5.symbol_info_tick(sym)
    off_h = round((tick.time - _t.time()) / 3600)          # server tz offset from UTC
    snow = datetime.utcfromtimestamp(tick.time)
    ticks = None
    for attempt in range(3):
        try:
            ticks = mt5.copy_ticks_range(sym, snow - timedelta(days=DAYS), snow + timedelta(hours=2),
                                         mt5.COPY_TICKS_INFO)
            break
        except Exception as ex:
            print(f'  {sym}: retry {attempt} ({ex})', flush=True); _t.sleep(1.0)
    if ticks is None or len(ticks) == 0:
        print(f'  {sym}: no ticks ({mt5.last_error()})'); continue
    df = pd.DataFrame(ticks)
    df = df[(df.bid > 0) & (df.ask > 0) & (df.ask >= df.bid)]
    mid = (df.ask + df.bid) / 2
    df = df.assign(half_bp=(df.ask - df.bid) / mid * 1e4 / 2,
                   utc_hr=pd.to_datetime(df.time - off_h * 3600, unit='s', utc=True).dt.hour,
                   dow=pd.to_datetime(df.time - off_h * 3600, unit='s', utc=True).dt.dayofweek)
    df = df[df.dow < 5]
    for hh in range(24):
        x = df[df.utc_hr == hh]['half_bp']
        if len(x) < 20:
            continue
        rows.append(dict(pair=sym, hour=hh, med_half_bp=float(x.median()),
                         p_le_010=float((x <= 0.10).mean()), p_le_020=float((x <= 0.20).mean()),
                         p_le_030=float((x <= 0.30).mean()), n=int(len(x))))
    # incremental save (resume-safe against the env's background sweeps)
    cur = pd.DataFrame(rows)
    if os.path.exists(OUT):
        cur = pd.concat([pd.read_parquet(OUT), cur], ignore_index=True).drop_duplicates(['pair', 'hour'], keep='last')
    cur.to_parquet(OUT, index=False)
    rows = []
    med = df['half_bp'].median()
    print(f'  {sym}: {len(df):,} ticks, overall median half {med:.3f}bp, '
          f'P(<=0.2bp) {(df.half_bp<=0.2).mean()*100:.0f}%', flush=True)

mt5.shutdown()
prof = pd.read_parquet(OUT)
print(f'\nprofile saved: {prof.pair.nunique()} pairs x hours -> {OUT}')

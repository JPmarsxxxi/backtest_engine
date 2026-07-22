"""
#027 — Pull a broad FX cross-section at M15 for intraday statistical arbitrage.
Reuses the Dukascopy 1-min BID puller pattern (_pull_fx_h1) but resamples to 15-min
(same download cost) and loops a 12-pair cross-section. Saves per-pair m15 parquets +
a combined pivoted close panel data/fx_intraday_m15.parquet.

12 pairs chosen for factor structure: 7 USD majors (PC1 = dollar factor) + 5 crosses
(JPY / EUR / commodity-FX blocs) -> enough breadth for a real PCA residual book.
"""
import lzma
import os
import struct
import sys
import time

import httpx
import pandas as pd

# sym -> price scale (5-digit non-JPY = 1e5 ; 3-digit JPY = 1e3)
PAIRS = {
    'EURUSD': 1e5, 'GBPUSD': 1e5, 'AUDUSD': 1e5, 'NZDUSD': 1e5,
    'USDCAD': 1e5, 'USDCHF': 1e5, 'USDJPY': 1e3,
    'EURGBP': 1e5, 'EURAUD': 1e5,
    'EURJPY': 1e3, 'GBPJPY': 1e3, 'AUDJPY': 1e3,
}
START = pd.Timestamp('2019-01-01')
END = pd.Timestamp('2026-06-13')
BAR = '15min'
OUTDIR = r'C:\Users\User\backtest_engine\backtest_engine2\data'
FEED = os.environ.get('DUKA_FEED', 'BID')          # BID (default) or ASK
SUF = '' if FEED == 'BID' else '_' + FEED.lower()  # ASK saved to *_ask_m15.parquet


def fetch(client, url):
    """Return response content, retrying on timeout / 503 (rate-limit) with backoff. None if genuinely unavailable."""
    for attempt in range(6):
        try:
            r = client.get(url, timeout=15)
            if r.status_code == 200 and len(r.content) > 0:
                return r.content
            # 503 / empty = server throttling or no-data -> back off and retry
        except Exception:
            pass
        time.sleep(min(2.0 * (attempt + 1), 8.0))   # 2,4,6,8,8,8s backoff
    return None


def pull_one(sym, scale, client):
    base = f'https://datafeed.dukascopy.com/datafeed/{sym}'
    frames = []
    ok = empty = miss = 0
    for day in pd.date_range(START, END, freq='D'):
        if day.weekday() >= 5:   # skip weekends (no FX data)
            continue
        url = f'{base}/{day.year}/{day.month - 1:02d}/{day.day:02d}/{FEED}_candles_min_1.bi5'
        content = fetch(client, url)
        if content is None:
            miss += 1; continue
        try:
            raw = lzma.decompress(content)
        except lzma.LZMAError:
            empty += 1; continue
        n = len(raw) // 24
        recs = [struct.unpack('>5if', raw[i * 24:(i + 1) * 24]) for i in range(n)]
        df = pd.DataFrame(recs, columns=['sec', 'open', 'close', 'low', 'high', 'vol'])
        df = df[df['vol'] > 0]
        if df.empty:
            empty += 1; continue
        df['open_time'] = day.tz_localize('UTC') + pd.to_timedelta(df['sec'], unit='s')
        df['close'] = df['close'] / scale
        frames.append(df[['open_time', 'close', 'vol']])
        ok += 1
        time.sleep(0.15)   # politeness: stay under Dukascopy's rate limiter
    if not frames:
        print(f'  {sym}: NO DATA (ok={ok} empty={empty} miss={miss})', flush=True)
        return None
    m1 = pd.concat(frames, ignore_index=True).set_index('open_time').sort_index()
    bars = m1.resample(BAR).agg({'close': 'last', 'vol': 'sum'}).dropna(subset=['close']).reset_index()
    bars = bars.rename(columns={'vol': 'quote_volume'})
    bars['symbol'] = sym
    bars.to_parquet(rf'{OUTDIR}\{sym.lower()}{SUF}_m15.parquet', index=False)
    print(f'  {sym}: {len(bars)} M15 bars ({ok} days, empty {empty}, miss {miss})', flush=True)
    return bars[['open_time', 'close']].rename(columns={'close': sym}).set_index('open_time')


def build_panel():
    """(Re)build the combined panel from ALL saved per-pair m15 files (works across resumed batches)."""
    cols = []
    for sym in PAIRS:
        f = rf'{OUTDIR}\{sym.lower()}{SUF}_m15.parquet'
        if os.path.exists(f):
            d = pd.read_parquet(f)[['open_time', 'close']].rename(columns={'close': sym}).set_index('open_time')
            cols.append(d)
    if not cols:
        print('no per-pair files yet'); return
    panel = pd.concat(cols, axis=1).sort_index()
    panel.to_parquet(rf'{OUTDIR}\fx_intraday{SUF}_m15.parquet')
    print(f'\nPANEL {panel.shape} ({len(cols)}/{len(PAIRS)} pairs) -> data/fx_intraday_m15.parquet')
    print(f'coverage {panel.index.min()} -> {panel.index.max()}')
    print((panel.notna().mean() * 100).round(1).to_string())


def main():
    # optional symbol list on argv restricts the pull (for small resumable batches)
    want = [s.upper() for s in sys.argv[1:]] if len(sys.argv) > 1 else list(PAIRS)
    client = httpx.Client(timeout=15)
    for sym in want:
        f = rf'{OUTDIR}\{sym.lower()}{SUF}_m15.parquet'
        if os.path.exists(f):
            print(f'{sym}: already on disk, skip', flush=True); continue
        print(f'pulling {sym} ...', flush=True)
        pull_one(sym, PAIRS[sym], client)
    build_panel()


if __name__ == '__main__':
    main()

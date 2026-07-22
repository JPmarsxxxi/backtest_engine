"""#030 - pull 1-MINUTE OHLC (keep high/low, not just close) for the tight-spread majors, so we can
test LIMIT-ORDER fills (did intrabar price trade through a limit level?). Raw Dukascopy .bi5 candles are
already 1-min OHLC (order in file: open, close, low, high) - the M15 puller discarded H/L; here we keep
them. BID and ASK (via DUKA_FEED). Per-pair-feed parquet {pair}_{feed}_1m.parquet. Resume-safe (skip on
disk). One pair via argv for parallel single-task jobs (beats the env's background sweeps).

Fill model this enables: limit BUY at L fills if ASK low <= L; limit SELL at L fills if BID high >= L.
"""
import lzma, os, struct, sys, time
import httpx
import pandas as pd

MAJORS = {'EURUSD': 1e5, 'USDJPY': 1e3, 'GBPUSD': 1e5, 'USDCAD': 1e5, 'AUDUSD': 1e5, 'USDCHF': 1e5}
START = pd.Timestamp('2023-01-01')
END = pd.Timestamp('2026-06-13')
OUTDIR = r'C:\Users\User\backtest_engine\backtest_engine2\data'
FEED = os.environ.get('DUKA_FEED', 'BID')
SUF = FEED.lower()


def fetch(client, url):
    for a in range(6):
        try:
            r = client.get(url, timeout=15)
            if r.status_code == 200 and len(r.content) > 0:
                return r.content
        except Exception:
            pass
        time.sleep(min(2.0 * (a + 1), 8.0))
    return None


def pull_one(sym, scale, client):
    base = f'https://datafeed.dukascopy.com/datafeed/{sym}'
    frames = []; ok = empty = miss = 0
    for day in pd.date_range(START, END, freq='D'):
        if day.weekday() >= 5:
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
        for c in ('open', 'high', 'low', 'close'):
            df[c] = df[c] / scale
        frames.append(df[['open_time', 'open', 'high', 'low', 'close']])
        ok += 1
        time.sleep(0.1)
    if not frames:
        print(f'  {sym}: NO DATA'); return
    out = pd.concat(frames, ignore_index=True).sort_values('open_time')
    out.to_parquet(rf'{OUTDIR}\{sym.lower()}_{SUF}_1m.parquet', index=False)
    print(f'  {sym} {FEED}: {len(out):,} 1-min bars ({ok} days, miss {miss})', flush=True)


def main():
    want = [s.upper() for s in sys.argv[1:]] if len(sys.argv) > 1 else list(MAJORS)
    client = httpx.Client(timeout=15)
    for sym in want:
        f = rf'{OUTDIR}\{sym.lower()}_{SUF}_1m.parquet'
        if os.path.exists(f):
            print(f'{sym} {FEED}: on disk, skip', flush=True); continue
        print(f'pulling {sym} {FEED} 1-min ...', flush=True)
        pull_one(sym, MAJORS.get(sym, 1e3 if sym.endswith('JPY') else 1e5), client)


if __name__ == '__main__':
    main()

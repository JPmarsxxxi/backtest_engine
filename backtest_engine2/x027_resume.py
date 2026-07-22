"""
#027 resume: wait out the Dukascopy rate-limit (503/timeout), then pull the remaining M15 pairs.
Probes once/min until the feed answers 200, then pulls only pairs not already on disk (politely).
Self-contained + resumable: if killed again, just relaunch — banked pairs are skipped.
"""
import lzma
import sys
import time

import httpx
import pandas as pd

import x027_pull_fx_intraday as P

PROBE = ('https://datafeed.dukascopy.com/datafeed/USDCHF/2024/02/06/'
         'BID_candles_min_1.bi5')   # a known-good weekday file (month is 0-indexed -> 02 = March)


def wait_clear(max_min=90):
    for i in range(max_min):
        try:
            r = httpx.get(PROBE, timeout=15)
            if r.status_code == 200 and len(r.content) > 100:
                lzma.decompress(r.content)               # confirm real data, not an error page
                print(f'[{i}m] throttle CLEAR (status 200, {len(r.content)}b) -> pulling', flush=True)
                return True
            print(f'[{i}m] still throttled: status {r.status_code}, {len(r.content)}b', flush=True)
        except Exception as e:
            print(f'[{i}m] still throttled: {type(e).__name__}', flush=True)
        time.sleep(60)
    print(f'still throttled after {max_min}m, giving up', flush=True)
    return False


def main():
    if not wait_clear():
        sys.exit(1)
    client = httpx.Client(timeout=15)
    import os
    for sym in P.PAIRS:
        f = rf'{P.OUTDIR}\{sym.lower()}_m15.parquet'
        if os.path.exists(f):
            print(f'{sym}: banked, skip', flush=True); continue
        print(f'pulling {sym} ...', flush=True)
        P.pull_one(sym, P.PAIRS[sym], client)
    P.build_panel()


if __name__ == '__main__':
    main()

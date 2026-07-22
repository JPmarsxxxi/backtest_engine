"""#030b - LIMIT-ORDER fill UPPER BOUND for the 1-bar FX reversion signal (majors, 1-min OHLC).
Question: does posting a LIMIT entry (capture spread instead of paying it) beat MARKET entry, once
adverse selection is honestly included? Fill = intrabar price traded THROUGH the limit level (optimistic
on queue: assumes touch=fill; realistic on adverse selection: fills are conditioned on price reaching you).

Signal: 1-bar M15 reversion per pair, sig(t)=-sign(M15 ret) (buy the dip / sell the rip). Hold one M15 bar.
  MARKET:  buy@ask_start -> sell@bid_end (pays full spread).  Always trades.
  LIMIT:   buy limit L=mid_start-k*half; fills if bid_low(window)<=L (price came to you) -> buy@L, sell@bid_end.
           (sell side symmetric with ask_high>=L.) Trades only when filled.
Compare TOTAL pnl over the SAME signal stream (unfilled limit trades contribute 0). Sweep k.
If even this optimistic upper bound doesn't beat market / isn't positive -> method 1 dead, no daemon needed.
"""
import glob, os
import numpy as np
import pandas as pd

MAJORS = ['EURUSD', 'USDJPY', 'GBPUSD', 'USDCAD', 'AUDUSD', 'USDCHF']
D = r'C:\Users\User\backtest_engine\backtest_engine2\data'

def load_pair(sym):
    fb, fa = f'{D}\\{sym.lower()}_bid_1m.parquet', f'{D}\\{sym.lower()}_ask_1m.parquet'
    if not (os.path.exists(fb) and os.path.exists(fa)):
        return None
    b = pd.read_parquet(fb).set_index('open_time').sort_index()
    a = pd.read_parquet(fa).set_index('open_time').sort_index()
    idx = b.index.intersection(a.index)
    b, a = b.loc[idx], a.loc[idx]
    df = pd.DataFrame({'bid_o': b.open, 'bid_l': b.low, 'bid_c': b.close,
                       'ask_o': a.open, 'ask_h': a.high, 'ask_c': a.close}, index=idx)
    df['mid_c'] = (df.bid_c + df.ask_c) / 2
    return df

def m15_windows(df):
    """aggregate 1-min -> M15 window fields needed for entry/fill/exit."""
    g = df.groupby(pd.Grouper(freq='15min'))
    w = pd.DataFrame({
        'ask_start': g['ask_o'].first(), 'bid_start': g['bid_o'].first(),
        'bid_low': g['bid_l'].min(), 'ask_high': g['ask_h'].max(),
        'bid_end': g['bid_c'].last(), 'ask_end': g['ask_c'].last(),
        'mid_close': g['mid_c'].last(), 'n': g['mid_c'].count(),
    }).dropna()
    w = w[w.n >= 5]                                  # need a populated window
    w['mid_start'] = (w.ask_start + w.bid_start) / 2
    w['half'] = (w.ask_start - w.bid_start) / 2
    return w

def build_windows(have):
    """M15 windows per pair + a shared cross-sectional reversal DIRECTION (the real pool signal,
    not crude per-pair sign): dir_i = -sign(z_i - mean_j z_j), dollar-neutral."""
    W = {s: m15_windows(load_pair(s)) for s in have}
    idx = None
    for s in have:
        idx = W[s].index if idx is None else idx.intersection(W[s].index)
    mid = pd.DataFrame({s: W[s].mid_close.reindex(idx) for s in have})
    ret = np.log(mid).diff()
    z = (ret / ret.rolling(96, min_periods=48).std()).clip(-8, 8)
    xs = z.sub(z.mean(axis=1), axis=0)              # cross-sectional demean
    direction = -np.sign(xs)                        # reversal: fade the relative move
    return W, idx, direction

def test_pair(sym, k, W, idx, direction):
    w = W[sym].reindex(idx)
    d = direction[sym].reindex(idx)                 # real cross-sectional reversal direction
    Wn = w.shift(-1)                                # entry-window fields aligned to decision t
    valid = d.notna() & (d != 0) & Wn.mid_start.notna()
    d = d[valid]; Wn = Wn[valid]
    half = Wn.half
    # MARKET: buy@ask_start / sell@bid_start ; exit sell@bid_end / buy@ask_end
    mkt = np.where(d > 0, Wn.bid_end - Wn.ask_start, Wn.bid_start - Wn.ask_end)
    # LIMIT: buy L=mid_start-k*half fills if bid_low<=L ; sell L=mid_start+k*half fills if ask_high>=L
    Lb = Wn.mid_start - k * half; La = Wn.mid_start + k * half
    fill_b = (d > 0) & (Wn.bid_low <= Lb)
    fill_s = (d < 0) & (Wn.ask_high >= La)
    filled = fill_b | fill_s
    lim = np.where(fill_b, Wn.bid_end - Lb, np.where(fill_s, La - Wn.ask_end, 0.0))
    # GROSS mid-to-mid (ZERO spread) = the signal's underlying edge; if ~0, no execution trick helps
    gross = d.values * (Wn.mid_close - Wn.mid_start)
    scale = 1e4 / Wn.mid_start                       # -> bp
    return dict(mkt_bp=(mkt * scale), lim_bp=(lim * scale), gross_bp=(gross * scale),
                filled=filled.values, n=len(d), fillrate=filled.mean())

def run():
    have = [s for s in MAJORS if load_pair(s) is not None]
    print(f'majors with 1-min bid+ask on disk: {have}\n')
    if not have:
        print('no data yet'); return
    W, idx, direction = build_windows(have)
    for k in (0.0, 0.5, 1.0, 1.5, 2.0):
        allm, alll, allg, allf, tot = [], [], [], [], 0
        for s in have:
            r = test_pair(s, k, W, idx, direction)
            if r is None: continue
            allm.append(r['mkt_bp']); alll.append(r['lim_bp']); allg.append(r['gross_bp'])
            allf.append(r['filled']); tot += r['n']
        m = np.concatenate(allm); l = np.concatenate(alll); gr = np.concatenate(allg); f = np.concatenate(allf)
        print(f'k={k:.1f} (limit mid-{k:.1f}*half): fill {f.mean()*100:4.1f}%  '
              f'GROSS(no-spread) {np.nanmean(gr):+.3f}  MARKET {np.nanmean(m):+.3f}  '
              f'LIMIT {np.nanmean(l):+.3f} bp/sig  n={tot:,}')
    print('\nread: LIMIT mean/sig > MARKET mean/sig => limit orders help (upper bound). '
          'k=0 posts at mid (save half-spread); k>=1 posts at/below bid (capture full spread, lower fill).')

if __name__ == '__main__':
    run()

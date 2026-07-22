# Close the loop: does the champion 200d regime filter rescue the OVERNIGHT strategy,
# tested over the LONG history (1990-2026) that contains real bears (2000/2008/2020/2022)?
# Overnight return = prev close -> today open (reconstructed from daily OHLC, long history).
# Filter: hold overnight only when S&P > its 200d average (PIT, known at prev close).
import json, time
import numpy as np, pandas as pd
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 research"}
def get(url, tries=4, timeout=60):
    last=None
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read().decode("utf-8","replace")
        except Exception as e:
            last=e; print(f"  retry {k+1} {type(e).__name__}"); time.sleep(2)
    raise last
def yahoo_ohlc(sym, period1="-631152000"):
    for host in ("query1","query2"):
        try:
            j=json.loads(get(f"https://{host}.finance.yahoo.com/v8/finance/chart/{sym}?period1={period1}&period2=9999999999&interval=1d"))
            res=j["chart"]["result"][0]; ts=pd.to_datetime(res["timestamp"],unit="s").normalize()
            q=res["indicators"]["quote"][0]
            return pd.DataFrame({"open":q["open"],"close":q["close"]},index=ts).dropna()
        except Exception as e:
            print(f"  {host} fail {sym}: {type(e).__name__}")
    raise RuntimeError(sym)

df=yahoo_ohlc("%5EGSPC")
print(f"SPX OHLC {df.index.min().date()} -> {df.index.max().date()}  n={len(df)}")
df=df[df.index>="1990-01-01"].copy()
df["overnight"]=df["open"]/df["close"].shift(1)-1        # prev close -> today open
df["intraday"]=df["close"]/df["open"]-1                  # today open -> today close
df["full"]=df["close"].pct_change(fill_method=None)
df["sma200"]=df["close"].rolling(200).mean()
df["regime_on"]=(df["close"]>df["sma200"]).shift(1)      # decision at prev close governs tonight
df=df.dropna(subset=["overnight","intraday","sma200","regime_on"])
df["yr"]=df.index.year

def stats(r):
    r=r.dropna(); eq=(1+r).cumprod(); dd=(eq/eq.cummax()-1).min()
    return (eq.iloc[-1]**(252/len(r))-1)*100, r.mean()/r.std()*np.sqrt(252), dd*100
def line(nm,r): c,s,d=stats(r); print(f"  {nm:34s} {c:6.1f} {s:7.2f} {d:7.1f}")

print(f"\n=== Overnight strategy, 1990-2026 (incl. 2000/2008/2020/2022) ===")
print(f"  {'strategy':34s} {'CAGR%':>6s} {'Sharpe':>7s} {'maxDD%':>7s}")
line("buy & hold (full day)", df["full"])
line("overnight, ALWAYS (no filter)", df["overnight"])
line("overnight, 200d-FILTERED", np.where(df["regime_on"],df["overnight"],0.0)*1.0 if False else df["regime_on"].astype(float)*df["overnight"])
line("intraday only (for contrast)", df["intraday"])

print("\nCrisis test — calendar-year return, overnight ALWAYS vs 200d-FILTERED:")
for yr in [2000,2001,2002,2008,2020,2022]:
    g=df[df.yr==yr]
    if len(g)==0: continue
    al=(1+g["overnight"]).prod()-1
    fl=(1+g["regime_on"].astype(float)*g["overnight"]).prod()-1
    print(f"  {yr}:  always {al*100:+6.1f}%   filtered {fl*100:+6.1f}%   (in-market {100*g['regime_on'].mean():.0f}% of nights)")

print("\nNote: index overnight = GROSS (no swap). Live CFD overnight pays ~5.8%/yr swap unless the")
print("swap-dodge timing is used; this test isolates whether the FILTER helps across real bears.")
